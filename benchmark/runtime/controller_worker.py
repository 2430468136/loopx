"""Two native Codex threads serving one benchmark Turn through the v1 Host ABI.

The product owns admission, independent validation, recovery and settlement.
This disposable adapter owns only native IO and its private attempt evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from loopx.capabilities.native_chat.codex_context import disable_mcp_servers
from loopx.control_plane.effect_runtime import effect_runtime_result
from loopx.control_plane.goals.first_party_host_admission import FirstPartyHostGoalAdmission
from loopx.control_plane.quota.turn_envelope import COMPLETE_TURN_HOST_REQUEST_SCHEMA
from loopx.control_plane.runtime.document_io import atomic_write_private_json
from loopx.control_plane.turn_driver.codex_cli import normalize_codex_result, prepare_codex_cli_input
from loopx.control_plane.turn_driver.codex_stdio import StdioNativeGoalTransport
from loopx.control_plane.turn_driver.executor import MATERIAL_HOST_RESULT_KINDS
from loopx.control_plane.turn_driver.host_candidate import extract_turn_work_identity
from loopx.control_plane.turn_driver.journal_store import turn_journal_path
from loopx.file_lock import try_exclusive_file_lock


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def object_schema(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


WORK_REPORT = object_schema({"finding": {"type": "string"},
                             "artifacts": {"type": "array", "items": {"type": "string"}}})


def role_policy(role, project, env, capture):
    # Start from the same workspace-only native permission owner as project
    # conversations. Preserve the guest's task root and exclude Core state.
    writable = role == "worker" and env["LOOPX_CODEX_SANDBOX"] != "read-only"
    context = {
        "kind": "project_workspace", "workspace_path": str(project), "project_ref": digest(str(project))[:24],
        "audience": "local_owner", "grant": "workspace_write" if writable else "workspace_read",
        "filesystem_scope": "workspace_only",
    }
    identity = effect_runtime_result("collaboration.project.session_identity", {"context": context})
    profile = identity["permissions_profile"]
    config = identity["host_config"]
    filesystem = config["permissions"][profile]["filesystem"]
    fs = filesystem[":workspace_roots"]
    for name in (".codex", ".loopx", ".git", "ACTIVE_GOAL_STATE.md"):
        fs[name] = "deny"
    for key in ("LOOPX_RUNTIME_ROOT", "LOOPX_REGISTRY", "CODEX_HOME", "LOOPX_SHARED_SKILLS", "LOOPX_WAKE_LOG_DIR"):
        path = Path(env[key]).resolve()
        if project.is_relative_to(path):
            raise ValueError("worker task root overlaps private trial state")
        if path.is_relative_to(project):
            fs[str(path.relative_to(project))] = "deny"
        else:
            filesystem[str(path)] = "deny"
    if role == "controller" and capture is not None:
        filesystem[str(capture)] = "read"
    if role == "controller" and env.get("LOOPX_TASK_DOC"):
        # The benchmark operator stages the public task separately on some hosts.
        task = Path(env["LOOPX_TASK_DOC"]).resolve(strict=True)
        if task.is_file() and not task.is_relative_to(project):
            filesystem[str(task)] = "read"
    return profile, config


class RoleTransport:
    def __init__(self, rpc, project, env, deadline, admission, evidence, save, capture=None):
        self.rpc, self.project, self.env = rpc, project, env
        self.deadline, self.admission = deadline, admission
        self.evidence, self.save = evidence, save
        self.capture = capture
        self.threads = {}
        self.permissions = {}

    def remaining(self):
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("controller-worker Turn deadline")
        return seconds

    def request(self, method, params):
        self.rpc.response_timeout_sec = min(30, self.remaining())
        return self.rpc.request(method, params)

    def start(self, role):
        cwd = str(self.project)
        effective = self.request("config/read", {"cwd": cwd, "includeLayers": False})["config"]
        config = {"web_search": "disabled", "project_doc_max_bytes": 0, "skills": {"include_instructions": False}}
        params = {"cwd": cwd, "runtimeWorkspaceRoots": [cwd], "model": self.env["MODEL_NAME"],
                  "approvalPolicy": "never"}
        profile, policy = role_policy(role, self.project, self.env, self.capture)
        config.update(policy)
        self.permissions[role] = profile
        params["permissions"] = profile
        config["features"] = {"apps": False, "multi_agent": False, "multi_agent_v2": False,
                              "hooks": False, "browser_use": False, "computer_use": False,
                              "goals": False, "js_repl": False}
        params["config"] = disable_mcp_servers(effective, config)
        with self.admission.launch_lifetime(operation="benchmark_role_thread_start"):
            observed = self.request("thread/start", params)
        thread = observed["thread"]["id"]
        if (not isinstance(thread, str) or not thread or thread in self.threads.values()
                or observed.get("approvalPolicy") != "never" or observed.get("runtimeWorkspaceRoots") != [cwd]
                or (observed.get("activePermissionProfile") or {}).get("id") != profile):
            raise RuntimeError("native role identity or permissions unconfirmed")
        self.threads[role] = thread
        self.evidence["threads"][role] = thread
        self.save()

    def send(self, role, prompt, schema):
        if role not in self.threads:
            self.start(role)
        thread = self.threads[role]
        params = {"threadId": thread, "cwd": str(self.project),
                  "input": [{"type": "text", "text": prompt}], "outputSchema": schema,
                  "effort": self.env["REASONING_EFFORT"], "approvalPolicy": "never"}
        params["permissions"] = self.permissions[role]
        began = time.monotonic()
        with self.admission.launch_lifetime(operation="benchmark_role_turn_start"):
            started = self.request("turn/start", params)
        turn = started["turn"]["id"]
        if not isinstance(turn, str) or not turn:
            raise RuntimeError("native role Turn missing identity")
        step = {"role": role, "turn_id": turn, "input_sha256": digest(prompt)}
        self.evidence["steps"].append(step)
        self.save()
        message = None
        terminal = False
        step["thread_token_usage_total"] = None
        next_check = time.monotonic() + 1
        try:
            while True:
                event = self.rpc.next_event(timeout_sec=min(1, self.remaining()))
                if time.monotonic() >= next_check:
                    with self.admission.launch_lifetime(operation="benchmark_role_continue"):
                        pass
                    next_check = time.monotonic() + 1
                if not event:
                    continue
                payload = event.get("params") or {}
                native_turn = payload.get("turn") or {}
                if payload.get("threadId") != thread or (payload.get("turnId") or native_turn.get("id")) != turn:
                    continue
                item = payload.get("item") or {}
                if event.get("method") == "item/completed" and item.get("type") == "agentMessage":
                    if item.get("phase") != "commentary":
                        message = item.get("text")
                if event.get("method") == "thread/tokenUsage/updated":
                    total = (payload.get("tokenUsage") or {}).get("total")
                    if isinstance(total, dict) and type(total.get("totalTokens")) is int:
                        step["thread_token_usage_total"] = {key: value for key, value in total.items()
                                                            if type(value) is int and value >= 0}
                if event.get("method") == "turn/completed":
                    terminal = True
                    step["native_status"] = native_turn.get("status")
                    if native_turn.get("status") != "completed" or native_turn.get("error"):
                        # Keep native failures attributable without copying backend messages.
                        error = native_turn.get("error") or {}
                        code = error.get("codexErrorInfo")
                        if isinstance(code, str):
                            step["codex_error_info"] = code
                        raise RuntimeError("native role Turn did not complete")
                    if not isinstance(message, str) or len(message.encode()) > 128 * 1024:
                        raise ValueError("missing or oversized native role result")
                    result = json.loads(message)
                    if not isinstance(result, dict):
                        raise ValueError("invalid native role result")
                    step["output_sha256"] = digest(result)
                    return result
        except BaseException:
            # Cancellation must also work while the Controller has no model
            # call running. Address the exact acknowledged native Turn.
            if not terminal:
                self.rpc.response_timeout_sec = 2
                try:
                    self.rpc.request("turn/interrupt", {"threadId": thread, "turnId": turn})
                    step["interrupt_acknowledged"] = True
                except Exception:
                    step["interrupt_acknowledged"] = False
            raise
        finally:
            step["elapsed_seconds"] = round(time.monotonic() - began, 3)
            self.save()


def run_roles(native, request):
    prompt, result_schema = prepare_codex_cli_input(request)
    decision_schema = object_schema({"instruction": {"type": ["string", "null"]},
                                     "result": {"anyOf": [result_schema, {"type": "null"}]}})
    prompt += ("\nYou are the Controller. Read the admitted contract and required material. "
               "Delegate bounded task work in instruction, including necessary task material and constraints. "
               "Keep Core protocol and private control context here. Assess the returned finding and artifacts. "
               "Give actionable feedback if another work segment is needed. Fill exactly one of instruction or result. "
               "Return the typed Turn result when ready. At most three Worker segments share this admission.")
    for index in range(4):
        decision = native.send("controller", prompt, decision_schema)
        instruction, result = decision.get("instruction"), decision.get("result")
        if result is not None:
            if instruction is not None or not isinstance(result, dict):
                raise ValueError("controller returned ambiguous decision")
            if index == 0 and result.get("result_kind") in MATERIAL_HOST_RESULT_KINDS:
                raise ValueError("controller returned a material result without a Worker return")
            return normalize_codex_result(request, result)
        if index == 3 or not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 16000:
            raise ValueError("controller exhausted delegation or returned an invalid task")
        returned = native.send("worker", "Execute this scoped task in the workspace. Return observed findings "
                               "and artifact paths.\n\n" + instruction, WORK_REPORT)
        if (not isinstance(returned.get("finding"), str) or not returned["finding"].strip()
                or not isinstance(returned.get("artifacts"), list)
                or not all(isinstance(path, str) for path in returned["artifacts"])):
            raise ValueError("worker returned an invalid report")
        prompt = ("Assess this Worker return as task evidence. Check the artifacts and decide the next "
                  "instruction or final result.\n" + json.dumps(returned, ensure_ascii=False))


@contextmanager
def interrupt_on_stop():
    def stop(*_):
        raise InterruptedError("controller-worker Host stopped")
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def serve(request, env):
    if request.get("schema_version") != COMPLETE_TURN_HOST_REQUEST_SCHEMA:
        raise ValueError("controller-worker requires a complete C1 Turn request")
    identity = extract_turn_work_identity(request)
    if (identity["goal_id"], identity["agent_id"]) != (env["LOOPX_GOAL_ID"], env["LOOPX_AGENT_ID"]):
        raise ValueError("controller-worker trial identity mismatch")
    project = Path(env["LOOPX_PROJECT"]).resolve()
    runtime = Path(env["LOOPX_RUNTIME_ROOT"]).resolve()
    key = request["turn_key"]
    journal = turn_journal_path(runtime, goal_id=identity["goal_id"], turn_key=key)
    capture = journal.parent / "decisions" / journal.name
    reference = request.get("detail_ref", {}).get("captured_decision", {})
    if reference.get("path") != str(capture) or capture.resolve(strict=True) != capture:
        raise ValueError("original Turn decision capture unavailable")
    path = runtime / "benchmark-role-attempts" / journal.name
    timeout = float(env["LOOPX_ROLE_TIMEOUT_SECONDS"])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("controller-worker requires a finite execution budget")
    deadline = time.monotonic() + timeout
    admission = FirstPartyHostGoalAdmission.for_plan(registry_path=Path(env["LOOPX_REGISTRY"]),
        goal_id=identity["goal_id"], planned_goal_ref=request.get("goal_ref"))
    with try_exclusive_file_lock(path) as lock:
        if lock is None:
            raise RuntimeError("original role attempt is still active")
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("input_sha256") == digest(request) and isinstance(prior.get("result"), dict):
                return prior["result"]
            raise RuntimeError("original role attempt requires reconciliation")
        evidence = {"turn_key": key, "input_sha256": digest(request), "threads": {}, "steps": []}
        def save():
            atomic_write_private_json(path, evidence)
            # Wakes are collected before SForge removes the guest. Keep the
            # result and retry state in the adapter's original private store.
            observation = {key: evidence[key] for key in ("turn_key", "threads", "steps")}
            atomic_write_private_json(Path(env["LOOPX_ROLE_OBSERVATION"]), observation)
        save()
        with interrupt_on_stop(), StdioNativeGoalTransport.spawn(
            [env["CODEX_BIN"], "app-server", "--listen", "stdio://"], cwd=str(project), env=env,
        ) as rpc:
            native = RoleTransport(rpc, project, env, deadline, admission, evidence, save, capture)
            native.request("initialize", {"clientInfo": {"name": "loopx_benchmark", "version": "1"},
                                         "capabilities": {"experimentalApi": True}})
            rpc.notify("initialized", {})
            result = run_roles(native, request)
        evidence["result"] = result
        save()
        return result


if __name__ == "__main__":
    print(json.dumps(serve(json.load(sys.stdin), os.environ), ensure_ascii=False))
