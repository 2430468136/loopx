"""Codex Controller and Worker sessions serving one admitted Turn over the v1 Host ABI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from loopx.control_plane.effect_runtime import effect_runtime_result
from loopx.control_plane.goals.first_party_host_admission import FirstPartyHostGoalAdmission
from loopx.control_plane.quota.turn_envelope import COMPLETE_TURN_HOST_REQUEST_SCHEMA
from loopx.control_plane.runtime.document_io import atomic_write_private_json
from loopx.control_plane.turn_driver.codex_cli import codex_cli_result_schema, normalize_codex_result, prepare_codex_cli_input
from loopx.control_plane.turn_driver.codex_stdio import StdioNativeGoalTransport, disable_mcp_servers
from loopx.control_plane.turn_driver.decision_capture import read_turn_observation_text
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


@dataclass(frozen=True)
class RoleHost:
    project: Path
    runtime: Path
    registry: Path
    codex_home: Path
    codex_bin: str
    model: str | None
    effort: str | None
    sandbox: str
    timeout: float
    material: Path | None = None
    material_sha256: str | None = None
    goal_state: Path | None = None

    @property
    def protected(self):
        return (self.runtime, self.registry, self.codex_home, self.project / ".codex",
                self.project / ".loopx", self.project / ".git", self.project / "ACTIVE_GOAL_STATE.md",
                *((self.goal_state,) if self.goal_state is not None else ()))


def read_material(path: Path) -> bytes:
    # This operator-supplied text enters the Controller context once per admission.
    with path.open("rb") as stream:
        data = stream.read(128 * 1024 + 1)
    if len(data) > 128 * 1024:
        raise ValueError("Controller material exceeds 128 KiB")
    data.decode("utf-8")
    return data


def role_policy(role, host):
    project = host.project
    writable = role == "worker" and host.sandbox != "read-only"
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
    for path in host.protected:
        if project.is_relative_to(path):
            raise ValueError("worker workspace overlaps private Host state")
        if path.is_relative_to(project):
            fs[str(path.relative_to(project))] = "deny"
        else:
            filesystem[str(path)] = "deny"
    return profile, config


class RoleTransport:
    def __init__(self, rpc, host, deadline, admission, evidence, save):
        self.rpc, self.host = rpc, host
        self.deadline, self.admission = deadline, admission
        self.evidence, self.save = evidence, save
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

    def start(self, role, *, resume_thread=None):
        cwd = str(self.host.project)
        effective = self.request("config/read", {"cwd": cwd, "includeLayers": False})["config"]
        config = {"web_search": "disabled", "project_doc_max_bytes": 0, "skills": {"include_instructions": False}}
        params = {"cwd": cwd, "runtimeWorkspaceRoots": [cwd], "model": self.host.model,
                  "approvalPolicy": "never"}
        profile, policy = role_policy(role, self.host)
        config.update(policy)
        self.permissions[role] = profile
        params["permissions"] = profile
        config["features"] = {"apps": False, "multi_agent": False, "multi_agent_v2": False,
                              "hooks": False, "browser_use": False, "computer_use": False,
                              "goals": False, "js_repl": False}
        params["config"] = disable_mcp_servers(effective, config)
        binding = self.evidence.get("host_models", {}).get(role)
        if resume_thread is not None:
            if not isinstance(binding, dict) or not all(
                isinstance(binding.get(key), str) and binding[key] for key in ("model", "modelProvider")
            ):
                raise RuntimeError("original Controller model binding unavailable")
            params.update(threadId=resume_thread, excludeTurns=True,
                          model=binding["model"], modelProvider=binding["modelProvider"])
        with self.admission.launch_lifetime(operation="turn_role_thread_start"):
            observed = self.request("thread/resume" if resume_thread is not None else "thread/start", params)
        thread = observed["thread"]["id"]
        if (not isinstance(thread, str) or not thread or thread in self.threads.values()
                or (resume_thread is not None and thread != resume_thread)
                or observed.get("approvalPolicy") != "never" or observed.get("runtimeWorkspaceRoots") != [cwd]
                or (observed.get("activePermissionProfile") or {}).get("id") != profile):
            raise RuntimeError("native role identity or permissions unconfirmed")
        observed_model = {key: observed.get(key) for key in ("model", "modelProvider")}
        if (not all(isinstance(value, str) and value for value in observed_model.values())
                or (resume_thread is not None and observed_model != binding)):
            raise RuntimeError("native role model binding unconfirmed")
        self.evidence.setdefault("host_models", {})[role] = observed_model
        self.threads[role] = thread
        self.evidence["threads"][role] = thread
        self.save()

    def send(self, role, prompt, schema):
        if role not in self.threads:
            self.start(role)
        thread = self.threads[role]
        params = {"threadId": thread, "cwd": str(self.host.project),
                  "input": [{"type": "text", "text": prompt}], "outputSchema": schema,
                  "effort": self.host.effort, "approvalPolicy": "never"}
        params["permissions"] = self.permissions[role]
        began = time.monotonic()
        with self.admission.launch_lifetime(operation="turn_role_turn_start"):
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
                    with self.admission.launch_lifetime(operation="turn_role_continue"):
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


def run_roles(native, request, material=None):
    prompt, result_schema = prepare_codex_cli_input(request)
    decision_schema = object_schema({"instruction": {"type": ["string", "null"]},
                                     "result": {"anyOf": [result_schema, {"type": "null"}]}})
    prompt += ("\nYou are the Controller. The Host read the journal-bound original decision, verified its "
               "identity and contents, and delivered it in full in Turn request.work_contract. "
               "The captured_decision read is fulfilled. Use that inline snapshot for the original decision. "
               "Read any remaining required task material. "
               "Delegate bounded task work in instruction, including necessary task material and constraints. "
               "Keep Core protocol and private control context here. Assess the returned finding and artifacts. "
               "Give actionable feedback if another work segment is needed. Fill exactly one of instruction or result. "
               "Return the typed Turn result when ready. At most three Worker segments share this admission.")
    if material is not None:
        prompt += "\nOperator-supplied task material:\n" + material
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


def prepare_role_attempt(path, journal, request):
    """Select the original result or the one Core-authorized receipt repair under the role lock."""
    request_hash = digest(request)
    if not path.exists():
        evidence = {"turn_key": request["turn_key"], "input_sha256": request_hash, "threads": {}, "steps": []}
        return evidence, evidence

    evidence = json.loads(path.read_text())
    if not isinstance(evidence.get("result"), dict):
        raise RuntimeError("original role attempt requires reconciliation")
    if evidence["input_sha256"] == request_hash:
        return evidence, evidence

    # Only the result-contract feedback may differ from the completed attempt.
    contract = dict(request["result_contract"])
    feedback = contract.pop("correction", None)
    if (not isinstance(feedback, str) or not feedback
            or digest({**request, "result_contract": contract}) != evidence["input_sha256"]):
        raise RuntimeError("original role attempt requires reconciliation")
    correction = evidence.get("correction")
    if correction is not None:
        if correction.get("input_sha256") == request_hash and isinstance(correction.get("result"), dict):
            return evidence, correction
        raise RuntimeError("original receipt correction requires reconciliation")

    audit = json.loads(journal.read_text()).get("recovery_audit") or {}
    planned = audit.get("planned") or {}
    if (planned.get("retry_failed") is not True or planned.get("reinvoke_host") is not True
            or planned.get("resume_from") != "host_execute" or (audit.get("actual") or {}).get("status") != "started"):
        raise RuntimeError("original Turn has no authorized Host retry")
    correction = {"input_sha256": request_hash}
    evidence["correction"] = correction
    return evidence, correction


def correct_original_receipt(native, request, evidence):
    controller = evidence["threads"].get("controller")
    if not isinstance(controller, str) or not controller or controller == evidence["threads"].get("worker"):
        raise RuntimeError("original Controller identity unavailable")
    native.start("controller", resume_thread=controller)
    feedback = request["result_contract"]["correction"]
    reply = native.send("controller", feedback + " Return the corrected typed Turn result.",
                        codex_cli_result_schema(request))
    if reply.get("result_kind") != evidence["result"].get("result_kind"):
        raise ValueError("receipt correction changed the original result kind")
    return normalize_codex_result(request, reply)


def serve(request, host):
    if request.get("schema_version") != COMPLETE_TURN_HOST_REQUEST_SCHEMA:
        raise ValueError("controller-worker requires a complete C1 Turn request")
    identity = extract_turn_work_identity(request)
    project, runtime = host.project, host.runtime
    key = request["turn_key"]
    journal = turn_journal_path(runtime, goal_id=identity["goal_id"], turn_key=key)
    capture = journal.parent / "decisions" / journal.name
    reference = request.get("detail_ref", {}).get("captured_decision", {})
    if reference.get("path") != str(capture) or capture.resolve(strict=True) != capture:
        raise ValueError("original Turn decision capture unavailable")
    # prepare_codex_cli_input validates the complete contract and its source hash.
    # Compare the actual journal bytes as JSON before attesting that the inline
    # work_contract is the original. Neither role needs access to private Core files.
    if digest(json.loads(read_turn_observation_text(capture))) != digest(request["work_contract"]):
        raise ValueError("original decision differs from the delivered Turn work contract")
    from loopx.registry import find_registry_goal, load_registry, resolve_state_file

    goal = find_registry_goal(load_registry(host.registry), identity["goal_id"])
    if not goal:
        raise ValueError("Controller-worker requires its original registered Goal")
    state = resolve_state_file(Path(goal["repo"]), goal.get("state_file"))
    host = replace(host, goal_state=state.resolve() if state else None)
    material = None
    if host.material is not None:
        source = host.material.resolve(strict=True)
        if not source.is_file() or any(source.is_relative_to(path) for path in host.protected):
            raise ValueError("Controller material overlaps protected state or is not a file")
        data = read_material(source)
        if hashlib.sha256(data).hexdigest() != host.material_sha256:
            raise ValueError("Controller material changed after admission")
        material = data.decode("utf-8")
    path = journal.parent / "controller-worker" / journal.name
    timeout = host.timeout
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("controller-worker requires a finite execution budget")
    deadline = time.monotonic() + timeout
    admission = FirstPartyHostGoalAdmission.for_plan(registry_path=host.registry,
        goal_id=identity["goal_id"], planned_goal_ref=request.get("goal_ref"))
    with try_exclusive_file_lock(path) as lock:
        if lock is None:
            raise RuntimeError("original role attempt is still active")
        evidence, attempt = prepare_role_attempt(path, journal, request)
        if isinstance(attempt.get("result"), dict):
            return attempt["result"]
        # Test the selected executable and workspace sandbox before paid inference.
        # Turn supervision also enforces the outer deadline over this preflight.
        probe = subprocess.run([host.codex_bin, "sandbox", "-P", ":workspace", "--",
                                sys.executable, "-I", "-c", "pass"], cwd=project,
                               capture_output=True, timeout=min(15, timeout))
        if probe.returncode:
            raise RuntimeError("Native Codex workspace sandbox is unavailable")
        def save():
            atomic_write_private_json(path, evidence)
        save()
        with StdioNativeGoalTransport.spawn(
            [host.codex_bin, "app-server", "--listen", "stdio://"], cwd=str(project), env=os.environ.copy(),
        ) as rpc:
            native = RoleTransport(rpc, host, deadline, admission, evidence, save)
            native.request("initialize", {"clientInfo": {"name": "loopx_turn", "version": "1"},
                                         "capabilities": {"experimentalApi": True}})
            rpc.notify("initialized", {})
            if attempt is evidence:
                result = run_roles(native, request, material)
            else:
                result = correct_original_receipt(native, request, evidence)
        attempt["result"] = result
        save()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "runtime", "registry"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--sandbox", choices=("read-only", "workspace-write", "danger-full-access"), required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--material", type=Path)
    parser.add_argument("--material-sha256")
    args = parser.parse_args()
    home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser().resolve()
    host = RoleHost(args.project.resolve(), args.runtime.resolve(), args.registry.resolve(), home,
                    args.codex_bin, args.model, args.effort, args.sandbox, args.timeout, args.material, args.material_sha256)
    with interrupt_on_stop():
        print(json.dumps(serve(json.load(sys.stdin), host), ensure_ascii=False))


if __name__ == "__main__":
    main()
