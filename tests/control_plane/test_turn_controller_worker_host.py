"""One Core admission, two native identities and independently checked work."""
import json
import os
from contextlib import nullcontext
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

import pytest

from loopx.control_plane.turn_driver import controller_worker as roles


@pytest.mark.parametrize("on_stop", [True, False], ids=["operator-stop", "terminal-error"])
def test_stop_addresses_only_the_active_native_turn(on_stop, tmp_path):
    class Endpoint:
        def __init__(self):
            self.interrupts = []

        def request(self, method, params):
            if method == "turn/start":
                return {"turn": {"id": "active"}}
            assert method == "turn/interrupt"
            self.interrupts.append(params)
            raise RuntimeError("interrupt acknowledgement unavailable")

        def next_event(self, *, timeout_sec):
            if on_stop:
                raise InterruptedError("stopped")
            return {"method": "turn/completed", "params": {
                "threadId": "original-worker", "turn": {"id": "active", "status": "failed"}}}

    class Admission:
        def launch_lifetime(self, *, operation):
            return nullcontext()

    rpc = Endpoint()
    state = {"steps": []}
    native = roles.RoleTransport(rpc, roles.RoleHost(tmp_path, tmp_path, tmp_path, tmp_path, "codex", None, "low", "read-only", 5),
                                 time.monotonic() + 5, Admission(), state, lambda: None)
    native.threads["worker"] = "original-worker"
    native.permissions["worker"] = "workspace"
    with pytest.raises(InterruptedError if on_stop else RuntimeError, match="stopped|did not complete"):
        native.send("worker", "check", {})
    assert rpc.interrupts == ([{"threadId": "original-worker", "turnId": "active"}] if on_stop else [])
    assert state["steps"][0]["thread_token_usage_total"] is None


@pytest.mark.parametrize("permission", ["danger-full-access", "read-only"])
def test_native_role_permissions_cap_the_callers_grant(tmp_path, permission):
    project = tmp_path / "task"
    project.mkdir()
    host = roles.RoleHost(project, tmp_path / "state", tmp_path / "registry", tmp_path / "home",
                          "codex", None, "low", permission, 10)
    for role in ("controller", "worker"):
        name, config = roles.role_policy(role, host)
        rules = config["permissions"][name]
        root = rules["filesystem"][":workspace_roots"]
        assert root["."] == ("write" if role == "worker" and permission != "read-only" else "read")
        assert root[".loopx"] == root[".codex"] == "deny"
        assert rules["filesystem"][str(host.runtime)] == "deny"
        assert rules["filesystem"][str(host.codex_home)] == "deny"
        assert rules["network"]["enabled"] is False


@pytest.mark.skipif(os.environ.get("LOOPX_TEST_NATIVE_ROLE_HOST") != "1",
                    reason="opt-in installed Codex with native filesystem isolation")
@pytest.mark.parametrize("entry", ["turn", "benchmark"])
def test_real_native_feedback_changes_the_checked_artifact(tmp_path, entry):
    from tests.control_plane.canonical_authority_fixture import initialize_canonical_source_fixture
    from tests.control_plane.codex_checkpoint_fixture import local_inference_provider

    src = Path(__file__).resolve().parents[2]
    fixture = runpy.run_path(str(src / "examples/loopx-turn-codex-cli-e2e-smoke.py"))
    project, runtime, _, registry = fixture["_write_fixture"](tmp_path, turn_count=2)
    goal, agent = fixture["GOAL_ID"], fixture["AGENT_ID"]
    initialize_canonical_source_fixture(runtime, goal,
        state_path=project / ".codex" / "goals" / goal / "ACTIVE_GOAL_STATE.md", provider="sqlite")
    (tmp_path / "skills").mkdir()
    marker, expected = fixture["MARKER_NAME"], fixture["_marker_value"](1)
    (project / "observation").write_text(expected)
    key = None
    inputs = []
    def respond(request, index, message):
        nonlocal key
        text = [part["text"] for item in request["input"] if item.get("role") == "user"
                for part in item["content"] if "text" in part][-1]
        inputs.append(text)
        if index == 1:
            contract = json.loads(text.split("Turn request:\n")[1].split("\nYou are the Controller.")[0])
            key = contract["turn_key"]
            answer = {"instruction": "Inspect the observation.", "result": None}
        elif index in (2, 5):
            command = ("if cat .loopx/registry.json >/dev/null; then exit 17; fi; cat observation"
                       if index == 2 else "cp observation " + marker)
            return {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
                    "name": "exec_command", "arguments": json.dumps({"cmd": command})}
        elif index == 3:
            output = str(request["input"][-1]["output"])
            denied = any(reason in output.lower() for reason in ("denied", "not permitted"))
            answer = {"finding": expected if expected in output and denied else "unverified", "artifacts": []}
        elif index == 4:
            answer = {"instruction": "Copy the checked observation into the output file." if expected in text
                      else "Report the missing observation.", "result": None}
        elif index == 6:
            answer = {"finding": "Written", "artifacts": [marker]}
        else:
            answer = {"instruction": None, "result": {
                "schema_version": "loopx_turn_result_v0", "turn_key": key, "result_kind": "validated_progress",
                "completed_phases": ["host_execute", "typed_result"], "classification": "fixture_progress",
                "recommended_action": "Check the output.", "next_action": "Continue the work.",
                "delivery_batch_scale": "single_surface", "delivery_outcome": "outcome_progress",
                "vision_unchanged_reason": "The objective is unchanged.", "summary": "Updated the artifact.",
                "work_evidence": ["artifact:marker"], "validation_evidence": ["check:marker-readback"]}}
        message["content"][0]["text"] = json.dumps(answer)
        return message

    binary = shutil.which("codex")
    assert binary, "Install the Codex CLI before enabling native role qualification"
    with local_inference_provider(respond) as (url, _, _):
        env = dict(os.environ, HOME=str(tmp_path / "home"), PYTHONPATH=str(src),
            LOOPX_CLI=str(Path(sys.executable).parent / "loopx"), CODEX_BIN=binary, MODEL_NAME="gpt-5.4",
            REASONING_EFFORT="low", OPENAI_BASE_URL=url, OPENAI_API_KEY="fixture", LOOPX_USAGE_PING="0",
            LOOPX_PROJECT=str(project), LOOPX_RUNTIME_ROOT=str(runtime), LOOPX_REGISTRY=str(registry),
            LOOPX_GOAL_ID=goal, LOOPX_AGENT_ID=agent, LOOPX_SHARED_SKILLS=str(tmp_path / "skills"),
            LOOPX_CODEX_HOME=str(tmp_path / "codex"), LOOPX_WAKE_LOG_DIR=str(tmp_path / "logs/wakes"),
            LOOPX_CODEX_SANDBOX="workspace-write", LOOPX_EXECUTION_MODE="turn", LOOPX_TASK_ENTRY="seeded-todo",
            LOOPX_TURN_TOPOLOGY="controller-worker", LOOPX_CODEX_TURN_TIMEOUT_SEC="45",
            LOOPX_VALIDATION_COMMAND_JSON=json.dumps(fixture["_validator_command"](expected)))
        if entry == "benchmark":
            from benchmark.runtime.scheduler import worker_command
            cmd = worker_command(env, python=sys.executable, source=str(src),
                                 state_file=str(tmp_path / "scheduler.json"), host_timeout=70) + ["--once"]
        else:
            # Standalone product invocation, with no benchmark variables or guest installation.
            material = tmp_path / "task-material.txt"
            material.write_text("Read the supplied observation as a checksum.")
            home = tmp_path / "role-home"
            home.mkdir()
            (home / "config.toml").write_text(
                f'model="gpt-5.4"\nmodel_reasoning_effort="low"\nmodel_provider="local"\n[model_providers.local]\nname="local"\nbase_url="{url}"\n'
                'wire_api="responses"\nenv_key="OPENAI_API_KEY"\n'
            )
            env = {k: v for k, v in os.environ.items() if not k.startswith("LOOPX_")}
            env.update(CODEX_HOME=str(home), OPENAI_API_KEY="fixture", PYTHONPATH=str(src), LOOPX_USAGE_PING="0")
            cmd = [sys.executable, "-m", "loopx.cli", "--registry", str(registry), "--runtime-root", str(runtime),
                   "--format", "json", "turn", "run-once", "--execute", "--no-global-sync",
                   "--goal-id", goal, "--agent-id", agent, "--project", str(project),
                   "--host", "codex-cli", "--execution-mode", "isolated-headless", "--topology", "controller-worker",
                   "--scheduler-owner", "outer_controller", "--prepare-selected-todo",
                   "--turn-instance-id", "direct-roles", "--timeout-seconds", "45", "--codex-bin", binary,
                   "--controller-material", str(material), "--codex-sandbox", "workspace-write",
                   "--validation-command-json", json.dumps(fixture["_validator_command"](expected))]
        completed = subprocess.run(cmd, cwd=project, env=env, timeout=85, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout[-2200:] + completed.stderr[-500:]
    assert len(inputs) == 7 and "Copy the checked observation" in inputs[4]
    assert all("work_contract" not in inputs[i] for i in (1, 2, 4, 5))
    assert (project / marker).read_text() == expected
    if entry == "benchmark":
        wake, = (tmp_path / "logs/wakes").iterdir()
        assert json.loads((wake / "receipt.json").read_text())["turn_execution"]["status"] == "committed"
    else:
        assert "Read the supplied observation as a checksum." in inputs[0]
        result = json.loads(completed.stdout)
        assert result["status"] == "committed" and result["host"]["topology"] == "controller-worker"
        start = cmd.index("--turn-instance-id")
        recovery = cmd[:start] + ["--resume-turn-key", result["resume_turn_key"]] + cmd[start + 2:]
        replay = subprocess.run(recovery, cwd=project, env=env, capture_output=True, text=True, timeout=15)
        assert replay.returncode == 0 and not any(json.loads(replay.stdout)["effects"].values())
        for flag, value in (("--topology", "single-worker"), ("--codex-sandbox", "read-only")):
            changed = list(recovery)
            changed[changed.index(flag) + 1] = value
            if flag == "--topology":
                offset = changed.index("--controller-material")
                del changed[offset:offset + 2]
            held = subprocess.run(changed, cwd=project, env=env, capture_output=True, text=True, timeout=15)
            assert held.returncode != 0 and "original Turn topology" in held.stdout + held.stderr
        material.write_text("Changed task material.")
        held = subprocess.run(recovery, cwd=project, env=env, capture_output=True, text=True, timeout=15)
        assert held.returncode != 0 and "original Turn topology" in held.stdout + held.stderr
    record, = (runtime / "goals" / goal / "turns/controller-worker").glob("*.json")
    observation = json.loads(record.read_text())
    assert len(set(observation["threads"].values())) == 2
    assert [s["role"] for s in observation["steps"]] == ["controller", "worker", "controller", "worker", "controller"]
    assert all(s["thread_token_usage_total"]["totalTokens"] > 0 for s in observation["steps"])
    assert fixture["_quota_spend_count"](runtime) == 1

    if entry == "turn":
        from loopx.control_plane.turn_driver import build_loopx_turn_host_request

        journal = json.loads((record.parent.parent / record.name).read_text())
        original = build_loopx_turn_host_request(journal["plan"])
        capture = Path(original["detail_ref"]["captured_decision"]["path"])
        changed_source = json.loads(capture.read_text())
        changed_source["recommended_action"] = "An unrelated action."
        capture.write_text(json.dumps(changed_source))
        host = roles.RoleHost(project, runtime, registry, home, binary, None, None, "workspace-write", 5)
        with pytest.raises(ValueError, match="original decision differs"):
            roles.serve(original, host)
