"""Transport the ordinary managed claim selected by the typed quota owner."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...command_invocation import current_loopx_cli_argv

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def compact_execution_preparation_request(
    payload: Mapping[str, Any], *, claim_source_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Retain only the exact Host-executable action beside a private capture."""

    preparation = payload.get("execution_preparation")
    selected = payload.get("selected_todo")
    if not isinstance(preparation, Mapping) or not isinstance(selected, Mapping):
        return None
    claim = preparation.get("claim")
    goal_id = str(payload.get("goal_id") or "").strip()
    agent_id = str(preparation.get("agent_id") or "").strip()
    todo_id = str(preparation.get("todo_id") or "").strip()
    selected_todo_id = str(selected.get("todo_id") or "").strip()
    registry_path = Path(str(payload.get("registry") or "")).expanduser()
    runtime_root = Path(str(payload.get("runtime_root") or "")).expanduser()
    if (
        preparation.get("schema_version") != "execution_preparation_v0"
        or preparation.get("state") != "claim_required"
        or not isinstance(claim, Mapping)
        or claim.get("host_execution_allowed") is not True
        or not goal_id
        or not agent_id
        or not todo_id
        or selected_todo_id != todo_id
        or not registry_path.is_absolute()
        or registry_path.resolve() != registry_path
        or not runtime_root.is_absolute()
        or runtime_root.resolve() != runtime_root
    ):
        return None
    claim_args = claim.get("cli_args")
    expected_claim = [
        "todo", "claim", "--goal-id", goal_id, "--todo-id", todo_id,
        "--agent-id", agent_id, "--claimed-by", agent_id,
    ]
    if claim_args != expected_claim:
        return None
    receipt = payload.get("heartbeat_receipt")
    turn_instance_id = (
        str(receipt.get("turn_instance_id") or "").strip()
        if isinstance(receipt, Mapping)
        else ""
    )
    claim_binding = {}
    if turn_instance_id and claim_source_sha256 is not None:
        operation_id = f"todo-claim:codex-capture:{claim_source_sha256}"
        expected_claim += ["--claim-operation-id", operation_id]
        claim_binding["claim_source_sha256"] = claim_source_sha256
    return {
        "goal_id": goal_id,
        "agent_id": agent_id,
        "todo_id": todo_id,
        "claim_args": expected_claim,
        # The lifecycle hook can run from any project.  Retain the exact route
        # that produced this digest-bound action instead of rediscovering Goal
        # authority from the hook process cwd.
        "authority_route": {
            "registry_path": str(registry_path),
            "runtime_root": str(runtime_root),
        },
        **({"turn_instance_id": turn_instance_id} if turn_instance_id else {}),
        **claim_binding,
    }


def default_runner(
    command: Sequence[str], *, cwd: Path, timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
        check=False,
    )


def run_json(
    arguments: Sequence[str],
    *,
    registry_path: Path,
    runtime_root: str | Path | None,
    cwd: Path,
    timeout: float,
    runner: CommandRunner,
) -> dict[str, Any]:
    command = [
        *current_loopx_cli_argv(),
        "--registry", str(registry_path),
        *(["--runtime-root", str(runtime_root)] if runtime_root is not None else []),
        "--format", "json",
        *arguments,
    ]
    completed = runner(command, cwd=cwd, timeout=timeout)
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("LoopX Host preparation returned non-JSON output") from exc
    if (
        completed.returncode != 0
        or not isinstance(payload, dict)
        or payload.get("ok") is False
    ):
        reason = payload.get("reason") if isinstance(payload, dict) else None
        raise RuntimeError(str(reason or completed.stderr or "LoopX Host preparation failed"))
    return payload


def execute_ordinary_claim(arguments, *, operation_id, registry_path, runtime_root, cwd, timeout, runner=default_runner):
    """Execute the quota owner's exact public claim and verify the returned operation."""
    claimed = run_json(arguments, registry_path=registry_path, runtime_root=runtime_root,
                       cwd=cwd, timeout=timeout, runner=runner)
    if claimed.get("claim_operation_id") != operation_id:
        raise RuntimeError("Native execution preparation did not recover its original managed claim")


def prepare_selected_turn_todo(decision, *, goal_id, agent_id, selected_todo_id, turn_instance_id,
                               registry_path, runtime_root, cwd, timeout):
    """The fresh Turn decision owns selection. The caller rereads it before dispatch."""
    import hashlib

    preparation = compact_execution_preparation_request(decision)
    if preparation is None:
        return None
    todo = preparation["todo_id"]
    route = {"registry_path": str(registry_path.resolve()), "runtime_root": str(runtime_root.resolve())}
    if (not turn_instance_id or decision.get("should_run") is not True
            or (preparation["goal_id"], preparation["agent_id"]) != (goal_id, agent_id)
            or preparation["authority_route"] != route
            or selected_todo_id is not None and selected_todo_id != todo
            or preparation.get("turn_instance_id", turn_instance_id) != turn_instance_id):
        raise ValueError("ordinary claim belongs to another Turn, Todo or authority route")
    identity = json.dumps([goal_id, agent_id, todo, turn_instance_id], separators=(",", ":")).encode()
    operation = "todo-claim:turn:" + hashlib.sha256(identity).hexdigest()
    execute_ordinary_claim([*preparation["claim_args"], "--claim-operation-id", operation],
        operation_id=operation, registry_path=registry_path, runtime_root=runtime_root, cwd=cwd, timeout=timeout)
    return todo
