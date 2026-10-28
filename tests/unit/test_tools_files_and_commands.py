"""Unit tests for strict tool validation, diff-gated file editing, and command safety."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from nemotron_agent.config import ENV_OPENROUTER_API_KEY
from nemotron_agent.logging_utils import (
    CommandExecutionError,
    ConflictError,
    PathViolationError,
    PermissionDeniedError,
    ToolValidationError,
)
from nemotron_agent.permissions import PermissionManager, PermissionPolicy
from nemotron_agent.providers.base import ToolCallRequest
from nemotron_agent.repo.context import ContextTracker, RepositoryContextService
from nemotron_agent.repo.discovery import PathGuard, RepositoryIndexer
from nemotron_agent.tools.commands import AllowlistedCommandRunner
from nemotron_agent.tools.files import FileEditor
from nemotron_agent.tools.registry import ToolRegistry


def _build_stack(
    repo: Path,
    *,
    approve: bool = True,
    interactive: bool = True,
    read_only: bool = False,
    outputs: list[str] | None = None,
) -> tuple[ToolRegistry, RepositoryContextService, FileEditor, AllowlistedCommandRunner]:
    sink = outputs if outputs is not None else []
    perms = PermissionManager(
        PermissionPolicy(
            interactive=interactive,
            read_only=read_only,
            is_free_endpoint=True,
        ),
        input_fn=lambda _p: ("y" if approve else "n"),
        output_fn=sink.append,
    )
    guard = PathGuard(repo, perms)
    indexer = RepositoryIndexer(repo)
    tracker = ContextTracker()
    ctx = RepositoryContextService(indexer=indexer, path_guard=guard, tracker=tracker)
    editor = FileEditor(path_guard=guard, permissions=perms, tracker=tracker)
    runner = AllowlistedCommandRunner(path_guard=guard, permissions=perms, timeout_seconds=10.0)
    registry = ToolRegistry(
        indexer=indexer,
        context_service=ctx,
        file_editor=editor,
        command_runner=runner,
    )
    return registry, ctx, editor, runner


def test_tool_registry_strictly_validates_tool_names_and_schemas(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    registry, *_ = _build_stack(repo)

    # Unknown tool rejected
    with pytest.raises(ToolValidationError, match="Rejected unknown tool 'bash'"):
        registry.validate_call(ToolCallRequest(id="1", name="bash", arguments_raw={"cmd": "ls"}))

    # Malformed JSON string rejected
    with pytest.raises(ToolValidationError, match="Invalid JSON"):
        registry.validate_call(ToolCallRequest(id="2", name="read_file", arguments_raw="{bad"))

    # Extra unknown arguments rejected (extra="forbid")
    with pytest.raises(ToolValidationError, match="Extra inputs are not permitted"):
        registry.validate_call(
            ToolCallRequest(
                id="3",
                name="read_file",
                arguments_raw={"path": "a.py", "sudo": True},
            )
        )

    # Invalid line range rejected
    with pytest.raises(ToolValidationError, match="end_line"):
        registry.validate_call(
            ToolCallRequest(
                id="4",
                name="read_file",
                arguments_raw={"path": "a.py", "start_line": 50, "end_line": 10},
            )
        )


def test_file_editor_diff_preview_conflict_check_and_narrow_preservation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "math_utils.py"
    target.write_text(
        "# User header comment\ndef add(a, b):\n    return a - b\n\ndef keep_me():\n    return 42\n",
        encoding="utf-8",
    )

    outputs: list[str] = []
    _, ctx, editor, _ = _build_stack(repo, approve=False, outputs=outputs)

    # 1. Must read file before editing
    with pytest.raises(ConflictError, match="before reading it"):
        editor.apply_edit(
            path="math_utils.py",
            old_text="return a - b",
            new_text="return a + b",
            description="Fix bug in add",
        )

    # Read file to establish baseline SHA-256
    ctx.read_file_lines(path="math_utils.py", start_line=1, end_line=20)

    # 2. User denies edit -> unified diff was displayed, file NOT modified!
    msg_denied = editor.apply_edit(
        path="math_utils.py",
        old_text="return a - b",
        new_text="return a + b",
        description="Fix subtraction typo",
    )
    assert "NOT modified" in msg_denied
    assert "return a - b" in target.read_text(encoding="utf-8")
    joined_preview = "\n".join(outputs)
    assert "--- a/math_utils.py" in joined_preview
    assert "+++ b/math_utils.py" in joined_preview
    assert "-    return a - b" in joined_preview
    assert "+    return a + b" in joined_preview

    # 3. External concurrent edit detected via SHA-256 conflict check!
    target.write_text(
        target.read_text(encoding="utf-8") + "\n# Concurrent user edit\n",
        encoding="utf-8",
    )
    with pytest.raises(ConflictError, match="Conflicting changes detected"):
        editor.apply_edit(
            path="math_utils.py",
            old_text="return a - b",
            new_text="return a + b",
        )

    # 4. Re-read and approve -> narrow edit preserves both header and concurrent user edit!
    outputs_ok: list[str] = []
    _, ctx2, editor2, _ = _build_stack(repo, approve=True, outputs=outputs_ok)
    ctx2.read_file_lines(path="math_utils.py", start_line=1, end_line=30)
    msg_ok = editor2.apply_edit(
        path="math_utils.py",
        old_text="return a - b",
        new_text="return a + b",
        description="Fix subtraction bug",
    )
    assert "Updated 'math_utils.py'" in msg_ok
    final_content = target.read_text(encoding="utf-8")
    assert "return a + b" in final_content
    assert "# User header comment" in final_content
    assert "# Concurrent user edit" in final_content
    assert "def keep_me():" in final_content
    assert len(editor2.changes) == 1


def test_command_runner_allowlist_blocklist_and_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    # Create a tiny passing unittest inside the temporary repo
    (repo / "test_sample.py").write_text(
        "import os, unittest\n"
        "class SampleTest(unittest.TestCase):\n"
        "    def test_env_stripped(self):\n"
        f"        self.assertNotIn('{ENV_OPENROUTER_API_KEY}', os.environ)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_OPENROUTER_API_KEY, "sk-or-v1-should-not-leak-to-child")

    outputs: list[str] = []
    _, _, _, runner = _build_stack(repo, approve=True, outputs=outputs)

    # Blocked shell & destructive commands
    for forbidden in (
        ["bash", "-c", "echo hi"],
        ["rm", "-rf", "."],
        ["curl", "https://example.com"],
        ["python3", "-c", "print(1)"],
        ["git", "push", "origin", "main"],
        ["git", "reset", "--hard"],
        ["./custom_script.sh"],
    ):
        with pytest.raises(PermissionDeniedError):
            runner.validate_command(forbidden)

    # Path escape in command argument rejected
    with pytest.raises(PathViolationError, match="outside project root"):
        runner.validate_command(["pytest", "../outside_dir"])

    # Approved allowlisted command executes with shell=False and stripped API key env!
    result = runner.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "."],
        cwd=".",
        reason="Verify sample unit test passes",
    )
    assert "Exit code: 0" in result
    assert len(runner.runs) == 1
    assert runner.runs[0].exit_code == 0

    rendered_prompt = "\n".join(outputs)
    assert "Executable" in rendered_prompt
    assert "Arguments (argv)" in rendered_prompt
    assert "Working directory" in rendered_prompt
    assert "Verify sample unit test passes" in rendered_prompt

    # Denied command does NOT execute
    _, _, _, runner_deny = _build_stack(repo, approve=False)
    denied_msg = runner_deny.run(
        [sys.executable, "-m", "unittest"],
        cwd=".",
        reason="Run unit tests",
    )
    assert "NOT run" in denied_msg
    assert len(runner_deny.runs) == 0


def test_command_runner_enforces_timeout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_hang.py").write_text(
        "import time, unittest\n"
        "class HangTest(unittest.TestCase):\n"
        "    def test_sleep(self):\n"
        "        time.sleep(5)\n",
        encoding="utf-8",
    )
    perms = PermissionManager(
        PermissionPolicy(interactive=True, auto_approve_commands=True),
        output_fn=lambda _s: None,
    )
    guard = PathGuard(repo, perms)
    runner = AllowlistedCommandRunner(path_guard=guard, permissions=perms, timeout_seconds=0.3)

    with pytest.raises(CommandExecutionError, match="timed out"):
        runner.run(
            [sys.executable, "-m", "unittest", "test_hang.py"],
            reason="Test timeout handling",
        )
    assert runner.runs[-1].timed_out is True
