"""Unit tests for repository discovery, .gitignore, secret exclusions, and path confinement."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemotron_agent.logging_utils import PathViolationError
from nemotron_agent.permissions import PermissionManager, PermissionPolicy
from nemotron_agent.repo.context import ContextTracker, RepositoryContextService
from nemotron_agent.repo.discovery import (
    PathGuard,
    RepositoryIndexer,
    is_secret_like_path,
)


def test_secret_like_path_classification() -> None:
    assert is_secret_like_path(".env") is True
    assert is_secret_like_path("config/.env.production") is True
    assert is_secret_like_path("certs/private.key") is True
    assert is_secret_like_path("tls/server.pem") is True
    assert is_secret_like_path(".ssh/id_ed25519") is True
    assert is_secret_like_path("credentials.json") is True

    # Templates and ordinary code files must NOT be flagged
    assert is_secret_like_path(".env.example") is False
    assert is_secret_like_path(".env.template") is False
    assert is_secret_like_path("src/auth_handler.py") is False


def test_gitignore_default_exclusions_and_overview(tmp_path: Path) -> None:
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".gitignore").write_text("ignored_dir/\n*.log\n!keep.log\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
        encoding="utf-8",
    )
    (repo / ".env").write_text("SECRET_TOKEN=12345\n", encoding="utf-8")
    (repo / ".env.example").write_text("SECRET_TOKEN=changeme\n", encoding="utf-8")

    src = repo / "src"
    src.mkdir()
    (src / "app.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")

    # Ignored and dependency dirs
    (repo / "node_modules" / "pkg").mkdir(parents=True)
    (repo / "node_modules" / "pkg" / "index.js").write_text("module.exports = {}\n", encoding="utf-8")
    (repo / "ignored_dir").mkdir()
    (repo / "ignored_dir" / "skip.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "debug.log").write_text("noise\n", encoding="utf-8")
    (repo / "keep.log").write_text("important\n", encoding="utf-8")

    indexer = RepositoryIndexer(repo, extra_exclusions=["extra_skip.txt"])
    (repo / "extra_skip.txt").write_text("skip me\n", encoding="utf-8")

    files, secret_count = indexer.iter_allowed_files()
    rel_paths = [rel for _, rel in files]

    assert "src/app.py" in rel_paths
    assert "pyproject.toml" in rel_paths
    assert ".env.example" in rel_paths
    assert "keep.log" in rel_paths

    assert ".env" not in rel_paths
    assert secret_count == 1
    assert "debug.log" not in rel_paths
    assert "extra_skip.txt" not in rel_paths
    assert not any("node_modules" in p for p in rel_paths)
    assert not any("ignored_dir" in p for p in rel_paths)

    overview = indexer.inspect_overview()
    text = overview.format_text()
    assert "Verified Facts:" in text
    assert "Python (1)" in text
    assert "Inferred Likely Commands" in text
    assert "pytest" in text


def test_path_guard_blocks_outside_root_git_and_symlink_escapes(tmp_path: Path) -> None:
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("outside data\n", encoding="utf-8")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (repo / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    (repo / "escape_link.txt").symlink_to(outside)

    perms = PermissionManager(PermissionPolicy(interactive=False, is_free_endpoint=True))
    guard = PathGuard(repo, perms)

    with pytest.raises(PathViolationError, match="outside the selected project root"):
        guard.resolve_path("../outside_secret.txt")

    with pytest.raises(PathViolationError, match="outside the selected project root"):
        guard.resolve_path(str(outside))

    with pytest.raises(PathViolationError, match="outside the selected project root"):
        guard.resolve_path("escape_link.txt")

    with pytest.raises(PathViolationError, match="internal `.git`"):
        guard.resolve_path(".git/config")

    # Secret file is hard-blocked on `:free` endpoints even in interactive mode!
    perms_free_interactive = PermissionManager(
        PermissionPolicy(interactive=True, is_free_endpoint=True),
        input_fn=lambda _p: "y",
    )
    guard_free = PathGuard(repo, perms_free_interactive)
    with pytest.raises(PathViolationError, match="hosted `:free` endpoint"):
        guard_free.resolve_path(".env")

    # On non-free endpoint, interactive user approval gates secret file read
    perms_paid = PermissionManager(
        PermissionPolicy(interactive=True, is_free_endpoint=False),
        input_fn=lambda _p: "y",
        output_fn=lambda _s: None,
    )
    guard_paid = PathGuard(repo, perms_paid)
    _, rel = guard_paid.resolve_path(".env", operation="read")
    assert rel == ".env"


def test_context_service_search_read_and_span_tracking(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    code_lines = [f"line_{i} = {i}" for i in range(1, 31)]
    (repo / "calc.py").write_text("\n".join(code_lines) + "\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02\x03binary")

    perms = PermissionManager(PermissionPolicy(interactive=False))
    guard = PathGuard(repo, perms)
    indexer = RepositoryIndexer(repo)
    tracker = ContextTracker(max_context_chars=10000)
    svc = RepositoryContextService(indexer=indexer, path_guard=guard, tracker=tracker)

    search_out = svc.search_files(query="line_15")
    assert "calc.py:15:" in search_out

    read_1 = svc.read_file_lines(path="calc.py", start_line=1, end_line=5)
    assert "lines 1-5 of 30" in read_1
    assert "   1 | line_1 = 1" in read_1

    read_2 = svc.read_file_lines(path="calc.py", start_line=6, end_line=10)
    assert "   6 | line_6 = 6" in read_2

    # Adjacent spans merge cleanly in formatted context references
    assert tracker.formatted_references() == ["calc.py:1-10"]
    assert tracker.get_recorded_hash("calc.py") is not None

    with pytest.raises(PathViolationError, match="binary file"):
        svc.read_file_lines(path="blob.bin")
