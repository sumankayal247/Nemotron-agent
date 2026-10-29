"""End-to-end CLI integration tests (offline via mock mode and temporary repositories)."""

from __future__ import annotations

from pathlib import Path

from nemotron_agent.cli import run_cli


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "sample"\n', encoding="utf-8")
    (repo / "hello.py").write_text('def greet(name: str) -> str:\n    return f"Hello, {name}!"\n', encoding="utf-8")
    (repo / "skip_me.py").write_text("x = 999\n", encoding="utf-8")
    return repo


def test_cli_one_shot_shorthand_and_ask_in_mock_mode(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cfg_dir = tmp_path / "user_cfg"
    out: list[str] = []

    # Shorthand invocation: `nemotron-agent --mock --repo <repo> "Read hello.py"`
    rc = run_cli(
        ["--mock", "--repo", str(repo), "Read hello.py and summarize it"],
        config_dir=cfg_dir,
        env={},
        output_fn=out.append,
    )
    assert rc == 0
    rendered = "\n".join(out)
    assert "[MOCK MODE]" in rendered
    assert "Hello," in rendered
    assert "Context used   : hello.py:1-2" in rendered


def test_cli_exclude_flag_filters_files_from_search(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cfg_dir = tmp_path / "user_cfg"
    out: list[str] = []

    rc = run_cli(
        [
            "ask",
            "--mock",
            "--repo",
            str(repo),
            "--exclude",
            "skip_me.py",
            "search for 999",
        ],
        config_dir=cfg_dir,
        env={},
        output_fn=out.append,
    )
    assert rc == 0
    rendered = "\n".join(out)
    assert "No matches found" in rendered


def test_cli_interactive_repl_slash_commands(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cfg_dir = tmp_path / "user_cfg"
    out: list[str] = []

    repl_inputs = iter(
        [
            "/status",
            "Read hello.py",
            "/context",
            "/clear",
            "/context",
            "/readonly on",
            "/model 1",
            "/exit",
        ]
    )

    rc = run_cli(
        ["--mock", "--repo", str(repo)],
        config_dir=cfg_dir,
        env={},
        input_fn=lambda _p: next(repl_inputs),
        output_fn=out.append,
    )
    assert rc == 0
    rendered = "\n".join(out)
    assert "Active Nemotron Agent Configuration:" in rendered
    assert "hello.py:1-2" in rendered
    assert "Cleared conversation history and tracked repository file context." in rendered
    assert "Strict read-only mode enabled" in rendered
    assert "Switched active model to 'nvidia/nemotron-3.5-lightning:free'." in rendered
    assert "Goodbye." in rendered


def test_cli_update_check_reports_current_version(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out: list[str] = []
    rc = run_cli(
        ["--mock", "--repo", str(repo), "update", "--check"],
        config_dir=tmp_path / "user_cfg",
        env={},
        output_fn=out.append,
    )
    assert rc == 0
    rendered = "\n".join(out)
    assert "nemotron-agent" in rendered
    assert "Install method:" in rendered


def test_cli_non_interactive_without_credentials_fails_clearly(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cfg_dir = tmp_path / "user_cfg"
    out: list[str] = []

    rc = run_cli(
        ["--non-interactive", "--repo", str(repo), "What is this repo?"],
        config_dir=cfg_dir,
        env={},
        output_fn=out.append,
    )
    assert rc == 1
    rendered = "\n".join(out)
    assert "Configuration incomplete for non-interactive run" in rendered


def test_local_non_git_folder_selection_and_repl_switching(tmp_path: Path) -> None:
    # Two plain local directories on disk WITHOUT .git or GitHub
    folder_a = tmp_path / "local_app_a"
    folder_a.mkdir()
    (folder_a / "main.py").write_text("APP_NAME = 'LocalAppA'\n", encoding="utf-8")

    folder_b = tmp_path / "My Local App B"  # includes spaces like real desktop folders
    folder_b.mkdir()
    (folder_b / "server.py").write_text("PORT = 8080\n", encoding="utf-8")

    cfg_dir = tmp_path / "user_cfg"
    out: list[str] = []

    # 1. Test `--path` on plain non-Git local folder
    rc1 = run_cli(
        ["--mock", "--path", str(folder_a), "Overview of this folder"],
        config_dir=cfg_dir,
        env={},
        output_fn=out.append,
    )
    assert rc1 == 0
    rendered1 = "\n".join(out)
    assert "no (plain local directory)" in rendered1
    assert str(folder_a.resolve()) in rendered1

    # 2. Test `--select-dir` picker + `/project` switching to `folder_b` (with spaces) in REPL
    out.clear()
    inputs = iter(
        [
            str(folder_a),              # interactive --select-dir prompt
            "Read main.py",             # inspect folder A
            f"/cd {folder_b}",          # switch live to folder B on disk
            "Read server.py",           # inspect folder B
            "/project",                 # show active local folder
            "/exit",
        ]
    )
    rc2 = run_cli(
        ["--mock", "--select-dir"],
        config_dir=cfg_dir,
        env={},
        input_fn=lambda _p: next(inputs),
        output_fn=out.append,
    )
    assert rc2 == 0
    rendered2 = "\n".join(out)
    assert "LocalAppA" in rendered2
    assert f"Switched active local codebase to '{folder_b.resolve()}'" in rendered2
    assert "PORT = 8080" in rendered2
    assert f"Active local codebase folder: {folder_b.resolve()}" in rendered2
