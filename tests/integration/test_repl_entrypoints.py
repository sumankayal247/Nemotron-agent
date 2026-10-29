"""Exercise the installed console scripts, not just the Python entry function."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["NA", "nemotron-agent"])
def test_installed_entrypoint_starts_the_same_repl(tmp_path: Path, script: str) -> None:
    scripts = Path(sys.executable).parent
    executable = scripts / (script + (".exe" if os.name == "nt" else ""))
    if not executable.is_file():
        pytest.skip("Console scripts are tested when the package is installed")
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "README.md").write_text("# Test\n", encoding="utf-8")
    env = {**os.environ, "NEMOTRON_AGENT_CONFIG_DIR": str(tmp_path / "config")}
    result = subprocess.run(
        [str(executable), "--mock", "--path", str(repo)],
        input="/version\n/exit\n",
        text=True,
        capture_output=True,
        env=env,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Nemotron Agent" in result.stdout
    assert "NA >" in result.stdout
    assert "nemotron-agent 0.1.0" in result.stdout
    assert "Goodbye." in result.stdout
