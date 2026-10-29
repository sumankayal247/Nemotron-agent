"""Tests for nemotron_agent.update.UpdateManager.

All subprocesses and network calls are faked; nothing is ever installed.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from nemotron_agent.update import InstallMethod, UpdateManager, _package_root


def _fake_dist(url: str | None = None) -> SimpleNamespace:
    dist = SimpleNamespace(_path="/usr/local/lib/python3.11/site-packages")
    if url is not None:
        import json

        dist.read_text = lambda name: json.dumps({"url": url}) if name == "direct_url.json" else ""
    return dist


def test_detect_dev_from_direct_url(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager()
    monkeypatch.setattr(
        "nemotron_agent.update.distribution",
        lambda name: _fake_dist("file:///home/user/Nemotron-agent"),
    )
    assert m.detect_install_method() == InstallMethod.DEV


def test_detect_git_from_direct_url(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager()
    monkeypatch.setattr(
        "nemotron_agent.update.distribution",
        lambda name: _fake_dist("git+https://github.com/sumankayal247/Nemotron-agent.git"),
    )
    assert m.detect_install_method() == InstallMethod.GIT


def test_detect_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager()
    monkeypatch.setattr(
        "nemotron_agent.update.distribution",
        lambda name: _fake_dist(None),
    )
    monkeypatch.setattr("nemotron_agent.update._package_root", lambda: None)
    assert m.detect_install_method() == InstallMethod.PIP


def test_detect_unknown_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager()

    def boom(name: str):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("nemotron_agent.update.distribution", boom)
    monkeypatch.setattr("nemotron_agent.update._package_root", lambda: None)
    assert m.detect_install_method() == InstallMethod.UNKNOWN


def test_current_version_from_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager()
    monkeypatch.setattr(
        "nemotron_agent.update.version", lambda name: "2.3.4"
    )
    assert m.current_version() == "2.3.4"


def test_check_for_update_compares_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager(_network=lambda: "9.9.9")
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "1.0.0")
    info = m.check_for_update()
    assert info.latest == "9.9.9"
    assert info.available is True


def test_check_for_update_up_to_date(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager(_network=lambda: "0.1.0")
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "0.1.0")
    info = m.check_for_update()
    assert info.available is False


def test_check_for_update_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager(_network=lambda: None)
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "0.1.0")
    info = m.check_for_update()
    assert info.latest is None
    assert info.available is False


def test_is_dirty_git(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager()
    monkeypatch.setattr(
        "nemotron_agent.update._package_root", lambda: Path("/repo/src/nemotron_agent")
    )
    monkeypatch.setattr(
        "nemotron_agent.update.UpdateManager._git_root",
        lambda self, start: Path("/repo"),
    )

    def run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=" M src/nemotron_agent/cli.py\n", stderr="")

    monkeypatch.setattr("nemotron_agent.update.subprocess.run", run)
    assert m.is_dirty() is True

    def clean(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("nemotron_agent.update.subprocess.run", clean)
    assert m.is_dirty() is False


def test_apply_update_check_only(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager(_network=lambda: "9.9.9")
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "1.0.0")
    monkeypatch.setattr("nemotron_agent.update.UpdateManager.detect_install_method", lambda self: InstallMethod.PIP)
    result = m.apply_update(check_only=True)
    assert result.ok is True
    assert "9.9.9" in result.message and "Update available." in result.message


def test_apply_update_up_to_date(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager(_network=lambda: "0.1.0")
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "0.1.0")
    monkeypatch.setattr("nemotron_agent.update.UpdateManager.detect_install_method", lambda self: InstallMethod.PIP)
    result = m.apply_update()
    assert result.ok is True
    assert "Nothing to update" in result.message


def test_apply_update_asks_without_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    m = UpdateManager(_network=lambda: "2.0.0")
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "1.0.0")
    monkeypatch.setattr("nemotron_agent.update.UpdateManager.detect_install_method", lambda self: InstallMethod.PIP)

    def fake_install(method):
        called.append(method)
        return True, ""

    monkeypatch.setattr(m, "_install", fake_install)
    result = m.apply_update(yes=False)
    assert result.ok is True
    assert called == []
    assert "--yes" in result.message


def test_apply_update_dirty_git_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager(_network=lambda: "2.0.0")
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "1.0.0")
    monkeypatch.setattr(
        "nemotron_agent.update.UpdateManager.detect_install_method",
        lambda self: InstallMethod.GIT,
    )
    monkeypatch.setattr(m, "is_dirty", lambda: True)
    result = m.apply_update(yes=True)
    assert result.ok is False
    assert "Refusing to update" in result.message or "Uncommitted changes" in result.message


def test_apply_update_pipx_success(monkeypatch: pytest.MonkeyPatch) -> None:
    cmds = []
    m = UpdateManager(_network=lambda: "2.0.0")
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "1.0.0")
    monkeypatch.setattr(
        "nemotron_agent.update.UpdateManager.detect_install_method",
        lambda self: InstallMethod.PIPX,
    )

    def run(cmd, **kwargs):
        cmds.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="upgraded", stderr="")

    monkeypatch.setattr("nemotron_agent.update.subprocess.run", run)
    monkeypatch.setattr(m, "restart", lambda: None)
    result = m.apply_update(yes=True, restart_on_success=False)
    assert result.ok is True
    assert cmds == [["pipx", "upgrade", "nemotron-agent"]]


def test_apply_update_pip_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager(_network=lambda: "2.0.0")
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "1.0.0")
    monkeypatch.setattr(
        "nemotron_agent.update.UpdateManager.detect_install_method",
        lambda self: InstallMethod.PIP,
    )

    def run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="permission denied")

    monkeypatch.setattr("nemotron_agent.update.subprocess.run", run)
    monkeypatch.setattr(m, "restart", lambda: None)
    result = m.apply_update(yes=True, restart_on_success=False)
    assert result.ok is False
    assert "Update failed" in result.message


def test_apply_update_dev_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    m = UpdateManager(_network=lambda: "2.0.0")
    monkeypatch.setattr("nemotron_agent.update.version", lambda name: "1.0.0")
    monkeypatch.setattr(
        "nemotron_agent.update.UpdateManager.detect_install_method",
        lambda self: InstallMethod.DEV,
    )
    result = m.apply_update(yes=True, restart_on_success=False)
    assert result.ok is False
    assert "pip install -e" in result.message


def test_restart_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    m = UpdateManager()
    monkeypatch.setattr(m, "restart", lambda: called.append(True))
    m.restart()
    assert called == [True]


def test_parse_version_handles_suffixes() -> None:
    from nemotron_agent.update import _parse_version

    assert _parse_version("0.1.0") < _parse_version("2.0.0")
    assert _parse_version("10.2.3") > _parse_version("9.9.9")
    assert _parse_version("1.0.0") == _parse_version("1.0.0")
