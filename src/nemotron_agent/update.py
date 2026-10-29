"""Self-update manager for nemotron-agent.

Detects how the package was installed (git checkout, pip, pipx, or editable
dev install), queries PyPI for the latest version, and applies updates while
never touching user data (sessions, config, credentials) and never overwriting
an uncommitted git working tree silently.

All network and subprocess actions are isolated in this module so they can be
monkeypatched in tests.
"""

from __future__ import annotations

import enum
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Callable

PACKAGE_NAME = "nemotron-agent"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"


class InstallMethod(str, enum.Enum):
    GIT = "git"
    PIP = "pip"
    PIPX = "pipx"
    DEV = "dev"
    UNKNOWN = "unknown"


@dataclass
class UpdateInfo:
    """Result of checking for an available update."""

    current: str
    latest: str | None = None
    available: bool = False
    method: InstallMethod = InstallMethod.UNKNOWN
    dirty: bool = False
    error: str | None = None
    update_url: str = field(default_factory=lambda: f"https://pypi.org/project/{PACKAGE_NAME}/")


@dataclass
class UpdateResult:
    ok: bool
    message: str
    restarted: bool = False
    method: InstallMethod | None = None


class UpdateManager:
    """Check for and apply nemotron-agent updates."""

    def __init__(
        self,
        package_name: str = PACKAGE_NAME,
        *,
        fetch_timeout: float = 10.0,
        subprocess_timeout: float = 120.0,
        _network: Callable[[], str | None] | None = None,
    ) -> None:
        self.package_name = package_name
        self.fetch_timeout = fetch_timeout
        self.subprocess_timeout = subprocess_timeout
        self._network = _network  # injection point: returns latest version or None

    # ------------------------------------------------------------------ detect
    def detect_install_method(self) -> InstallMethod:
        """Best-effort install-method detection."""

        root = _package_root()

        try:
            dist = distribution(self.package_name)
        except PackageNotFoundError:
            dist = None

        direct_url: dict = {}
        if dist is not None:
            try:
                raw = dist.read_text("direct_url.json") or "{}"
                direct_url = json.loads(raw)
            except Exception:
                direct_url = {}
        url = str(direct_url.get("url", "") or "")
        if url:
            if "git+" in url or url.endswith(".git"):
                return InstallMethod.GIT
            # Editable/local installs describe themselves via a directory URL.
            return InstallMethod.DEV

        if root is not None:
            if self._git_root(root) is not None:
                return InstallMethod.GIT

        location = ""
        if dist is not None:
            location = str(getattr(dist, "_path", "") or "")
        elif root is not None:
            location = str(root)

        if "pipx/venvs" in location or "pipx" in sys.prefix:
            return InstallMethod.PIPX

        if shutil.which("pipx"):
            try:
                proc = subprocess.run(
                    ["pipx", "list"],
                    capture_output=True,
                    text=True,
                    timeout=self.subprocess_timeout,
                )
                if self.package_name in proc.stdout and (
                    dist is not None and "pipx" in str(getattr(dist, "_path", ""))
                ):
                    return InstallMethod.PIPX
            except Exception:
                pass

        if "site-packages" in location or "dist-packages" in location:
            return InstallMethod.PIP
        return InstallMethod.UNKNOWN

    def _git_root(self, start: Path) -> Path | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return Path(proc.stdout.strip())
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ versions
    def current_version(self) -> str:
        """Installed version (metadata, git describe, or packaged fallback)."""

        try:
            return version(self.package_name)
        except PackageNotFoundError:
            pass
        root = _package_root()
        if root is not None:
            git_root = self._git_root(root)
            target = git_root or root
            try:
                proc = subprocess.run(
                    ["git", "-C", str(target), "describe", "--tags", "--always"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
            except Exception:
                pass
        from nemotron_agent import __version__

        return __version__

    def fetch_latest_version(self) -> str | None:
        """Latest release version from PyPI, or None if unavailable."""

        if self._network is not None:
            return self._network()
        try:
            req = urllib.request.Request(
                PYPI_URL,
                headers={"Accept": "application/json", "User-Agent": "nemotron-agent-update-check"},
            )
            with urllib.request.urlopen(req, timeout=self.fetch_timeout) as response:
                if response.status != 200:
                    return None
                data = json.loads(response.read().decode("utf-8"))
            return str(data["info"]["version"])
        except Exception:
            return None

    def check_for_update(self) -> UpdateInfo:
        """Gather current/latest version info without modifying anything."""

        current = self.current_version()
        latest = self.fetch_latest_version()
        info = UpdateInfo(
            current=current,
            latest=latest,
            method=self.detect_install_method(),
        )
        if latest is not None:
            info.available = _parse_version(latest) > _parse_version(current)
        if info.method == InstallMethod.GIT:
            info.dirty = self.is_dirty()
        return info

    # ------------------------------------------------------------------ safety
    def is_dirty(self) -> bool:
        """True when the git working tree has uncommitted changes."""

        root = _package_root()
        target = self._git_root(root) if root is not None else None
        if target is None:
            target = Path.cwd()
            if self._git_root(target) is None:
                return False
        try:
            proc = subprocess.run(
                ["git", "-C", str(target), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                return False
            return bool(proc.stdout.strip())
        except Exception:
            return False

    # ------------------------------------------------------------------ apply
    def apply_update(
        self,
        *,
        check_only: bool = False,
        yes: bool = False,
        restart_on_success: bool = True,
    ) -> UpdateResult:
        """Apply an available update.

        ``check_only`` reports status without installing. ``yes`` skips the
        confirmation step (caller is responsible for consent). Returns a result
        object; callers decide whether to restart based on ``restarted``.
        """

        info = self.check_for_update()
        lines: list[str] = [
            f"nemotron-agent {info.current}",
            f"Install method: {info.method.value}",
        ]
        if info.latest:
            lines.append(f"Latest version: {info.latest}")
            lines.append("Update available." if info.available else "You are up to date.")
        else:
            lines.append("Latest version: unable to reach PyPI.")
        if info.error:
            lines.append(f"Note: {info.error}")
        if info.dirty:
            lines.append("Warning: uncommitted changes in the working tree.")

        if check_only:
            return UpdateResult(True, "\n".join(lines), method=info.method)

        if not info.available:
            return UpdateResult(True, "\n".join(lines) + "\nNothing to update.", method=info.method)

        if info.dirty and info.method == InstallMethod.GIT:
            return UpdateResult(
                False,
                "\n".join(lines)
                + "\nRefusing to update: uncommitted changes in the working tree. Commit or stash them, then re-run.",
                method=info.method,
            )

        if not yes:
            return UpdateResult(
                True,
                "\n".join(lines)
                + f"\nRun `nemotron-agent update --yes` to install {info.latest} ({info.method.value}).",
                method=info.method,
            )

        method = info.method
        ok, detail = self._install(method)

        message = "\n".join(lines)
        if ok:
            message += f"\nUpdated to {info.latest}."
            restarted = False
            if restart_on_success:
                try:
                    self.restart()
                    restarted = True
                except SystemExit:
                    raise
                except Exception as exc:  # pragma: no cover - restart environment-specific
                    restarted = False
                    message += f"\nAuto-restart failed ({exc}). Restart nemotron-agent manually."
            return UpdateResult(True, message, restarted=restarted, method=method)

        return UpdateResult(
            False,
            message + f"\nUpdate failed: {detail}\nTry: pip install --upgrade {self.package_name}",
            method=method,
        )

    def _install(self, method: InstallMethod) -> tuple[bool, str]:
        if method == InstallMethod.DEV:
            return False, "Editable install detected; reinstall with `pip install -e .` in the source checkout."
        if method == InstallMethod.GIT:
            return self._git_pull()
        if method == InstallMethod.PIPX:
            cmd = ["pipx", "upgrade", self.package_name]
        elif method == InstallMethod.PIP:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", self.package_name]
        else:
            return False, f"Unsupported install method for auto-update: {method.value}"
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )
        except FileNotFoundError as exc:
            return False, f"Required tool not found: {exc.filename}"
        except Exception as exc:  # pragma: no cover - subprocess environment
            return False, f"{type(exc).__name__}: {exc}"
        if proc.returncode == 0:
            return True, (proc.stdout or "").strip() or "Updated successfully."
        return False, (proc.stderr or proc.stdout or "install failed").strip()

    def _git_pull(self) -> tuple[bool, str]:
        if self.is_dirty():
            return False, "Uncommitted changes present; aborting pull."
        proc = subprocess.run(
            ["git", "pull", "--ff-only"],
            capture_output=True,
            text=True,
            timeout=self.subprocess_timeout,
            cwd=str(_package_root() or Path.cwd()),
        )
        if proc.returncode == 0:
            return True, proc.stdout.strip() or "Pulled latest changes."
        return False, (proc.stderr or proc.stdout or "git pull failed").strip()

    # ------------------------------------------------------------------ restart
    def restart(self) -> None:
        """Re-execute the current CLI process (replaces this process)."""

        sys.stdout.flush()
        sys.stderr.flush()
        argv = [sys.executable, "-c", "from nemotron_agent.cli import main; raise SystemExit(main())"] + sys.argv[1:]
        os.execv(sys.executable, argv)


def _parse_version(text: str) -> tuple:
    parts = []
    for chunk in str(text).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _package_root() -> Path | None:
    try:
        import nemotron_agent  # noqa: F401
    except ImportError:
        return None
    return Path(nemotron_agent.__file__).resolve().parent
