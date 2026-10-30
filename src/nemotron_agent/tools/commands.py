"""Allowlisted development command execution (argv-only, shell=False, explicit approval)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from nemotron_agent.config import (
    ENV_API_KEY,
    ENV_OPENROUTER_API_KEY,
)
from nemotron_agent.logging_utils import (
    CommandExecutionError,
    PathViolationError,
    PermissionDeniedError,
    redact_text,
)
from nemotron_agent.permissions import PermissionManager
from nemotron_agent.repo.discovery import PathGuard

BLOCKED_EXECUTABLES: frozenset[str] = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "dash",
        "fish",
        "csh",
        "tcsh",
        "ksh",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "sudo",
        "su",
        "doas",
        "rm",
        "rmdir",
        "mv",
        "cp",
        "dd",
        "mkfs",
        "chmod",
        "chown",
        "chgrp",
        "kill",
        "killall",
        "pkill",
        "reboot",
        "shutdown",
        "curl",
        "wget",
        "nc",
        "netcat",
        "ncat",
        "ssh",
        "scp",
        "sftp",
        "rsync",
        "ftp",
        "telnet",
        "docker",
        "podman",
        "kubectl",
        "helm",
        "terraform",
        "systemctl",
        "service",
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "brew",
        "pip",
        "pip3",
        "env",
        "nohup",
        "xargs",
        "find",
        "eval",
        "exec",
    }
)

BLOCKED_INLINE_CODE_FLAGS: frozenset[str] = frozenset(
    {
        "-c",
        "--command",
        "-e",
        "--eval",
        "--print",
        "--interactive",
    }
)

DEFAULT_ALLOWED_COMMAND_PREFIXES: tuple[tuple[str, ...], ...] = (
    # Common inspection & file search tools
    ("grep",),
    ("find",),
    ("cat",),
    ("head",),
    ("tail",),
    ("wc",),
    # Python test, lint, typecheck, format
    ("pytest",),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("python", "-m", "unittest"),
    ("python3", "-m", "unittest"),
    ("ruff", "check"),
    ("ruff", "format"),
    ("flake8",),
    ("mypy",),
    ("black",),
    ("isort",),
    # Node / TypeScript test, lint, check
    ("npm", "test"),
    ("npm", "run", "test"),
    ("npm", "run", "lint"),
    ("npm", "run", "format"),
    ("npm", "run", "typecheck"),
    ("npm", "run", "check"),
    ("npx", "jest"),
    ("npx", "vitest", "run"),
    ("npx", "eslint"),
    ("npx", "prettier", "--check"),
    ("npx", "tsc", "--noEmit"),
    # Rust & Go
    ("cargo", "test"),
    ("cargo", "check"),
    ("cargo", "clippy"),
    ("cargo", "fmt", "--check"),
    ("go", "test"),
    ("go", "vet"),
    # Safe read-only Git inspection
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
)

_STRIPPED_CHILD_ENV_VARS: frozenset[str] = frozenset(
    {
        ENV_API_KEY,
        ENV_OPENROUTER_API_KEY,
        "OPENAI_API_KEY",
        "NVIDIA_API_KEY",
    }
)


@dataclass(frozen=True)
class CommandRunRecord:
    argv: list[str]
    executable: str
    cwd: str
    reason: str
    exit_code: int
    duration_ms: int
    timed_out: bool = False


class AllowlistedCommandRunner:
    """Validates and executes bounded development commands without shell invocation."""

    def __init__(
        self,
        *,
        path_guard: PathGuard,
        permissions: PermissionManager,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 32768,
        extra_allowed_prefixes: Sequence[Sequence[str]] = (),
    ) -> None:
        self.path_guard = path_guard
        self.permissions = permissions
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.allowed_prefixes: list[tuple[str, ...]] = list(DEFAULT_ALLOWED_COMMAND_PREFIXES)
        for prefix in extra_allowed_prefixes:
            cleaned = tuple(str(tok).strip() for tok in prefix if str(tok).strip())
            if cleaned:
                self.allowed_prefixes.append(cleaned)
        self.runs: list[CommandRunRecord] = []

    def validate_command(self, argv: Sequence[str], *, cwd: str = ".") -> tuple[str, list[str], Path]:
        """Validate `argv` against blocklists, allowlist prefixes, and path confinement.

        Returns `(executable_display_path, normalized_argv, resolved_cwd)`.
        """
        if not argv:
            raise PermissionDeniedError("Command argument list cannot be empty.")

        tokens = [str(tok).strip() for tok in argv]
        if any(not tok for tok in tokens):
            raise PermissionDeniedError("Command contains empty argument tokens.")

        resolved_cwd, _ = self.path_guard.resolve_path(cwd, operation="read")
        if not resolved_cwd.exists() or not resolved_cwd.is_dir():
            raise CommandExecutionError(
                f"Command working directory '{cwd}' does not exist or is not a directory."
            )

        raw_exe = tokens[0]
        exe_basename = self._normalize_executable_name(raw_exe, resolved_cwd)

        if exe_basename.lower() in BLOCKED_EXECUTABLES:
            raise PermissionDeniedError(
                f"Command blocked by security policy: executable '{exe_basename}' is disallowed."
            )

        # Block inline script evaluation flags (e.g., python -c, node -e)
        for arg in tokens[1:]:
            flag_head = arg.split("=", 1)[0]
            if flag_head in BLOCKED_INLINE_CODE_FLAGS:
                raise PermissionDeniedError(
                    f"Command blocked by security policy: inline execution flag '{flag_head}' is disallowed."
                )

        # Check allowlist prefixes using normalized executable basename
        normalized_for_match = (exe_basename, *tokens[1:])
        if not self._matches_allowlist(normalized_for_match):
            raise PermissionDeniedError(
                f"Command {list(tokens)!r} is not in the development command allowlist. "
                "Only approved test, lint, typecheck, format, and read-only git inspection "
                "commands are permitted."
            )

        # Extra guard for git: block options that write arbitrary files or run external programs
        if exe_basename == "git":
            for arg in tokens[1:]:
                if arg.startswith("--output") or arg.startswith("--ext-diff") or arg.startswith("-c"):
                    raise PermissionDeniedError(
                        f"Git option '{arg}' is blocked in read-only git inspection commands."
                    )

        # Inspect path-like arguments to reject obvious escapes outside project_root
        self._check_argument_paths(tokens[1:], resolved_cwd)

        resolved_exe = self._resolve_executable_path(raw_exe, exe_basename)
        exec_argv = [resolved_exe, *tokens[1:]]
        return resolved_exe, exec_argv, resolved_cwd

    def run(self, argv: Sequence[str], *, cwd: str = ".", reason: str) -> str:
        """Validate, obtain explicit user approval, and execute command (`shell=False`)."""
        resolved_exe, exec_argv, resolved_cwd = self.validate_command(argv, cwd=cwd)

        approved = self.permissions.request_command_approval(
            executable=resolved_exe,
            argv=list(argv),
            cwd=resolved_cwd,
            reason=reason,
        )
        if not approved:
            return f"User declined execution of command {list(argv)!r}. Command was NOT run."

        child_env = {
            k: v for k, v in os.environ.items() if k not in _STRIPPED_CHILD_ENV_VARS
        }

        start = time.monotonic()
        try:
            completed = subprocess.run(
                exec_argv,
                cwd=str(resolved_cwd),
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_seconds,
                env=child_env,
            )
        except FileNotFoundError as exc:
            raise CommandExecutionError(
                f"Executable '{resolved_exe}' was not found on this system."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self.runs.append(
                CommandRunRecord(
                    argv=list(argv),
                    executable=resolved_exe,
                    cwd=str(resolved_cwd),
                    reason=reason,
                    exit_code=-1,
                    duration_ms=elapsed_ms,
                    timed_out=True,
                )
            )
            partial_out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            clean_partial = self._truncate_and_redact(partial_out)
            raise CommandExecutionError(
                f"Command {list(argv)!r} timed out after {self.timeout_seconds:.1f}s."
                + (f"\nPartial output:\n{clean_partial}" if clean_partial else "")
            ) from exc
        except KeyboardInterrupt:
            raise
        except OSError as exc:
            raise CommandExecutionError(
                f"Failed to execute command {list(argv)!r}: {exc}"
            ) from exc

        elapsed_ms = int((time.monotonic() - start) * 1000)
        self.runs.append(
            CommandRunRecord(
                argv=list(argv),
                executable=resolved_exe,
                cwd=str(resolved_cwd),
                reason=reason,
                exit_code=completed.returncode,
                duration_ms=elapsed_ms,
                timed_out=False,
            )
        )

        combined_parts: list[str] = [
            f"Command: {list(argv)!r}",
            f"Exit code: {completed.returncode} ({elapsed_ms} ms)",
        ]
        if completed.stdout.strip():
            combined_parts.append("STDOUT:\n" + self._truncate_and_redact(completed.stdout))
        if completed.stderr.strip():
            combined_parts.append("STDERR:\n" + self._truncate_and_redact(completed.stderr))
        if not completed.stdout.strip() and not completed.stderr.strip():
            combined_parts.append("(no stdout/stderr output)")
        return "\n".join(combined_parts)

    def _normalize_executable_name(self, raw_exe: str, resolved_cwd: Path) -> str:
        """Allow bare command names or `.venv/bin/<tool>` inside the project root only."""
        if "/" not in raw_exe and "\\" not in raw_exe:
            return raw_exe

        candidate = Path(raw_exe).expanduser()
        if not candidate.is_absolute():
            candidate = (resolved_cwd / candidate).resolve()
        else:
            candidate = candidate.resolve()

        project_root = self.path_guard.project_root
        allowed_venv_dirs = [
            (project_root / ".venv" / "bin").resolve(),
            (project_root / "venv" / "bin").resolve(),
            (project_root / ".venv" / "Scripts").resolve(),
            (project_root / "venv" / "Scripts").resolve(),
        ]
        # Also allow the active interpreter (`sys.executable`) for `python -m pytest`
        active_py = Path(sys.executable).resolve()
        if candidate == active_py:
            return "python"

        if any(candidate.parent == vdir for vdir in allowed_venv_dirs):
            stem = candidate.stem if sys.platform == "win32" else candidate.name
            return stem

        raise PermissionDeniedError(
            f"Direct path executable '{raw_exe}' is blocked. Use bare allowlisted tool names "
            "(such as 'pytest' or 'ruff')."
        )

    def _matches_allowlist(self, normalized_tokens: tuple[str, ...]) -> bool:
        for prefix in self.allowed_prefixes:
            if len(normalized_tokens) >= len(prefix) and normalized_tokens[: len(prefix)] == prefix:
                return True
        return False

    def _check_argument_paths(self, args: Sequence[str], resolved_cwd: Path) -> None:
        """Reject arguments that traverse outside `project_root` or point to absolute system paths."""
        root = self.path_guard.project_root
        for arg in args:
            # Extract value part of --flag=/some/path
            candidate_str = arg.split("=", 1)[1] if arg.startswith("-") and "=" in arg else arg
            if candidate_str.startswith("-"):
                continue

            if candidate_str.startswith("/") or candidate_str.startswith("~") or ".." in Path(candidate_str).parts:
                p = Path(candidate_str).expanduser()
                resolved = p.resolve() if p.is_absolute() else (resolved_cwd / p).resolve()
                if not (resolved == root or resolved.is_relative_to(root)):
                    raise PathViolationError(
                        f"Command argument '{arg}' resolves outside project root ('{root}')."
                    )

    def _resolve_executable_path(self, raw_exe: str, exe_basename: str) -> str:
        if "/" in raw_exe or "\\" in raw_exe:
            candidate = Path(raw_exe).expanduser()
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (self.path_guard.project_root / candidate).resolve()
            )
            return str(resolved)

        # Prefer project-local virtualenv binaries when available (e.g., .venv/bin/pytest)
        root = self.path_guard.project_root
        venv_candidates = [
            root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / exe_basename,
            root / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / exe_basename,
        ]
        for vc in venv_candidates:
            if vc.is_file() and os.access(vc, os.X_OK):
                return str(vc.resolve())

        found = shutil.which(exe_basename)
        if found:
            return found

        if exe_basename in ("python", "python3"):
            return sys.executable

        return exe_basename

    def _truncate_and_redact(self, text: str) -> str:
        scrubbed = redact_text(text)
        encoded = scrubbed.encode("utf-8", errors="replace")
        if len(encoded) <= self.max_output_bytes:
            return scrubbed.rstrip()
        clipped = encoded[: self.max_output_bytes].decode("utf-8", errors="ignore")
        return clipped.rstrip() + f"\n... [output truncated at {self.max_output_bytes} bytes]"
