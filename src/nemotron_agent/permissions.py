"""Permission policies and explicit user approval gates (read-only by default)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from nemotron_agent.logging_utils import PathViolationError, PermissionDeniedError

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


@dataclass
class PermissionPolicy:
    """Configurable permission settings; safe read-only defaults."""

    interactive: bool = True
    read_only: bool = False  # If True, hard-blocks writes & commands without even prompting
    allow_outside_root_prompt: bool = False  # Reject outside-root access unless explicitly enabled
    is_free_endpoint: bool = True  # Hard-blocks reading secret-like files on hosted `:free` endpoints
    auto_approve_edits: bool = False  # Only for explicit user flag in controlled tests; False by default
    auto_approve_commands: bool = False  # False by default


@dataclass
class ApprovalAuditEntry:
    action: str  # "edit_file", "run_command", "read_secret", "outside_root"
    target: str
    approved: bool
    reason: str = ""


class PermissionManager:
    """Centralized gatekeeper for file writes, command execution, secrets, and paths."""

    def __init__(
        self,
        policy: PermissionPolicy | None = None,
        *,
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
    ) -> None:
        self.policy = policy or PermissionPolicy()
        self._input_fn = input_fn
        self._output_fn = output_fn
        self.audit_log: list[ApprovalAuditEntry] = []
        self._approved_secret_files: set[str] = set()
        self._approved_outside_paths: set[Path] = set()

    def set_output_handler(self, output_fn: OutputFn) -> None:
        """Replace the sink used for approval and security notices."""
        self._output_fn = output_fn

    @property
    def has_approved_secret_reads(self) -> bool:
        return bool(self._approved_secret_files)

    def clear_scoped_approvals(self) -> None:
        """Do not carry sensitive-file or outside-path approvals into another project."""
        self._approved_secret_files.clear()
        self._approved_outside_paths.clear()

    def check_outside_root(self, candidate: Path, project_root: Path, *, operation: str) -> None:
        """Reject paths outside `project_root` unless explicitly enabled and approved by user."""
        resolved = candidate.resolve()
        if resolved in self._approved_outside_paths:
            return

        if not self.policy.allow_outside_root_prompt or not self.policy.interactive:
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="outside_root",
                    target=str(resolved),
                    approved=False,
                    reason=f"Blocked {operation} outside project root ({project_root})",
                )
            )
            raise PathViolationError(
                f"Access denied: path '{resolved}' is outside the selected project root "
                f"('{project_root}')."
            )

        self._output_fn("")
        self._output_fn("=== Security Warning: Path Outside Project Root ===")
        self._output_fn(f"  Operation    : {operation}")
        self._output_fn(f"  Target path  : {resolved}")
        self._output_fn(f"  Project root : {project_root}")
        ans = self._ask_yes_no("Explicitly approve access outside the project root? [y/N]: ")
        self.audit_log.append(
            ApprovalAuditEntry(action="outside_root", target=str(resolved), approved=ans)
        )
        if not ans:
            raise PathViolationError(
                f"User denied {operation} access to path outside project root: '{resolved}'."
            )
        self._approved_outside_paths.add(resolved)

    def check_secret_file_access(self, rel_path: str, *, operation: str) -> None:
        """Enforce secret-file protection; hard-block on hosted `:free` endpoints."""
        if self.policy.is_free_endpoint:
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="read_secret",
                    target=rel_path,
                    approved=False,
                    reason="Hard-blocked on hosted :free endpoint",
                )
            )
            raise PathViolationError(
                f"Access denied to secret-like file '{rel_path}': never send secrets, `.env` "
                "files, private keys, or credentials to a hosted `:free` endpoint."
            )

        if operation != "read":
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="write_secret",
                    target=rel_path,
                    approved=False,
                    reason="Modifying secret/credential files is blocked",
                )
            )
            raise PathViolationError(
                f"Refusing to modify secret-like or credential file '{rel_path}'."
            )

        if rel_path in self._approved_secret_files:
            return

        if not self.policy.interactive:
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="read_secret",
                    target=rel_path,
                    approved=False,
                    reason="Non-interactive mode denies secret files",
                )
            )
            raise PathViolationError(
                f"Access denied to secret-like file '{rel_path}' without interactive approval."
            )

        self._output_fn("")
        self._output_fn("=== Sensitive File Access Request ===")
        self._output_fn(f"  File      : {rel_path}")
        self._output_fn(f"  Operation : {operation}")
        self._output_fn(
            "  Risk      : Contents of this file will be included in model context sent to your "
            "configured provider. Do NOT approve if it contains live secrets or keys."
        )
        ans = self._ask_yes_no(f"Allow reading sensitive file '{rel_path}'? [y/N]: ")
        self.audit_log.append(
            ApprovalAuditEntry(action="read_secret", target=rel_path, approved=ans)
        )
        if not ans:
            raise PathViolationError(f"User denied access to sensitive file '{rel_path}'.")
        self._approved_secret_files.add(rel_path)

    def request_file_write_approval(
        self,
        *,
        rel_path: str,
        unified_diff: str,
        description: str,
        is_new_file: bool,
        risk_warning: str | None = None,
    ) -> bool:
        """Display unified diff and require explicit user approval before modifying a file."""
        if self.policy.read_only:
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="edit_file",
                    target=rel_path,
                    approved=False,
                    reason="Read-only mode active",
                )
            )
            raise PermissionDeniedError(
                f"Cannot modify '{rel_path}': session is in strict read-only mode."
            )

        self._output_fn("")
        action_label = "Create new file" if is_new_file else "Edit existing file"
        self._output_fn(f"=== Proposed File Change ({action_label}): {rel_path} ===")
        self._output_fn(f"Reason: {description}")
        if risk_warning:
            self._output_fn(f"WARNING: {risk_warning}")
        self._output_fn("--- Unified Diff Preview ---")
        self._output_fn(unified_diff.rstrip("\n"))
        self._output_fn("----------------------------")

        if self.policy.auto_approve_edits:
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="edit_file", target=rel_path, approved=True, reason=description
                )
            )
            return True

        if not self.policy.interactive:
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="edit_file",
                    target=rel_path,
                    approved=False,
                    reason="Non-interactive mode denies file writes",
                )
            )
            raise PermissionDeniedError(
                f"File modification to '{rel_path}' requires interactive approval."
            )

        approved = self._ask_yes_no(f"Apply this change to '{rel_path}'? [y/N]: ")
        self.audit_log.append(
            ApprovalAuditEntry(
                action="edit_file", target=rel_path, approved=approved, reason=description
            )
        )
        return approved

    def request_command_approval(
        self,
        *,
        executable: str,
        argv: list[str],
        cwd: Path,
        reason: str,
    ) -> bool:
        """Display exact executable, argument array, working directory, and reason for approval."""
        target_repr = repr(argv)
        if self.policy.read_only:
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="run_command",
                    target=target_repr,
                    approved=False,
                    reason="Read-only mode active",
                )
            )
            raise PermissionDeniedError(
                f"Cannot execute {argv!r}: session is in strict read-only mode."
            )

        self._output_fn("")
        self._output_fn("=== Command Execution Approval Request ===")
        self._output_fn(f"  Executable        : {executable}")
        self._output_fn(f"  Arguments (argv)  : {argv!r}")
        self._output_fn(f"  Working directory : {cwd}")
        self._output_fn(f"  Reason            : {reason}")

        if self.policy.auto_approve_commands:
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="run_command", target=target_repr, approved=True, reason=reason
                )
            )
            return True

        if not self.policy.interactive:
            self.audit_log.append(
                ApprovalAuditEntry(
                    action="run_command",
                    target=target_repr,
                    approved=False,
                    reason="Non-interactive mode denies command execution",
                )
            )
            raise PermissionDeniedError(
                f"Command {argv!r} requires interactive approval."
            )

        approved = self._ask_yes_no("Approve executing this command? [y/N]: ")
        self.audit_log.append(
            ApprovalAuditEntry(
                action="run_command", target=target_repr, approved=approved, reason=reason
            )
        )
        return approved

    def _ask_yes_no(self, prompt: str) -> bool:
        try:
            ans = self._input_fn(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            self._output_fn("\nCancelled; action denied.")
            return False
        return ans in ("y", "yes")
