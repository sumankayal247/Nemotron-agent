"""Narrow file editing with conflict detection, unified diff generation, and approval."""

from __future__ import annotations

import difflib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nemotron_agent.logging_utils import ConflictError, PathViolationError
from nemotron_agent.permissions import PermissionManager
from nemotron_agent.repo.context import ContextTracker, compute_sha256
from nemotron_agent.repo.discovery import PathGuard, is_binary_file


@dataclass(frozen=True)
class FileChangeRecord:
    path: str
    created: bool
    lines_added: int
    lines_removed: int
    description: str


def generate_unified_diff(
    rel_path: str,
    original_text: str,
    updated_text: str,
    *,
    is_new_file: bool = False,
    context_lines: int = 3,
) -> str:
    """Build a standard unified diff preview."""
    orig_lines = [] if is_new_file else original_text.splitlines(keepends=True)
    new_lines = updated_text.splitlines(keepends=True)
    # Ensure trailing newline representation produces clean diff output
    if orig_lines and not orig_lines[-1].endswith("\n"):
        orig_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    from_label = "/dev/null" if is_new_file else f"a/{rel_path}"
    to_label = f"b/{rel_path}"
    diff_iter = difflib.unified_diff(
        orig_lines,
        new_lines,
        fromfile=from_label,
        tofile=to_label,
        n=context_lines,
    )
    return "".join(diff_iter)


def _count_diff_stats(diff_text: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


class FileEditor:
    """Applies narrow, reviewable file changes only after conflict check and user approval."""

    def __init__(
        self,
        *,
        path_guard: PathGuard,
        permissions: PermissionManager,
        tracker: ContextTracker,
        require_prior_read: bool = True,
    ) -> None:
        self.path_guard = path_guard
        self.permissions = permissions
        self.tracker = tracker
        self.require_prior_read = require_prior_read
        self.changes: list[FileChangeRecord] = []

    def apply_edit(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        create_if_missing: bool = False,
        description: str = "Apply targeted code change",
    ) -> str:
        resolved_path, rel_path = self.path_guard.resolve_path(path, operation="write")

        if create_if_missing:
            if resolved_path.exists():
                raise ConflictError(
                    f"Cannot create '{rel_path}' with `create_if_missing=True`: file already exists. "
                    "Read the file first and propose a narrow search-and-replace edit."
                )
            original_text = ""
            updated_text = new_text
            risk_warning = None
        else:
            if not resolved_path.exists():
                raise ConflictError(
                    f"Cannot edit '{rel_path}': file does not exist. Set `create_if_missing=True` "
                    "only if you intend to create a brand-new file."
                )
            if resolved_path.is_dir():
                raise PathViolationError(f"Cannot edit '{rel_path}': path is a directory.")
            if is_binary_file(resolved_path):
                raise PathViolationError(f"Refusing to edit binary file '{rel_path}'.")

            raw_bytes = resolved_path.read_bytes()
            current_sha = compute_sha256(raw_bytes)
            recorded_sha = self.tracker.get_recorded_hash(rel_path)

            if self.require_prior_read and recorded_sha is None:
                raise ConflictError(
                    f"Refusing to edit '{rel_path}' before reading it. Call `read_file` on "
                    f"'{rel_path}' first to inspect current contents and avoid clobbering edits."
                )
            if recorded_sha is not None and recorded_sha != current_sha:
                raise ConflictError(
                    f"Conflicting changes detected in '{rel_path}': the file was modified on disk "
                    "since it was last read. Call `read_file` again before proposing an edit."
                )

            try:
                original_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PathViolationError(
                    f"Refusing to edit '{rel_path}': file is not valid UTF-8 text."
                ) from exc

            match_count = original_text.count(old_text)
            if match_count == 0:
                raise ConflictError(
                    f"Target `old_text` snippet was not found in '{rel_path}'. Read the relevant "
                    "lines again and provide an exact matching snippet."
                )
            if match_count > 1:
                raise ConflictError(
                    f"Target `old_text` matched {match_count} locations in '{rel_path}'. Include "
                    "more surrounding lines so only the intended location is modified."
                )

            updated_text = original_text.replace(old_text, new_text, 1)
            risk_warning = None
            if original_text.strip() and not updated_text.strip():
                risk_warning = (
                    f"This edit removes all remaining content from '{rel_path}', leaving it empty."
                )
            elif not new_text.strip() and len(old_text.splitlines()) >= 10:
                risk_warning = f"This edit deletes {len(old_text.splitlines())} lines without replacement."

        diff_text = generate_unified_diff(
            rel_path,
            original_text,
            updated_text,
            is_new_file=create_if_missing,
        )
        if not diff_text.strip():
            return f"No effective diff produced for '{rel_path}'; file unchanged."

        approved = self.permissions.request_file_write_approval(
            rel_path=rel_path,
            unified_diff=diff_text,
            description=description,
            is_new_file=create_if_missing,
            risk_warning=risk_warning,
        )
        if not approved:
            return (
                f"User declined proposed edit to '{rel_path}'. File was NOT modified."
            )

        # The file or a parent symlink may change while the approval prompt is open.
        final_path, final_rel = self.path_guard.resolve_path(path, operation="write")
        if final_path != resolved_path or final_rel != rel_path:
            raise ConflictError(
                f"Path for '{rel_path}' changed while awaiting approval."
            )
        if create_if_missing:
            if resolved_path.exists() or resolved_path.is_symlink():
                raise ConflictError(
                    f"'{rel_path}' appeared while awaiting approval; no file was changed."
                )
        else:
            try:
                final_sha = compute_sha256(resolved_path.read_bytes())
            except OSError as exc:
                raise ConflictError(
                    f"Cannot recheck '{rel_path}' after approval: {exc}"
                ) from exc
            if final_sha != current_sha:
                raise ConflictError(
                    f"'{rel_path}' changed while awaiting approval; no file was changed."
                )

        self._atomic_write_text(resolved_path, updated_text)
        new_sha = compute_sha256(updated_text.encode("utf-8"))
        self.tracker.update_file_hash(rel_path, new_sha)

        added, removed = _count_diff_stats(diff_text)
        self.changes.append(
            FileChangeRecord(
                path=rel_path,
                created=create_if_missing,
                lines_added=added,
                lines_removed=removed,
                description=description,
            )
        )
        verb = "Created" if create_if_missing else "Updated"
        return f"{verb} '{rel_path}' (+{added} / -{removed} lines)."

    @staticmethod
    def _atomic_write_text(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        existing_mode: int | None = None
        if target.exists():
            try:
                existing_mode = stat.S_IMODE(target.stat().st_mode)
            except OSError:
                existing_mode = None

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.nemotron-", dir=target.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            if existing_mode is not None:
                try:
                    os.chmod(tmp_name, existing_mode)
                except OSError:
                    pass
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
