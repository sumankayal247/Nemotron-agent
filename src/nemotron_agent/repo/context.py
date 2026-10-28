"""Targeted repository file listing, search, line-ranged reading, and context tracking."""

from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from nemotron_agent.logging_utils import AgentError, PathViolationError
from nemotron_agent.repo.discovery import PathGuard, RepositoryIndexer, is_binary_file

MAX_LINES_PER_READ = 400
MAX_LINE_DISPLAY_CHARS = 300


def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class ContextSpan:
    """Tracks a specific file and 1-based inclusive line range used as model context."""

    path: str
    start_line: int
    end_line: int

    def label(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class ContextTracker:
    """Tracks files read, line ranges, SHA-256 snapshots (for conflict checks), and budget."""

    def __init__(self, max_context_chars: int = 48000) -> None:
        self.max_context_chars = max_context_chars
        self._spans: list[ContextSpan] = []
        self._file_hashes: dict[str, str] = {}
        self._chars_used: int = 0

    @property
    def chars_used(self) -> int:
        return self._chars_used

    def record_read(
        self,
        rel_path: str,
        start_line: int,
        end_line: int,
        *,
        sha256: str,
        char_count: int,
    ) -> None:
        self._file_hashes[rel_path] = sha256
        self._chars_used += max(0, char_count)
        span = ContextSpan(path=rel_path, start_line=start_line, end_line=end_line)
        if span not in self._spans:
            self._spans.append(span)

    def update_file_hash(self, rel_path: str, sha256: str) -> None:
        """Update stored SHA-256 after an approved write by the agent itself."""
        self._file_hashes[rel_path] = sha256

    def get_recorded_hash(self, rel_path: str) -> str | None:
        return self._file_hashes.get(rel_path)

    def remaining_chars(self) -> int:
        return max(0, self.max_context_chars - self._chars_used)

    def spans(self) -> list[ContextSpan]:
        return list(self._spans)

    def formatted_references(self) -> list[str]:
        """Return merged, sorted `path:start-end` labels for files used as context."""
        by_path: dict[str, list[tuple[int, int]]] = {}
        for s in self._spans:
            by_path.setdefault(s.path, []).append((s.start_line, s.end_line))

        refs: list[str] = []
        for path in sorted(by_path):
            intervals = sorted(by_path[path])
            merged: list[tuple[int, int]] = []
            for st, en in intervals:
                if merged and st <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], en))
                else:
                    merged.append((st, en))
            range_str = ", ".join(f"{st}-{en}" for st, en in merged)
            refs.append(f"{path}:{range_str}")
        return refs

    def clear(self) -> None:
        self._spans.clear()
        self._file_hashes.clear()
        self._chars_used = 0


class RepositoryContextService:
    """Provides bounded, safe repository file listing, search, and slice reading."""

    def __init__(
        self,
        *,
        indexer: RepositoryIndexer,
        path_guard: PathGuard,
        tracker: ContextTracker,
    ) -> None:
        self.indexer = indexer
        self.path_guard = path_guard
        self.tracker = tracker

    def list_files(
        self,
        *,
        directory: str = ".",
        glob_pattern: str = "*",
        max_entries: int = 100,
    ) -> str:
        resolved_dir, rel_dir = self.path_guard.resolve_path(directory, operation="read")
        if not resolved_dir.exists():
            return f"Directory '{rel_dir}' does not exist."
        if not resolved_dir.is_dir():
            return f"Path '{rel_dir}' is a file, not a directory."

        allowed_files, secret_skipped = self.indexer.iter_allowed_files(subpath=resolved_dir)
        matched: list[str] = []
        for _, rel_f in allowed_files:
            fname = rel_f.rsplit("/", 1)[-1]
            if glob_pattern in ("*", "**", "**/*") or fnmatch.fnmatch(rel_f, glob_pattern) or fnmatch.fnmatch(fname, glob_pattern):
                matched.append(rel_f)

        total = len(matched)
        shown = matched[:max_entries]
        header = f"Files under '{rel_dir}' matching '{glob_pattern}' (showing {len(shown)} of {total}"
        if secret_skipped:
            header += f"; {secret_skipped} secret-like file(s) excluded"
        header += "):"

        if not shown:
            return f"{header}\n  (no matching files)"
        body = "\n".join(f"  {p}" for p in shown)
        if total > len(shown):
            body += f"\n  ... ({total - len(shown)} more files; narrow directory or glob_pattern)"
        return f"{header}\n{body}"

    def search_files(
        self,
        *,
        query: str,
        directory: str = ".",
        file_glob: str = "*",
        case_sensitive: bool = False,
        is_regex: bool = False,
        max_results: int = 25,
    ) -> str:
        resolved_dir, rel_dir = self.path_guard.resolve_path(directory, operation="read")
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            return f"Search directory '{rel_dir}' does not exist or is not a directory."

        flags = 0 if case_sensitive else re.IGNORECASE
        if is_regex:
            try:
                pattern = re.compile(query, flags)
            except re.error as exc:
                raise AgentError(f"Invalid search regular expression {query!r}: {exc}") from exc
        else:
            pattern = re.compile(re.escape(query), flags)

        allowed_files, _ = self.indexer.iter_allowed_files(subpath=resolved_dir)
        hits: list[str] = []
        truncated = False

        for resolved_f, rel_f in allowed_files:
            fname = rel_f.rsplit("/", 1)[-1]
            if file_glob not in ("*", "**", "**/*") and not (
                fnmatch.fnmatch(rel_f, file_glob) or fnmatch.fnmatch(fname, file_glob)
            ):
                continue
            if is_binary_file(resolved_f):
                continue
            try:
                raw = resolved_f.read_bytes()
                # Skip huge generated files (> 1 MB) during text search
                if len(raw) > 1_048_576:
                    continue
                text = raw.decode("utf-8", errors="replace")
            except OSError:
                continue

            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    snippet = line.strip()
                    if len(snippet) > MAX_LINE_DISPLAY_CHARS:
                        snippet = snippet[:MAX_LINE_DISPLAY_CHARS] + "..."
                    hits.append(f"  {rel_f}:{lineno}: {snippet}")
                    if len(hits) >= max_results:
                        truncated = True
                        break
            if truncated:
                break

        if not hits:
            return f"No matches found for {query!r} under '{rel_dir}'."
        header = f"Search matches for {query!r} under '{rel_dir}' ({len(hits)} match(es)):"
        footer = (
            f"\n  ... (reached max_results={max_results}; refine query or path for more)"
            if truncated
            else ""
        )
        return f"{header}\n" + "\n".join(hits) + footer

    def read_file_lines(
        self,
        *,
        path: str,
        start_line: int = 1,
        end_line: int = 200,
    ) -> str:
        resolved_f, rel_f = self.path_guard.resolve_path(path, operation="read")
        if not resolved_f.exists():
            return f"File '{rel_f}' does not exist."
        if resolved_f.is_dir():
            return f"Path '{rel_f}' is a directory; use `list_files` instead."

        # Also block ignored files ONLY if they are secret-like (already checked by PathGuard);
        # however, binary files must never be dumped into model prompts.
        if is_binary_file(resolved_f):
            raise PathViolationError(
                f"Refusing to read binary file '{rel_f}' into text context."
            )

        if self.tracker.remaining_chars() <= 0:
            return (
                f"Context character budget ({self.tracker.max_context_chars} chars) is full. "
                "Answer using already gathered context or ask the user to `/clear` context."
            )

        raw_bytes = resolved_f.read_bytes()
        sha = compute_sha256(raw_bytes)
        text = raw_bytes.decode("utf-8", errors="replace")
        all_lines = text.splitlines()
        total_lines = len(all_lines)

        if total_lines == 0:
            self.tracker.record_read(rel_f, 1, 1, sha256=sha, char_count=0)
            return f"File '{rel_f}' is empty (0 lines)."

        clamped_start = max(1, min(start_line, total_lines))
        max_end = min(end_line, clamped_start + MAX_LINES_PER_READ - 1, total_lines)
        selected = all_lines[clamped_start - 1 : max_end]

        formatted_lines = [
            f"{lineno:4d} | {line[:500]}"
            for lineno, line in enumerate(selected, start=clamped_start)
        ]
        slice_body = "\n".join(formatted_lines)

        # Respect remaining context character budget
        budget = self.tracker.remaining_chars()
        if len(slice_body) > budget:
            slice_body = slice_body[:budget] + "\n... [truncated to stay within model context budget]"

        self.tracker.record_read(
            rel_f,
            clamped_start,
            max_end,
            sha256=sha,
            char_count=len(slice_body),
        )

        header = f"File: {rel_f} (lines {clamped_start}-{max_end} of {total_lines})"
        more_note = (
            f"\n[Lines {max_end + 1}-{total_lines} not shown; call `read_file` with start_line={max_end + 1} if needed]"
            if max_end < total_lines
            else ""
        )
        return f"{header}\n{slice_body}{more_note}"
