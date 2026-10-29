"""Private, bounded JSON stores for explicit memory and resumable sessions.

These stores do not serialize credentials, approval decisions, executable objects,
file hashes, or provider clients. Conversation snapshots can contain source code;
users should treat their private config directory accordingly.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from nemotron_agent.config import _ensure_not_inside_repo
from nemotron_agent.logging_utils import AgentError, redact_text, redact_value
from nemotron_agent.providers.base import ChatMessage

MAX_STORE_BYTES = 4 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _private_write(path: Path, data: object, project_root: Path) -> None:
    # Check the *resolved* destination too: a pre-existing symlinked subdirectory
    # must not redirect otherwise-private data into the active repository.
    _ensure_not_inside_repo(path.parent.resolve(), project_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        payload = json.dumps(redact_value(data), ensure_ascii=False, indent=2) + "\n"
        if len(payload.encode("utf-8")) > MAX_STORE_BYTES:
            raise AgentError("Private store exceeds the 4 MiB limit; it was not saved.")
        fd, tmp_name = tempfile.mkstemp(prefix=".na-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    except OSError as exc:
        raise AgentError(f"Unable to save private data at {path}: {exc}") from exc


def _read_json(path: Path) -> object:
    try:
        # Never follow a session/memory file symlink supplied by a third party.
        if path.is_symlink() or path.stat().st_size > MAX_STORE_BYTES:
            raise AgentError(f"Refusing unsafe or oversized private data file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentError(f"Unable to read private data at {path}: {exc}") from exc


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = Field(max_length=4000)
    steps: list[str] = Field(default_factory=list, max_length=50)
    approved: bool = False

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, value: list[str]) -> list[str]:
        if any(not step.strip() or len(step) > 5000 for step in value):
            raise ValueError("Plan steps must be nonempty and at most 5000 characters.")
        return value


class SavedSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=500)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=500)
    created_at: str = Field(max_length=64)


class SessionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    id: str
    created_at: str = Field(max_length=64)
    updated_at: str = Field(max_length=64)
    title: str = Field(max_length=120)
    last_prompt: str = Field(default="", max_length=100_000)
    repo_path: str = Field(min_length=1, max_length=4096)
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=500)
    base_url: str = Field(default="", max_length=2048)
    messages: list[ChatMessage] = Field(default_factory=list, max_length=10000)
    spans: list[SavedSpan] = Field(default_factory=list, max_length=5000)
    context_chars: int = Field(default=0, ge=0, le=10_000_000)
    plan: Plan | None = None
    artifacts: list[ArtifactRecord] = Field(default_factory=list, max_length=1000)
    agent: str = Field(default="coder", max_length=100)
    skills: list[str] = Field(default_factory=list, max_length=100)
    effort: Literal["low", "medium", "high"] = "high"
    read_only: bool = False

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if UUID(value).hex != value:
            raise ValueError("Invalid session ID.")
        return value

    @field_validator("messages")
    @classmethod
    def _no_system_messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if any(message.role == "system" for message in value):
            raise ValueError("Saved system messages are not trusted.")
        if any(message.images for message in value):
            raise ValueError(
                "Clipboard images must not be stored in session snapshots."
            )
        return value


class SessionStore:
    def __init__(self, config_dir: Path) -> None:
        self.directory = config_dir / "sessions"

    def save(self, snapshot: SessionSnapshot, project_root: Path) -> None:
        _private_write(
            self.directory / f"{snapshot.id}.json",
            snapshot.model_dump(mode="json"),
            project_root,
        )

    def list(self, *, exclude_id: str | None = None) -> list[SessionSnapshot]:
        if not self.directory.is_dir():
            return []
        snapshots: list[SessionSnapshot] = []
        try:
            paths = sorted(islice(self.directory.glob("*.json"), 1000))
        except OSError as exc:
            raise AgentError(f"Cannot list saved sessions: {exc}") from exc
        for path in paths:
            try:
                # ID validation prevents arbitrary paths and fabricated filenames.
                if UUID(path.stem).hex != path.stem:
                    continue
                snapshot = SessionSnapshot.model_validate(_read_json(path))
                if snapshot.id == path.stem and snapshot.id != exclude_id:
                    snapshots.append(snapshot)
            except (AgentError, ValueError, ValidationError):
                continue
        return sorted(snapshots, key=lambda item: item.updated_at, reverse=True)


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2000)
    repo_path: str = Field(min_length=1, max_length=4096)
    created_at: str = Field(max_length=64)


class MemoryStore:
    """Only explicit /learn additions are persisted, scoped to one repository."""

    def __init__(self, config_dir: Path) -> None:
        self.path = config_dir / "memory.json"

    def all(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        raw = _read_json(self.path)
        if not isinstance(raw, list) or len(raw) > 10000:
            raise AgentError("Invalid persistent memory file.")
        try:
            return [MemoryEntry.model_validate(item) for item in raw]
        except ValidationError as exc:
            raise AgentError(f"Invalid persistent memory file: {exc}") from exc

    def for_repo(self, project_root: Path) -> list[MemoryEntry]:
        root = str(project_root.resolve())
        return [entry for entry in self.all() if entry.repo_path == root]

    def add(self, project_root: Path, text: str) -> MemoryEntry:
        clean = text.strip()
        if not clean or len(clean) > 2000:
            raise AgentError("Learned information must contain 1–2000 characters.")
        if redact_text(clean) != clean:
            raise AgentError(
                "Refusing to store information that appears to contain a credential or secret."
            )
        entries = self.all()
        if len(entries) >= 10000:
            raise AgentError("Persistent memory is full.")
        entry = MemoryEntry(
            text=clean, repo_path=str(project_root.resolve()), created_at=utc_now()
        )
        entries.append(entry)
        _private_write(
            self.path, [item.model_dump(mode="json") for item in entries], project_root
        )
        return entry
