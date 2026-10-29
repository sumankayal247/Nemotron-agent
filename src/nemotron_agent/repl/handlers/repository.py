"""Handlers for repository slash commands."""

from __future__ import annotations

from pathlib import Path

from nemotron_agent.logging_utils import AgentError
from nemotron_agent.providers.base import ToolCallRequest
from nemotron_agent.repl.commands import (
    Invocation,
)
from nemotron_agent.repl.storage import ArtifactRecord
from nemotron_agent.repo.discovery import is_secret_like_path


class RepositoryCommands:
    def _local_tool(self, name: str, arguments: dict[str, object]) -> None:
        try:
            result = self.session.tool_registry.execute_call(
                ToolCallRequest(id="local-command", name=name, arguments_raw=arguments)
            )
        except OSError as exc:
            raise AgentError(f"Unable to access repository resource: {exc}") from exc
        self.output(result.raw_output)
        if result.ok and name == "read_file":
            self.output(
                "(Local read; use @file in a prompt to include its contents for the model.)"
            )

    def cmd_tools(self, inv: Invocation) -> None:
        specs = self.session.tool_registry.get_tool_specs(
            read_only=self.session.permissions.policy.read_only
        )
        self.output("Available agent tools:")
        for spec in specs:
            self.output(f"  {spec.name} — {spec.description}")
        if self.session.permissions.policy.read_only:
            self.output("Strict read-only mode hides edits and command execution.")

    def cmd_search(self, inv: Invocation) -> None:
        self._local_tool("search_files", {"query": " ".join(inv.args)})

    def cmd_read(self, inv: Invocation) -> None:
        self._local_tool("read_file", {"path": inv.args[0]})

    def cmd_files(self, inv: Invocation) -> None:
        self._local_tool("list_files", {"directory": inv.args[0] if inv.args else "."})

    def _artifact_path(self, record: ArtifactRecord) -> Path:
        raw = Path(record.path)
        if raw.is_absolute():
            raise AgentError("Invalid artifact path.")
        try:
            path = (self.session.project_root / raw).resolve()
        except (OSError, RuntimeError) as exc:
            raise AgentError(f"Invalid artifact path: {exc}") from exc
        if not path.is_relative_to(self.session.project_root):
            raise AgentError("Artifact path escapes the repository.")
        rel = path.relative_to(self.session.project_root).as_posix()
        if rel == ".git" or rel.startswith(".git/") or is_secret_like_path(rel):
            raise AgentError(
                "Sensitive or internal paths cannot be opened as artifacts."
            )
        if self.session.indexer.matcher.is_ignored(rel, is_dir=path.is_dir()):
            raise AgentError("Artifact is excluded by repository rules.")
        return path

    def cmd_artifacts(self, inv: Invocation) -> None:
        args = list(inv.args)
        action = args.pop(0).lower() if args else "list"
        if action == "list" and not args:
            self.output("Agent-created artifacts in this session:")
            if not self.artifacts:
                self.output("  (none)")
            for record in self.artifacts:
                try:
                    self._artifact_path(record)
                    self.output(f"  {record.path} ({record.created_at})")
                except AgentError:
                    continue
            return
        if action == "path" and not args:
            self.output(
                f"Artifact files remain in the project: {self.session.project_root}"
            )
            self.output(
                f"Their session provenance is recorded in: {self.sessions.directory / (self.session_id + '.json')}"
            )
            return
        if action == "open" and len(args) == 1:
            selector = args[0]
            exact = [item for item in self.artifacts if item.path == selector]
            matches = exact or [
                item for item in self.artifacts if Path(item.path).name == selector
            ]
            if len(matches) != 1:
                raise AgentError(
                    "Artifact not found or ambiguous; use /artifacts list and specify its path."
                )
            path = self._artifact_path(matches[0])
            if not path.is_file():
                raise AgentError("The artifact file is no longer available.")
            self._local_tool(
                "read_file",
                {"path": path.relative_to(self.session.project_root).as_posix()},
            )
            return
        self.output("Usage: /artifacts [list|open <name>|path]")

    def cmd_project(self, inv: Invocation) -> None:
        if not inv.args:
            if inv.invoked == "cd":
                self.output("Usage: /cd <local-folder-path>")
            else:
                self.output(
                    f"Active local codebase folder: {self.session.project_root}"
                )
            return
        try:
            root = Path(" ".join(inv.args)).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise AgentError(f"Invalid project path: {exc}") from exc
        if not root.is_dir():
            raise AgentError(
                f"Local codebase directory '{root}' does not exist or is not a directory."
            )
        self._check_private_paths(root)
        self.save()
        new_root = self.session.switch_project_root(root)
        self.config_manager.set_project_root(new_root)
        self.credential_manager.set_project_root(new_root)
        self._new_identity()
        self.plan = None
        self.artifacts = []
        self.refresh_guidance()
        self.output(
            f"Switched active local codebase to '{new_root}' (previous file context cleared)."
        )
