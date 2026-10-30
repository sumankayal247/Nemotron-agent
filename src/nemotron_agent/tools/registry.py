"""Bounded tool registry, strict schema validation, and untrusted-output framing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from nemotron_agent.logging_utils import AgentError, ToolValidationError, redact_text
from nemotron_agent.providers.base import ToolCallRequest, ToolSpec
from nemotron_agent.repo.context import RepositoryContextService
from nemotron_agent.repo.discovery import RepositoryIndexer
from nemotron_agent.tools.commands import AllowlistedCommandRunner
from nemotron_agent.tools.files import FileEditor
from nemotron_agent.tools.schemas import (
    EditFileArgs,
    ListFilesArgs,
    ReadFileArgs,
    RepoOverviewArgs,
    RunCommandArgs,
    SearchFilesArgs,
    clean_json_schema,
)


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], str]
    mutating: bool = False


@dataclass(frozen=True)
class ToolExecutionResult:
    call_id: str
    tool_name: str
    ok: bool
    raw_output: str
    framed_output: str


def wrap_untrusted_tool_output(tool_name: str, output: str) -> str:
    """Wrap repository/tool text in untrusted-data tags so models treat it as data, not instructions."""
    clean = redact_text(output)
    return (
        f'<untrusted_repo_data tool="{tool_name}">\n'
        f"{clean}\n"
        f"</untrusted_repo_data>"
    )


class ToolRegistry:
    """Validates and executes the bounded set of repository, file, and command tools."""

    def __init__(
        self,
        *,
        indexer: RepositoryIndexer,
        context_service: RepositoryContextService,
        file_editor: FileEditor,
        command_runner: AllowlistedCommandRunner,
        max_output_bytes: int = 32768,
    ) -> None:
        self._indexer = indexer
        self._context = context_service
        self._editor = file_editor
        self._runner = command_runner
        self._max_output_bytes = max_output_bytes
        self._tools: dict[str, RegisteredTool] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        self._register(
            RegisteredTool(
                name="repo_overview",
                description=(
                    "Lightweight repository inspection: identifies project root, languages, "
                    "package/build manifests, top-level layout, and inferred dev commands."
                ),
                args_model=RepoOverviewArgs,
                handler=self._handle_repo_overview,
                mutating=False,
            )
        )
        self._register(
            RegisteredTool(
                name="list_files",
                description=(
                    "List allowed files inside the repository respecting .gitignore and "
                    "secret/binary exclusions."
                ),
                args_model=ListFilesArgs,
                handler=self._handle_list_files,
                mutating=False,
            )
        )
        self._register(
            RegisteredTool(
                name="search_files",
                description=(
                    "Targeted text or regex search across allowed repository files; returns "
                    "matching file paths, line numbers, and snippets."
                ),
                args_model=SearchFilesArgs,
                handler=self._handle_search_files,
                mutating=False,
            )
        )
        self._register(
            RegisteredTool(
                name="read_file",
                description=(
                    "Read a bounded line range (up to 400 lines per call) from a repository "
                    "text file with line numbers."
                ),
                args_model=ReadFileArgs,
                handler=self._handle_read_file,
                mutating=False,
            )
        )
        self._register(
            RegisteredTool(
                name="edit_file",
                description=(
                    "Propose a narrow search-and-replace edit to an existing file (must have been "
                    "read first) or create a new file. Shows a unified diff and requires explicit "
                    "user approval before writing."
                ),
                args_model=EditFileArgs,
                handler=self._handle_edit_file,
                mutating=True,
            )
        )
        self._register(
            RegisteredTool(
                name="run_command",
                description=(
                    "Request execution of an allowlisted development command (tests, linter, "
                    "formatter, typecheck, or read-only git status/diff) as an argv list without "
                    "a shell. Requires explicit user approval."
                ),
                args_model=RunCommandArgs,
                handler=self._handle_run_command,
                mutating=True,
            )
        )

    def _register(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool

    def get_tool_specs(self, *, read_only: bool = False) -> list[ToolSpec]:
        """Export JSON-Schema specs for available tools."""
        specs: list[ToolSpec] = []
        for tool in self._tools.values():
            if read_only and tool.mutating:
                continue
            specs.append(
                ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=clean_json_schema(tool.args_model),
                )
            )
        return specs

    def validate_call(self, call: ToolCallRequest) -> tuple[RegisteredTool, BaseModel]:
        """Strictly validate tool existence, JSON syntax, and Pydantic schema in app code."""
        name = call.name.strip()
        tool = self._tools.get(name)
        if tool is None:
            allowed = ", ".join(sorted(self._tools.keys()))
            raise ToolValidationError(
                f"Rejected unknown tool '{name}'. Allowed tools: {allowed}."
            )

        raw_args = call.arguments_raw
        if isinstance(raw_args, str):
            stripped = raw_args.strip()
            if not stripped:
                parsed_dict: Any = {}
            else:
                try:
                    parsed_dict = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ToolValidationError(
                        f"Invalid JSON arguments for tool '{name}': {exc.msg}"
                    ) from exc
        elif isinstance(raw_args, dict):
            parsed_dict = raw_args
        else:
            raise ToolValidationError(
                f"Arguments for tool '{name}' must be a JSON object, got {type(raw_args).__name__}."
            )

        if not isinstance(parsed_dict, dict):
            raise ToolValidationError(
                f"Arguments for tool '{name}' must decode to a JSON object, got {type(parsed_dict).__name__}."
            )

        # Smart defaults for read_file when only start_line or only end_line is given
        if name == "read_file" and isinstance(parsed_dict, dict):
            if "start_line" in parsed_dict and "end_line" not in parsed_dict:
                try:
                    s_line = int(parsed_dict["start_line"])
                    parsed_dict["end_line"] = s_line + 199
                except (ValueError, TypeError):
                    pass
            elif "end_line" in parsed_dict and "start_line" not in parsed_dict:
                try:
                    e_line = int(parsed_dict["end_line"])
                    parsed_dict["start_line"] = max(1, e_line - 199)
                except (ValueError, TypeError):
                    pass

        try:
            validated_args = tool.args_model.model_validate(parsed_dict)
        except ValidationError as exc:
            concise_errors = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolValidationError(
                f"Invalid arguments for tool '{name}': {concise_errors}"
            ) from exc

        return tool, validated_args

    def execute_call(self, call: ToolCallRequest) -> ToolExecutionResult:
        """Validate and execute a single tool call, catching and reporting errors safely."""
        try:
            tool, validated_args = self.validate_call(call)
            raw_out = tool.handler(validated_args)
            bounded_out = self._bound_output(raw_out)
            return ToolExecutionResult(
                call_id=call.id,
                tool_name=tool.name,
                ok=True,
                raw_output=bounded_out,
                framed_output=wrap_untrusted_tool_output(tool.name, bounded_out),
            )
        except AgentError as exc:
            err_msg = f"Tool error ({call.name}): {redact_text(str(exc))}"
            return ToolExecutionResult(
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                raw_output=err_msg,
                framed_output=wrap_untrusted_tool_output(call.name, err_msg),
            )

    def _bound_output(self, text: str) -> str:
        clean = redact_text(text)
        encoded = clean.encode("utf-8", errors="replace")
        if len(encoded) <= self._max_output_bytes:
            return clean
        clipped = encoded[: self._max_output_bytes].decode("utf-8", errors="ignore")
        return clipped + f"\n... [tool output truncated at {self._max_output_bytes} bytes]"

    # --- Tool Handlers ---

    def _handle_repo_overview(self, _args: BaseModel) -> str:
        overview = self._indexer.inspect_overview()
        return overview.format_text()

    def _handle_list_files(self, args: BaseModel) -> str:
        assert isinstance(args, ListFilesArgs)
        return self._context.list_files(
            directory=args.directory,
            glob_pattern=args.glob_pattern,
            max_entries=args.max_entries,
        )

    def _handle_search_files(self, args: BaseModel) -> str:
        assert isinstance(args, SearchFilesArgs)
        return self._context.search_files(
            query=args.query,
            directory=args.directory,
            file_glob=args.file_glob,
            case_sensitive=args.case_sensitive,
            is_regex=args.is_regex,
            max_results=args.max_results,
        )

    def _handle_read_file(self, args: BaseModel) -> str:
        assert isinstance(args, ReadFileArgs)
        return self._context.read_file_lines(
            path=args.path,
            start_line=args.start_line,
            end_line=args.end_line,
        )

    def _handle_edit_file(self, args: BaseModel) -> str:
        assert isinstance(args, EditFileArgs)
        return self._editor.apply_edit(
            path=args.path,
            old_text=args.old_text,
            new_text=args.new_text,
            create_if_missing=args.create_if_missing,
            description=args.description,
        )

    def _handle_run_command(self, args: BaseModel) -> str:
        assert isinstance(args, RunCommandArgs)
        return self._runner.run(
            args.argv,
            cwd=args.cwd,
            reason=args.reason,
        )
