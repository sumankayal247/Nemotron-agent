"""Strict Pydantic input schemas for all bounded agent tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RepoOverviewArgs(BaseModel):
    """Arguments for `repo_overview` (takes no arguments)."""

    model_config = ConfigDict(extra="forbid")


class ListFilesArgs(BaseModel):
    """Arguments for `list_files`."""

    model_config = ConfigDict(extra="forbid")

    directory: str = Field(
        default=".",
        description="Relative directory path inside the repository to list (default '.').",
    )
    glob_pattern: str = Field(
        default="*",
        description="Filename or relative path glob filter such as '*.py' or 'src/*.py'.",
    )
    max_entries: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of file paths to return (1-500).",
    )


class SearchFilesArgs(BaseModel):
    """Arguments for `search_files`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=300,
        description="Substring or regular expression to search for across repository text files.",
    )
    directory: str = Field(
        default=".",
        description="Relative subdirectory inside the repository to scope the search.",
    )
    file_glob: str = Field(
        default="*",
        description="Optional file glob filter such as '*.py' or '*.ts'.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether matching should be case-sensitive.",
    )
    is_regex: bool = Field(
        default=False,
        description="Treat `query` as a Python regular expression instead of a literal string.",
    )
    max_results: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum matching lines to return (1-100).",
    )


class ReadFileArgs(BaseModel):
    """Arguments for `read_file`."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        max_length=500,
        description="Relative path to the file inside the repository.",
    )
    start_line: int = Field(
        default=1,
        ge=1,
        description="1-based starting line number to read.",
    )
    end_line: int = Field(
        default=200,
        ge=1,
        description="1-based inclusive ending line number to read (max 400 lines per call).",
    )

    @model_validator(mode="after")
    def _check_line_window(self) -> "ReadFileArgs":
        if self.end_line < self.start_line:
            raise ValueError("`end_line` must be greater than or equal to `start_line`.")
        if (self.end_line - self.start_line + 1) > 400:
            raise ValueError("Cannot read more than 400 lines in a single `read_file` call.")
        return self


class EditFileArgs(BaseModel):
    """Arguments for `edit_file` (narrow search-and-replace or new file creation with diff approval)."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        max_length=500,
        description="Relative path to the target file inside the repository.",
    )
    old_text: str = Field(
        default="",
        description=(
            "Exact existing text snippet to replace (must match uniquely in the file). "
            "Must be empty only when `create_if_missing` is true for a brand-new file."
        ),
    )
    new_text: str = Field(
        description="Replacement text (or initial content when `create_if_missing` is true).",
    )
    create_if_missing: bool = Field(
        default=False,
        description="Set true ONLY when creating a new file that does not yet exist.",
    )
    description: str = Field(
        default="Apply targeted code change",
        min_length=3,
        max_length=300,
        description="Concise explanation of why this edit is needed (shown to user with diff).",
    )

    @model_validator(mode="after")
    def _validate_edit_semantics(self) -> "EditFileArgs":
        if self.create_if_missing:
            if self.old_text != "":
                raise ValueError(
                    "When `create_if_missing` is true, `old_text` must be an empty string."
                )
        else:
            if not self.old_text:
                raise ValueError(
                    "`old_text` cannot be empty when editing an existing file; provide the exact "
                    "snippet to replace so unrelated lines are preserved."
                )
            if self.old_text == self.new_text:
                raise ValueError("`new_text` is identical to `old_text` (no-op edit).")
        return self


class RunCommandArgs(BaseModel):
    """Arguments for `run_command` (allowlisted development command as an argument array)."""

    model_config = ConfigDict(extra="forbid")

    argv: list[str] = Field(
        min_length=1,
        max_length=32,
        description=(
            "Command as an explicit argument array `[executable, arg1, arg2, ...]` — "
            "NEVER a shell string. Must match an allowlisted test, lint, or check command."
        ),
    )
    cwd: str = Field(
        default=".",
        description="Working directory relative to the project root (default '.').",
    )
    reason: str = Field(
        min_length=3,
        max_length=300,
        description="Clear reason for running this command (shown to the user before approval).",
    )

    @field_validator("argv")
    @classmethod
    def _validate_argv_elements(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for idx, token in enumerate(value):
            if not isinstance(token, str) or not token.strip():
                raise ValueError(f"`argv[{idx}]` must be a non-empty string.")
            if "\x00" in token:
                raise ValueError(f"`argv[{idx}]` contains an invalid null byte.")
            cleaned.append(token)
        return cleaned


def clean_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Export a clean OpenAI-compatible JSON Schema without top-level title clutter."""
    raw = model_cls.model_json_schema()
    raw.pop("title", None)
    raw.setdefault("type", "object")
    raw.setdefault("properties", {})
    raw["additionalProperties"] = False
    return raw
