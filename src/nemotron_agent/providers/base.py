"""Provider abstraction, normalized chat/tool models, and fallback tool-call parser."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemotron_agent.config import ModelCapabilities
from nemotron_agent.logging_utils import MalformedResponseError

ChatRole = Literal["system", "user", "assistant", "tool"]

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


class ToolCallRequest(BaseModel):
    """Normalized unvalidated tool invocation emitted by a model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments_raw: str | dict[str, Any]


class ChatMessage(BaseModel):
    """Normalized conversation message across providers."""

    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class ToolSpec(BaseModel):
    """JSON-Schema specification of an application tool."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class CompletionRequest(BaseModel):
    """Normalized request sent to a ProviderClient."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    messages: list[ChatMessage]
    tools: list[ToolSpec] = Field(default_factory=list)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    max_tokens: int | None = None
    temperature: float | None = 0.1


class CompletionResponse(BaseModel):
    """Normalized response returned by a ProviderClient."""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw_model: str | None = None
    used_fallback_parser: bool = False


class ConnectionTestResult(BaseModel):
    """Outcome of a lightweight, repository-free provider connection validation."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    provider: str
    model_id: str
    latency_ms: int
    message: str


class ProviderClient(ABC):
    """Abstract interface for model providers (OpenRouter, OpenAI-compatible, Mock)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier of the active provider."""

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a single chat completion step."""

    @abstractmethod
    def validate_connection(self, model_id: str) -> ConnectionTestResult:
        """Validate credentials and model reachability WITHOUT sending repository files."""

    def close(self) -> None:
        """Release underlying network resources if applicable."""


def build_fallback_tool_instructions(tools: list[ToolSpec]) -> str:
    """Build strict prompt instructions for models without native OpenAI tool calling."""
    if not tools:
        return ""
    tool_defs = [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]
    schema_text = json.dumps(tool_defs, indent=2)
    return (
        "\n\n## Structured Tool Request Format (Fallback Mode)\n"
        "Native function-calling is disabled for this model. When you need to call one or more "
        "tools, emit ONLY valid JSON inside `<tool_call>...</tool_call>` tags using this exact "
        "object structure:\n"
        '<tool_call>\n{"name": "<tool_name>", "arguments": { ... }}\n</tool_call>\n'
        "Rules:\n"
        "- Do not invent tool names or extra argument keys.\n"
        "- Every `<tool_call>` block MUST contain a single strict JSON object with `name` (string) "
        "and `arguments` (JSON object).\n"
        "- When you have enough information to answer the user without calling further tools, "
        "respond directly without any `<tool_call>` tags.\n\n"
        f"Available tools JSON specification:\n{schema_text}\n"
    )


def parse_fallback_tool_calls(text: str) -> tuple[str, list[ToolCallRequest]]:
    """Extract and strictly validate `<tool_call>{...}</tool_call>` blocks from model text.

    Returns `(cleaned_prose_content, parsed_tool_calls)`.
    Raises `MalformedResponseError` if `<tool_call>` tags are malformed or contain invalid JSON.
    """
    if "<tool_call" not in text.lower():
        return text, []

    lower = text.lower()
    open_count = lower.count("<tool_call>")
    close_count = lower.count("</tool_call>")
    if open_count != close_count or open_count == 0:
        raise MalformedResponseError(
            "Model emitted an unclosed or malformed `<tool_call>` block."
        )

    matches = list(_TOOL_CALL_TAG_RE.finditer(text))
    if len(matches) != open_count:
        raise MalformedResponseError(
            "Model emitted nested or malformed `<tool_call>` tags that failed parsing."
        )

    calls: list[ToolCallRequest] = []
    for idx, match in enumerate(matches, start=1):
        raw_body = match.group(1).strip()
        # Strip optional markdown code fences inside <tool_call> if the model added them.
        if raw_body.startswith("```"):
            lines = raw_body.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                raw_body = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise MalformedResponseError(
                f"Model emitted invalid JSON inside `<tool_call>` #{idx}: {exc.msg}"
            ) from exc

        if not isinstance(parsed, dict):
            raise MalformedResponseError(
                f"Model `<tool_call>` #{idx} must be a JSON object, got {type(parsed).__name__}."
            )
        extra_keys = set(parsed.keys()) - {"name", "arguments"}
        if extra_keys:
            raise MalformedResponseError(
                f"Model `<tool_call>` #{idx} contains unexpected top-level keys: {sorted(extra_keys)}."
            )
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise MalformedResponseError(
                f"Model `<tool_call>` #{idx} is missing a valid string `name`."
            )
        if not isinstance(arguments, dict):
            raise MalformedResponseError(
                f"Model `<tool_call>` #{idx} `arguments` must be a JSON object."
            )
        calls.append(
            ToolCallRequest(
                id=f"fallback-call-{idx}",
                name=name.strip(),
                arguments_raw=arguments,
            )
        )

    cleaned_content = _TOOL_CALL_TAG_RE.sub("", text).strip()
    return cleaned_content, calls
