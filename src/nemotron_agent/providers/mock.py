"""Offline deterministic and scriptable mock model client for tests and `--mock` mode."""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Callable, Sequence, Union

from nemotron_agent.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ConnectionTestResult,
    ProviderClient,
    ToolCallRequest,
    parse_fallback_tool_calls,
)

ScriptedStep = Union[
    CompletionResponse,
    Exception,
    Callable[[CompletionRequest], CompletionResponse],
]

_FILE_TOKEN_RE = re.compile(r"\b([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8})\b")


class MockProviderClient(ProviderClient):
    """Mock provider that never touches the network or requires API keys."""

    def __init__(
        self,
        scripted_responses: Sequence[ScriptedStep] | None = None,
        *,
        provider: str = "mock",
    ) -> None:
        self._provider = provider
        self._queue: deque[ScriptedStep] = deque(scripted_responses or ())
        self.requests: list[CompletionRequest] = []
        self.validation_calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return self._provider

    def validate_connection(self, model_id: str) -> ConnectionTestResult:
        self.validation_calls.append(model_id)
        return ConnectionTestResult(
            ok=True,
            provider=self._provider,
            model_id=model_id,
            latency_ms=1,
            message=f"Mock provider ready (offline) for model '{model_id}'.",
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)

        if self._queue:
            step = self._queue.popleft()
            if isinstance(step, Exception):
                raise step
            if callable(step):
                return step(request)
            # If the scripted response contains <tool_call> tags and no native tool_calls,
            # run through the same fallback parser so tests exercise realistic behavior.
            if not step.tool_calls and "<tool_call" in step.content.lower():
                cleaned, calls = parse_fallback_tool_calls(step.content)
                return step.model_copy(
                    update={
                        "content": cleaned,
                        "tool_calls": calls,
                        "used_fallback_parser": bool(calls),
                    }
                )
            return step

        return self._heuristic_response(request)

    def _heuristic_response(self, request: CompletionRequest) -> CompletionResponse:
        """Offline repository-aware heuristic workflow for `--mock` CLI usage."""
        # Find the last user prompt (ignoring fallback <tool_result> wrappers)
        last_user_idx = -1
        last_user_prompt = ""
        for idx in range(len(request.messages) - 1, -1, -1):
            msg = request.messages[idx]
            if msg.role == "user" and not msg.content.lstrip().startswith("<tool_result"):
                last_user_idx = idx
                last_user_prompt = msg.content
                break

        trailing = request.messages[last_user_idx + 1 :] if last_user_idx >= 0 else []
        tool_observations = [
            m.content
            for m in trailing
            if m.role == "tool" or (m.role == "user" and m.content.lstrip().startswith("<tool_result"))
        ]

        available_tool_names = {t.name for t in request.tools}
        lower_prompt = last_user_prompt.lower()

        if not tool_observations and available_tool_names:
            # Step 1: pick a bounded, appropriate tool based on user prompt
            call = self._select_initial_tool(last_user_prompt, lower_prompt, available_tool_names)
            if call is not None:
                return self._emit_tool_call(request, call, plan_note="Inspecting repository context in mock mode.")

        # Step 2: summarize observed tool outputs truthfully
        summary_lines = [
            "[MOCK MODE] Offline response generated without calling an external API.",
            "",
        ]
        if tool_observations:
            summary_lines.append("Verified observations from executed tools:")
            for obs in tool_observations:
                trimmed = obs.strip()
                if len(trimmed) > 800:
                    trimmed = trimmed[:800] + "\n... [truncated in mock summary]"
                summary_lines.append(trimmed)
        else:
            summary_lines.append(f"Received prompt: {last_user_prompt!r}")

        return CompletionResponse(
            content="\n".join(summary_lines),
            finish_reason="stop",
            usage={"prompt_tokens": 64, "completion_tokens": 64, "total_tokens": 128},
            raw_model=request.model_id,
        )

    def _select_initial_tool(
        self,
        prompt: str,
        lower_prompt: str,
        available: set[str],
    ) -> ToolCallRequest | None:
        # Explicit command request in mock mode
        if ("run pytest" in lower_prompt or "run tests" in lower_prompt) and "run_command" in available:
            return ToolCallRequest(
                id="mock-call-1",
                name="run_command",
                arguments_raw={"argv": ["pytest"], "reason": "Run project test suite requested by user"},
            )

        # Explicit file read if a relative filename is mentioned
        file_match = _FILE_TOKEN_RE.search(prompt)
        if file_match and ("read" in lower_prompt or "show" in lower_prompt or "inspect" in lower_prompt):
            if "read_file" in available:
                return ToolCallRequest(
                    id="mock-call-1",
                    name="read_file",
                    arguments_raw={"path": file_match.group(1), "start_line": 1, "end_line": 120},
                )

        # Search request
        if "search" in lower_prompt and "search_files" in available:
            words = [w for w in re.findall(r"[A-Za-z0-9_]{3,}", prompt) if w.lower() not in {"search", "for", "the", "files", "in", "repo"}]
            query = words[-1] if words else "def"
            return ToolCallRequest(
                id="mock-call-1",
                name="search_files",
                arguments_raw={"query": query, "max_results": 10},
            )

        # Default lightweight repo inspection
        if "repo_overview" in available:
            return ToolCallRequest(
                id="mock-call-1",
                name="repo_overview",
                arguments_raw={},
            )
        return None

    @staticmethod
    def _emit_tool_call(
        request: CompletionRequest,
        call: ToolCallRequest,
        *,
        plan_note: str,
    ) -> CompletionResponse:
        if request.capabilities.supports_tool_calling:
            return CompletionResponse(
                content=plan_note,
                tool_calls=[call],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 48, "completion_tokens": 24, "total_tokens": 72},
                raw_model=request.model_id,
            )
        # Exercise fallback `<tool_call>` parser when native tool calling is disabled
        args_obj = json.loads(call.arguments_raw) if isinstance(call.arguments_raw, str) else call.arguments_raw
        raw_text = (
            f"{plan_note}\n"
            f"<tool_call>\n{json.dumps({'name': call.name, 'arguments': args_obj})}\n</tool_call>"
        )
        cleaned, parsed_calls = parse_fallback_tool_calls(raw_text)
        return CompletionResponse(
            content=cleaned,
            tool_calls=parsed_calls,
            finish_reason="stop",
            usage={"prompt_tokens": 48, "completion_tokens": 24, "total_tokens": 72},
            raw_model=request.model_id,
            used_fallback_parser=True,
        )
