"""OpenAI-compatible provider client (configured for OpenRouter by default)."""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Callable

import httpx

from nemotron_agent.logging_utils import (
    AuthenticationError,
    EndpointUnavailableError,
    MalformedResponseError,
    ModelCapabilityError,
    ProviderError,
    RateLimitError,
    redact_text,
    register_secret,
)
from nemotron_agent.providers.base import (
    MAX_IMAGE_BYTES,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    ConnectionTestResult,
    ProviderClient,
    ToolCallRequest,
    build_fallback_tool_instructions,
    parse_fallback_tool_calls,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRY_DELAY_SECONDS = 10.0


class OpenAICompatibleProviderClient(ProviderClient):
    """HTTP client for OpenRouter and compatible `/chat/completions` endpoints."""

    def __init__(
        self,
        *,
        provider: str = "openrouter",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        sleep_fn: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        cleaned_url = base_url.strip().rstrip("/")
        if not cleaned_url:
            raise ProviderError(f"Base URL for provider '{provider}' is not configured.")
        cleaned_key = api_key.strip()
        if not cleaned_key:
            raise AuthenticationError(
                f"No API key provided for provider '{provider}'. "
                "Run `nemotron-agent auth set-key` or set the appropriate environment variable."
            )
        register_secret(cleaned_key)

        self._provider = provider.strip().lower()
        self._base_url = cleaned_url
        self._api_key = cleaned_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._sleep_fn = sleep_fn

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Title": "nemotron-agent",
        }
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    @property
    def provider_name(self) -> str:
        return self._provider

    def close(self) -> None:
        self._client.close()

    def validate_connection(self, model_id: str) -> ConnectionTestResult:
        """Perform a minimal connection check without sending any repository data."""
        start = time.monotonic()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8,
        }
        data = self._post_with_retries("/chat/completions", payload, model_id=model_id)
        _ = self._parse_completion_payload(data, expect_fallback_tools=False)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ConnectionTestResult(
            ok=True,
            provider=self._provider,
            model_id=model_id,
            latency_ms=elapsed_ms,
            message=f"Connected to {self._provider} and verified model '{model_id}'.",
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a normalized chat completion request respecting model capability flags."""
        images = [image for message in request.messages for image in message.images]
        if images and not request.capabilities.supports_vision:
            raise ModelCapabilityError(
                "The selected model is not configured for image input. Choose a vision-capable model before sending an image."
            )
        if any(message.images and message.role != "user" for message in request.messages):
            raise ModelCapabilityError("Images may only be attached to user messages.")
        if sum(len(image.data) for image in images) > MAX_IMAGE_BYTES:
            raise ModelCapabilityError("Image data exceeds the 5 MB limit for this request.")
        use_native_tools = bool(request.capabilities.supports_tool_calling and request.tools)
        messages_payload = self._serialize_messages(
            request.messages,
            use_native_tools=use_native_tools,
            fallback_tools=request.tools if not use_native_tools else [],
        )

        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": messages_payload,
        }
        max_tokens = request.max_tokens or request.capabilities.default_max_tokens
        if max_tokens:
            payload["max_tokens"] = max_tokens

        if request.capabilities.supports_temperature and request.temperature is not None:
            payload["temperature"] = request.temperature

        if use_native_tools:
            payload["tools"] = [tool.to_openai_tool() for tool in request.tools]
            payload["tool_choice"] = "auto"

        data = self._post_with_retries("/chat/completions", payload, model_id=request.model_id)
        return self._parse_completion_payload(data, expect_fallback_tools=bool(request.tools))

    def _serialize_messages(
        self,
        messages: list[ChatMessage],
        *,
        use_native_tools: bool,
        fallback_tools: list[Any],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        fallback_suffix = build_fallback_tool_instructions(fallback_tools) if fallback_tools else ""
        injected_fallback = False

        for msg in messages:
            if msg.role == "system":
                content = msg.content
                if fallback_suffix and not injected_fallback:
                    content = f"{content}{fallback_suffix}"
                    injected_fallback = True
                serialized.append({"role": "system", "content": content})
            elif msg.role == "user":
                if msg.images:
                    image_parts: list[dict[str, Any]] = [{"type": "text", "text": msg.content}]
                    for image in msg.images:
                        encoded = base64.b64encode(image.data).decode("ascii")
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
                        })
                    serialized.append({"role": "user", "content": image_parts})
                else:
                    serialized.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                if use_native_tools and msg.tool_calls:
                    tc_list = []
                    for tc in msg.tool_calls:
                        args_str = (
                            tc.arguments_raw
                            if isinstance(tc.arguments_raw, str)
                            else json.dumps(tc.arguments_raw)
                        )
                        tc_list.append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": args_str,
                                },
                            }
                        )
                    entry: dict[str, Any] = {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": tc_list,
                    }
                    serialized.append(entry)
                elif msg.tool_calls:
                    # Fallback mode history representation
                    blocks = []
                    if msg.content:
                        blocks.append(msg.content)
                    for tc in msg.tool_calls:
                        if isinstance(tc.arguments_raw, str):
                            try:
                                args_obj = json.loads(tc.arguments_raw)
                            except json.JSONDecodeError:
                                # A previously rejected malformed call may still be
                                # in history when a user switches tool modes. Do not
                                # crash or execute it while serializing the transcript.
                                args_obj = {
                                    "_invalid_arguments": redact_text(tc.arguments_raw)
                                }
                        else:
                            args_obj = tc.arguments_raw
                        blocks.append(
                            "<tool_call>\n"
                            + json.dumps({"name": tc.name, "arguments": args_obj})
                            + "\n</tool_call>"
                        )
                    serialized.append(
                        {"role": "assistant", "content": "\n".join(blocks)}
                    )
                else:
                    serialized.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool":
                if use_native_tools:
                    tool_entry: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "call-0",
                        "content": msg.content,
                    }
                    if msg.name:
                        tool_entry["name"] = msg.name
                    serialized.append(tool_entry)
                else:
                    # Convert tool output into user-role observation for endpoints without native tool role
                    label = msg.name or msg.tool_call_id or "tool"
                    serialized.append(
                        {
                            "role": "user",
                            "content": f"<tool_result name={json.dumps(label)}>\n{msg.content}\n</tool_result>",
                        }
                    )

        if fallback_suffix and not injected_fallback:
            serialized.insert(0, {"role": "system", "content": fallback_suffix.strip()})

        return serialized

    def _post_with_retries(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        model_id: str,
    ) -> dict[str, Any]:
        last_error: ProviderError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(endpoint, json=payload)
            except httpx.TimeoutException as exc:
                last_error = EndpointUnavailableError(
                    f"Request to {self._provider} ({self._base_url}) timed out after "
                    f"{self._timeout_seconds:.1f}s for model '{model_id}'."
                )
                if attempt < self._max_retries:
                    delay = min(_MAX_RETRY_DELAY_SECONDS, 0.5 * (2**attempt))
                    logger.info("Timeout on attempt %d; retrying in %.2fs", attempt + 1, delay)
                    self._sleep_fn(delay)
                    continue
                raise last_error from exc
            except httpx.TransportError as exc:
                last_error = EndpointUnavailableError(
                    f"Network error contacting {self._provider} ({self._base_url}): "
                    f"{redact_text(str(exc))}"
                )
                if attempt < self._max_retries:
                    delay = min(_MAX_RETRY_DELAY_SECONDS, 0.5 * (2**attempt))
                    logger.info("Network error on attempt %d; retrying in %.2fs", attempt + 1, delay)
                    self._sleep_fn(delay)
                    continue
                raise last_error from exc

            # Check HTTP status errors
            if response.status_code >= 400:
                err = self._classify_http_error(response, model_id=model_id)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                    delay = self._compute_retry_delay(response, attempt)
                    logger.info(
                        "HTTP %d from %s on attempt %d; retrying in %.2fs",
                        response.status_code,
                        self._provider,
                        attempt + 1,
                        delay,
                    )
                    self._sleep_fn(delay)
                    last_error = err
                    continue
                raise err

            # Parse JSON body
            try:
                data = response.json()
            except ValueError as exc:
                snippet = redact_text(response.text[:240])
                raise MalformedResponseError(
                    f"Provider '{self._provider}' returned non-JSON response (HTTP {response.status_code}): "
                    f"{snippet}"
                ) from exc

            if not isinstance(data, dict):
                raise MalformedResponseError(
                    f"Provider '{self._provider}' returned unexpected JSON root type "
                    f"{type(data).__name__}; expected object."
                )

            # OpenRouter sometimes embeds upstream errors inside HTTP 200 payloads.
            if "error" in data and data["error"]:
                embedded_code, embedded_msg = self._extract_error_details(data)
                status = embedded_code if isinstance(embedded_code, int) else 502
                err = self._map_status_to_error(status, embedded_msg, model_id=model_id)
                if status in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                    delay = min(_MAX_RETRY_DELAY_SECONDS, 0.5 * (2**attempt))
                    self._sleep_fn(delay)
                    last_error = err
                    continue
                raise err

            return data

        assert last_error is not None
        raise last_error

    def _compute_retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After", "").strip()
        if retry_after:
            try:
                parsed = float(retry_after)
                if parsed >= 0:
                    return min(_MAX_RETRY_DELAY_SECONDS, parsed)
            except ValueError:
                pass
        return min(_MAX_RETRY_DELAY_SECONDS, 0.5 * (2**attempt))

    def _classify_http_error(self, response: httpx.Response, *, model_id: str) -> ProviderError:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                _, detail = self._extract_error_details(body)
        except ValueError:
            detail = response.text[:300].strip()

        detail = redact_text(detail or f"HTTP {response.status_code}")
        return self._map_status_to_error(response.status_code, detail, model_id=model_id)

    def _map_status_to_error(
        self,
        status_code: int,
        detail: str,
        *,
        model_id: str,
    ) -> ProviderError:
        clean_detail = redact_text(detail)
        if len(clean_detail) > 2000:
            clean_detail = clean_detail[:2000] + "… [truncated]"
        lower = clean_detail.lower()

        if status_code in (401, 403):
            return AuthenticationError(
                f"Authentication failed with {self._provider} (HTTP {status_code}): {clean_detail}. "
                "Verify your API key with `nemotron-agent auth set-key` or check environment variables.",
                status_code=status_code,
            )

        if status_code == 429:
            return RateLimitError(
                f"Rate limit reached on {self._provider} for model '{model_id}' (HTTP 429): "
                f"{clean_detail}. Free-tier endpoints often enforce strict rate limits. "
                "Wait briefly and retry, or explicitly change models with `nemotron-agent config set-model`.",
                status_code=429,
            )

        if status_code == 404:
            return EndpointUnavailableError(
                f"Model '{model_id}' or endpoint at {self._base_url} was not found on "
                f"{self._provider} (HTTP 404): {clean_detail}. Provider model IDs and `:free` "
                "availability can change; check provider docs or select another model explicitly.",
                status_code=404,
            )

        if status_code in (400, 422) and any(
            kw in lower
            for kw in (
                "tool",
                "function call",
                "functions are not supported",
                "support tool use",
                "endpoints found that support tool",
                "response_format",
            )
        ):
            return ModelCapabilityError(
                f"Model '{model_id}' on {self._provider} rejected tool/structured parameters "
                f"(HTTP {status_code}): {clean_detail}. You can disable native tool calling for "
                "this model via `nemotron-agent config set-capability --tool-calling off` to use "
                "the validated structured-prompt fallback.",
                status_code=status_code,
            )

        if status_code >= 500:
            return EndpointUnavailableError(
                f"Provider '{self._provider}' returned upstream error (HTTP {status_code}) for "
                f"model '{model_id}': {clean_detail}",
                status_code=status_code,
            )

        return ProviderError(
            f"Provider '{self._provider}' error (HTTP {status_code}) for model '{model_id}': "
            f"{clean_detail}",
            status_code=status_code,
        )

    @staticmethod
    def _extract_error_details(body: dict[str, Any]) -> tuple[int | None, str]:
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            msg = str(err.get("message") or err)
            metadata = err.get("metadata")
            if isinstance(metadata, dict) and metadata.get("raw"):
                msg = f"{msg} ({metadata['raw']})"
            parsed_code = int(code) if isinstance(code, int) or (isinstance(code, str) and code.isdigit()) else None
            return parsed_code, msg
        if err is not None:
            return None, str(err)
        return None, json.dumps(body)[:300]

    def _parse_completion_payload(
        self,
        data: dict[str, Any],
        *,
        expect_fallback_tools: bool,
    ) -> CompletionResponse:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise MalformedResponseError(
                f"Provider '{self._provider}' returned response without a non-empty `choices` array."
            )

        first = choices[0]
        if not isinstance(first, dict):
            raise MalformedResponseError("Provider `choices[0]` is not a JSON object.")

        message = first.get("message")
        if not isinstance(message, dict):
            raise MalformedResponseError("Provider `choices[0].message` is missing or invalid.")

        raw_content = message.get("content")
        content_text = self._normalize_content(raw_content)

        native_tool_calls: list[ToolCallRequest] = []
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls is not None:
            if not isinstance(raw_tool_calls, list):
                raise MalformedResponseError("Provider `message.tool_calls` must be a list.")
            for idx, item in enumerate(raw_tool_calls, start=1):
                if not isinstance(item, dict):
                    raise MalformedResponseError(f"`tool_calls[{idx - 1}]` must be an object.")
                fn = item.get("function")
                if not isinstance(fn, dict):
                    raise MalformedResponseError(
                        f"`tool_calls[{idx - 1}].function` is missing or not an object."
                    )
                fn_name = fn.get("name")
                if not isinstance(fn_name, str) or not fn_name.strip():
                    raise MalformedResponseError(
                        f"`tool_calls[{idx - 1}].function.name` must be a non-empty string."
                    )
                fn_args = fn.get("arguments", "{}")
                if not isinstance(fn_args, (str, dict)):
                    raise MalformedResponseError(
                        f"`tool_calls[{idx - 1}].function.arguments` must be a JSON string or object."
                    )
                call_id = str(item.get("id") or f"call-{idx}")
                native_tool_calls.append(
                    ToolCallRequest(
                        id=call_id,
                        name=fn_name.strip(),
                        arguments_raw=fn_args,
                    )
                )

        used_fallback = False
        if not native_tool_calls and expect_fallback_tools and "<tool_call" in content_text.lower():
            content_text, native_tool_calls = parse_fallback_tool_calls(content_text)
            used_fallback = bool(native_tool_calls)

        usage_raw = data.get("usage")
        usage: dict[str, int] = {}
        if isinstance(usage_raw, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = usage_raw.get(key)
                if isinstance(val, int):
                    usage[key] = val

        finish_reason = first.get("finish_reason")
        raw_model = data.get("model")

        return CompletionResponse(
            content=content_text,
            tool_calls=native_tool_calls,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage=usage,
            raw_model=str(raw_model) if raw_model is not None else None,
            used_fallback_parser=used_fallback,
        )

    @staticmethod
    def _normalize_content(raw_content: Any) -> str:
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for block in raw_content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "\n".join(parts)
        raise MalformedResponseError(
            f"Unsupported `message.content` type: {type(raw_content).__name__}."
        )
