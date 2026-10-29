"""Unit tests for OpenRouter/OpenAI-compatible client, mock client, and first-run setup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from nemotron_agent.cli import build_subcommand_parser, handle_config_or_auth_command
from nemotron_agent.config import ConfigManager, ModelCapabilities
from nemotron_agent.credentials import CredentialManager
from nemotron_agent.logging_utils import (
    AuthenticationError,
    EndpointUnavailableError,
    MalformedResponseError,
    ModelCapabilityError,
    RateLimitError,
)
from nemotron_agent.providers.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    ToolSpec,
    parse_fallback_tool_calls,
)
from nemotron_agent.providers.mock import MockProviderClient
from nemotron_agent.providers.openrouter import OpenAICompatibleProviderClient
from nemotron_agent.setup_wizard import ensure_ready_or_run_setup, run_interactive_setup


class FakeKeyring:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.data: dict[tuple[str, str], str] = {}

    def backend_Description(self) -> tuple[bool, str, str | None]:
        return (self.available, "FakeKeyring", None if self.available else "no keyring")

    def get_password(self, service: str, username: str) -> str | None:
        return self.data.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.data[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.data.pop((service, username), None)


def test_validate_connection_sends_no_repo_files_or_tools() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-or-v1-testkey123456"
        body = json.loads(request.content.decode("utf-8"))
        captured.append(body)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            },
        )

    client = OpenAICompatibleProviderClient(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-v1-testkey123456",
        transport=httpx.MockTransport(handler),
    )
    res = client.validate_connection("nvidia/nemotron-3.5-lightning:free")
    client.close()

    assert res.ok is True
    assert len(captured) == 1
    assert "tools" not in captured[0]
    assert captured[0]["max_tokens"] == 8
    assert captured[0]["messages"] == [{"role": "user", "content": "Reply with OK."}]


def test_capability_native_vs_fallback_tool_calling() -> None:
    tool_spec = ToolSpec(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        if "tools" in body:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path": "README.md"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": 'Checking file.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "README.md"}}\n</tool_call>',
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    client = OpenAICompatibleProviderClient(
        api_key="sk-or-v1-testkey123456",
        transport=httpx.MockTransport(handler),
    )

    # 1. Native tool calling enabled
    native_resp = client.complete(
        CompletionRequest(
            model_id="nvidia/nemotron-3.5-lightning:free",
            messages=[ChatMessage(role="user", content="Read README")],
            tools=[tool_spec],
            capabilities=ModelCapabilities(supports_tool_calling=True),
        )
    )
    assert "tools" in seen_payloads[0]
    assert len(native_resp.tool_calls) == 1
    assert native_resp.tool_calls[0].name == "read_file"
    assert native_resp.used_fallback_parser is False

    # 2. Native tool calling disabled (e.g., Super VL preset)
    fallback_resp = client.complete(
        CompletionRequest(
            model_id="nvidia/nemotron-3.5-super-vl:free",
            messages=[
                ChatMessage(role="system", content="You are an agent."),
                ChatMessage(role="user", content="Read README"),
            ],
            tools=[tool_spec],
            capabilities=ModelCapabilities(supports_tool_calling=False, supports_vision=True),
        )
    )
    assert "tools" not in seen_payloads[1]
    assert "Structured Tool Request Format" in seen_payloads[1]["messages"][0]["content"]
    assert len(fallback_resp.tool_calls) == 1
    assert fallback_resp.tool_calls[0].name == "read_file"
    assert fallback_resp.tool_calls[0].arguments_raw == {"path": "README.md"}
    assert fallback_resp.content == "Checking file."
    assert fallback_resp.used_fallback_parser is True
    client.close()


def test_fallback_parser_rejects_malformed_json() -> None:
    with pytest.raises(MalformedResponseError, match="invalid JSON"):
        parse_fallback_tool_calls("<tool_call>{not valid json}</tool_call>")

    with pytest.raises(MalformedResponseError, match="unclosed or malformed"):
        parse_fallback_tool_calls("<tool_call>{\"name\": \"read_file\"}")

    with pytest.raises(MalformedResponseError, match="unexpected top-level keys"):
        parse_fallback_tool_calls('<tool_call>{"name": "read_file", "arguments": {}, "shell": "rm -rf"}</tool_call>')


def test_provider_error_classification_and_bounded_retries() -> None:
    sleeps: list[float] = []
    attempts = 0

    def rate_limit_then_ok(_req: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.25"},
                json={"error": {"message": "Rate limited on free tier"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "OK"}}]},
        )

    client = OpenAICompatibleProviderClient(
        api_key="sk-or-v1-retrykey123456",
        max_retries=2,
        sleep_fn=sleeps.append,
        transport=httpx.MockTransport(rate_limit_then_ok),
    )
    res = client.validate_connection("nvidia/nemotron-3.5-lightning:free")
    assert res.ok is True
    assert attempts == 3
    assert sleeps == [0.25, 0.25]
    client.close()

    # Persistent 429 raises RateLimitError without switching models
    client_429 = OpenAICompatibleProviderClient(
        api_key="sk-or-v1-retrykey123456",
        max_retries=1,
        sleep_fn=lambda _s: None,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(429, json={"error": {"message": "Too many requests"}})
        ),
    )
    with pytest.raises(RateLimitError, match="nvidia/nemotron-3-ultra-550b-a55b:free"):
        client_429.validate_connection("nvidia/nemotron-3-ultra-550b-a55b:free")
    client_429.close()

    # 401 fails immediately without retrying and redacts key
    auth_calls = 0

    def unauth_handler(_r: httpx.Request) -> httpx.Response:
        nonlocal auth_calls
        auth_calls += 1
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid key sk-or-v1-badkey99999999"}},
        )

    client_401 = OpenAICompatibleProviderClient(
        api_key="sk-or-v1-badkey99999999",
        max_retries=3,
        sleep_fn=lambda _s: None,
        transport=httpx.MockTransport(unauth_handler),
    )
    with pytest.raises(AuthenticationError) as exc_info:
        client_401.validate_connection("nvidia/nemotron-3.5-lightning:free")
    assert auth_calls == 1
    assert "sk-or-v1-badkey99999999" not in str(exc_info.value)
    client_401.close()

    # 400 unsupported tool parameter raises ModelCapabilityError
    client_400 = OpenAICompatibleProviderClient(
        api_key="sk-or-v1-testkey123456",
        max_retries=2,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                400,
                json={"error": {"message": "No endpoints found that support tool use"}},
            )
        ),
    )
    with pytest.raises(ModelCapabilityError, match="set-capability --tool-calling off"):
        client_400.validate_connection("nvidia/nemotron-3.5-super-vl:free")
    client_400.close()

    # 404 raises EndpointUnavailableError
    client_404 = OpenAICompatibleProviderClient(
        api_key="sk-or-v1-testkey123456",
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(404, json={"error": {"message": "Model not found"}})
        ),
    )
    with pytest.raises(EndpointUnavailableError, match="was not found"):
        client_404.validate_connection("nvidia/nonexistent:free")
    client_404.close()


def test_first_run_setup_and_subsequent_reuse(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "user_config"
    kr = FakeKeyring(available=True)
    cfg_mgr = ConfigManager(config_dir=cfg_dir, env={})
    cred_mgr = CredentialManager(config_dir=cfg_dir, env={}, keyring_adapter=kr)

    outputs: list[str] = []
    # User chooses preset 1 (Nemotron 3.5 Lightning), accepts default OpenRouter base URL
    inputs = iter(["1", ""])
    secret_inputs = iter(["sk-or-v1-firstsetupkey12345"])
    mock_client = MockProviderClient(provider="openrouter")

    outcome = run_interactive_setup(
        config_manager=cfg_mgr,
        credential_manager=cred_mgr,
        input_fn=lambda _p: next(inputs),
        secret_input_fn=lambda _p: next(secret_inputs),
        output_fn=outputs.append,
        client_factory=lambda *_args: mock_client,
    )

    assert outcome.completed is True
    assert outcome.config is not None
    assert outcome.config.model_id == "nvidia/nemotron-3.5-lightning:free"
    assert outcome.config.provider == "openrouter"
    assert mock_client.validation_calls == ["nvidia/nemotron-3.5-lightning:free"]

    # Ensure key was NEVER printed in setup output
    joined_out = "\n".join(outputs)
    assert "sk-or-v1-firstsetupkey12345" not in joined_out
    assert "Nemotron 3 Ultra" in joined_out

    # Subsequent launch reuses saved configuration without asking any prompts!
    def fail_if_prompted(_prompt: str) -> str:
        raise AssertionError("Should not prompt on subsequent launch!")

    second = ensure_ready_or_run_setup(
        cfg_mgr,
        cred_mgr,
        resolved=cfg_mgr.resolve(),
        interactive=True,
        input_fn=fail_if_prompted,
        secret_input_fn=fail_if_prompted,
        output_fn=lambda _s: None,
    )
    assert second.completed is True
    assert second.config is not None
    assert second.config.model_id == "nvidia/nemotron-3.5-lightning:free"


def test_setup_validation_failure_allows_retry_or_exit_without_auto_switching(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "user_config"
    kr = FakeKeyring(available=True)
    cfg_mgr = ConfigManager(config_dir=cfg_dir, env={})
    cred_mgr = CredentialManager(config_dir=cfg_dir, env={}, keyring_adapter=kr)

    # Attempt 1 fails with AuthenticationError, user declines retry ("n")
    inputs = iter(["4", "", "n"])
    secrets = iter(["sk-or-v1-wrongkey123456"])
    outputs: list[str] = []

    class FailingClient(MockProviderClient):
        def validate_connection(self, model_id: str):
            raise AuthenticationError("401 Unauthorized")

    outcome = run_interactive_setup(
        config_manager=cfg_mgr,
        credential_manager=cred_mgr,
        input_fn=lambda _p: next(inputs),
        secret_input_fn=lambda _p: next(secrets),
        output_fn=outputs.append,
        client_factory=lambda *_args: FailingClient(provider="openrouter"),
    )
    assert outcome.completed is False
    assert not cfg_mgr.exists()
    assert any("never switches models or providers automatically" in line for line in outputs)


def test_cli_config_and_auth_commands(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    kr = FakeKeyring(available=True)
    cfg_mgr = ConfigManager(config_dir=cfg_dir, env={})
    cred_mgr = CredentialManager(config_dir=cfg_dir, env={}, keyring_adapter=kr)
    parser = build_subcommand_parser()
    out: list[str] = []

    # Set key
    rc = handle_config_or_auth_command(
        parser.parse_args(["auth", "set-key"]),
        config_manager=cfg_mgr,
        credential_manager=cred_mgr,
        secret_input_fn=lambda _p: "sk-or-v1-clikey123456789",
        output_fn=out.append,
        client_factory=lambda *_a: MockProviderClient(provider="openrouter"),
    )
    assert rc == 0

    # Set model to preset 2 (Nemotron 3.5 Super VL)
    rc = handle_config_or_auth_command(
        parser.parse_args(["config", "set-model", "2"]),
        config_manager=cfg_mgr,
        credential_manager=cred_mgr,
        output_fn=out.append,
        client_factory=lambda *_a: MockProviderClient(provider="openrouter"),
    )
    assert rc == 0

    # Show config (must redact key)
    out.clear()
    rc = handle_config_or_auth_command(
        parser.parse_args(["config", "show"]),
        config_manager=cfg_mgr,
        credential_manager=cred_mgr,
        output_fn=out.append,
    )
    assert rc == 0
    rendered = "\n".join(out)
    assert "nvidia/nemotron-3.5-super-vl:free" in rendered
    assert "[REDACTED]" in rendered
    assert "sk-or-v1-clikey123456789" not in rendered

    # Clear credentials
    out.clear()
    rc = handle_config_or_auth_command(
        parser.parse_args(["auth", "clear"]),
        config_manager=cfg_mgr,
        credential_manager=cred_mgr,
        output_fn=out.append,
    )
    assert rc == 0
    assert cred_mgr.resolve_api_key("openrouter") is None


def test_fallback_history_serialization_tolerates_previously_rejected_tool_json() -> (
    None
):
    from nemotron_agent.providers.base import ToolCallRequest

    client = OpenAICompatibleProviderClient(
        api_key="sk-or-v1-testkey123456",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )
        ),
    )
    request = CompletionRequest(
        model_id="model",
        capabilities=ModelCapabilities(supports_tool_calling=False),
        messages=[
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCallRequest(
                        id="bad", name="read_file", arguments_raw="{bad json"
                    )
                ],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="bad",
                name="read_file",
                content="Invalid JSON arguments",
            ),
            ChatMessage(role="user", content="Continue"),
        ],
    )
    assert client.complete(request).content == "ok"
    client.close()


def test_setup_shows_provider_key_link_and_uses_hidden_paste(tmp_path: Path) -> None:
    cfg = ConfigManager(config_dir=tmp_path / "cfg", env={})
    creds = CredentialManager(
        config_dir=tmp_path / "cfg", env={}, keyring_adapter=FakeKeyring()
    )
    selected = cfg.resolve(cli_model="3")
    outputs: list[str] = []
    secret_prompts: list[str] = []
    client = MockProviderClient(provider="openrouter")

    def hidden(prompt: str) -> str:
        secret_prompts.append(prompt)
        return "sk-or-v1-privateuserkey12345"

    result = run_interactive_setup(
        cfg,
        creds,
        input_fn=lambda _: pytest.fail(
            "the already-selected model must not be requested again"
        ),
        secret_input_fn=hidden,
        output_fn=outputs.append,
        client_factory=lambda *_: client,
        initial_selection=selected,
    )
    assert result.completed and result.config.model_id == selected.model_id
    assert "https://openrouter.ai/settings/keys" in "\n".join(outputs)
    assert (
        "paste" in secret_prompts[0].lower() and "hidden" in secret_prompts[0].lower()
    )
    assert "sk-or-v1-privateuserkey12345" not in "\n".join(outputs)
    assert client.validation_calls == [selected.model_id]


def test_image_request_uses_multimodal_payload_and_checks_capability():
    import base64

    from nemotron_agent.providers.base import ImageAttachment

    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "I see it."}}]
            },
        )

    client = OpenAICompatibleProviderClient(
        api_key="sk-or-v1-testkey123456", transport=httpx.MockTransport(handler)
    )
    image = ImageAttachment(
        mime_type="image/png", data=b"private-image-bytes", width=2, height=1
    )
    tool = ToolSpec(
        name="read_file",
        description="Read",
        parameters={"type": "object", "properties": {}},
    )
    for native in (True, False):
        request = CompletionRequest(
            model_id="vision-model",
            messages=[
                ChatMessage(role="system", content="Be safe."),
                ChatMessage(role="user", content="Describe this", images=[image]),
            ],
            tools=[tool],
            capabilities=ModelCapabilities(
                supports_vision=True, supports_tool_calling=native
            ),
        )
        assert "images" not in request.model_dump()["messages"][1]
        client.complete(request)
        parts = captured[-1]["messages"][1]["content"]
        assert parts[0] == {"type": "text", "text": "Describe this"}
        data_url = parts[1]["image_url"]["url"]
        assert data_url.startswith("data:image/png;base64,")
        assert base64.b64decode(data_url.split(",", 1)[1]) == image.data
        assert ("tools" in captured[-1]) is native

    with pytest.raises(ModelCapabilityError, match="image input"):
        client.complete(
            CompletionRequest(
                model_id="text-only",
                messages=[ChatMessage(role="user", content="Describe", images=[image])],
                capabilities=ModelCapabilities(supports_vision=False),
            )
        )
    assert len(captured) == 2
    client.close()


def test_provider_error_does_not_echo_image_data():
    import base64

    from nemotron_agent.logging_utils import ProviderError
    from nemotron_agent.providers.base import ImageAttachment

    data = b"a-private-image-with-sensitive-details"
    encoded = base64.b64encode(data).decode()

    def handler(_request):
        return httpx.Response(
            400,
            json={
                "error": {"message": f"Invalid image data:image/png;base64,{encoded}"}
            },
        )

    client = OpenAICompatibleProviderClient(
        api_key="sk-or-v1-testkey123456", transport=httpx.MockTransport(handler)
    )
    request = CompletionRequest(
        model_id="vision-model",
        messages=[
            ChatMessage(
                role="user",
                content="Explain",
                images=[
                    ImageAttachment(mime_type="image/png", data=data, width=1, height=1)
                ],
            )
        ],
        capabilities=ModelCapabilities(supports_vision=True),
    )
    with pytest.raises(ProviderError) as caught:
        client.complete(request)
    assert encoded not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    client.close()


def test_secret_input_refuses_echoing_fallback(monkeypatch):
    import warnings

    import nemotron_agent.setup_wizard as setup
    from nemotron_agent.logging_utils import AgentError

    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: False)
    with pytest.raises(AgentError, match="interactive terminal"):
        setup.secure_secret_input("Key: ")
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)

    def echo_warning(_prompt):
        warnings.warn("Cannot disable echo", setup.getpass.GetPassWarning)
        pytest.fail("unmasked input must not be read")

    monkeypatch.setattr(setup.getpass, "getpass", echo_warning)
    with pytest.raises(AgentError, match="cannot hide"):
        setup.secure_secret_input("Key: ")


@pytest.mark.parametrize(
    "base_url,warning",
    [
        ("http://openrouter.ai/api/v1", "not encrypted"),
        ("https://example.invalid/api/v1", "not the official OpenRouter host"),
        ("https://user:password@openrouter.ai/api/v1", "Invalid API endpoint"),
        ("https://openrouter.ai/api/v1?api_key=do-not-log", "Invalid API endpoint"),
    ],
)
def test_setup_requires_approval_before_sending_key_to_unsafe_endpoint(
    tmp_path, base_url, warning
):
    cfg = ConfigManager(config_dir=tmp_path / "cfg", env={})
    creds = CredentialManager(
        config_dir=tmp_path / "cfg", env={}, keyring_adapter=FakeKeyring()
    )
    selected = cfg.resolve(cli_model="1", cli_base_url=base_url)
    outputs = []
    result = run_interactive_setup(
        cfg,
        creds,
        initial_selection=selected,
        input_fn=lambda _: "n",
        secret_input_fn=lambda _: pytest.fail(
            "never ask for key before endpoint approval"
        ),
        output_fn=outputs.append,
        client_factory=lambda *_: pytest.fail("do not connect before approval"),
    )
    assert not result.completed
    assert warning in "\n".join(outputs)
    assert "do-not-log" not in "\n".join(outputs)


def test_explicit_cli_model_is_not_asked_again_when_key_missing(tmp_path):
    cfg = ConfigManager(config_dir=tmp_path / "cfg", env={})
    creds = CredentialManager(
        config_dir=tmp_path / "cfg", env={}, keyring_adapter=FakeKeyring()
    )
    selected = cfg.resolve(cli_model="3")
    prompts = []
    outcome = ensure_ready_or_run_setup(
        cfg,
        creds,
        resolved=selected,
        input_fn=lambda prompt: prompts.append(prompt) or "",
        secret_input_fn=lambda _: "sk-or-v1-explicit123456789",
        output_fn=lambda _: None,
        client_factory=lambda *_: MockProviderClient(
            scripted_responses=[CompletionResponse(content="OK")]
        ),
    )
    assert outcome.completed and outcome.config.model_id == selected.model_id
    assert not any("Choose a model preset" in prompt for prompt in prompts)


def test_setup_respects_new_key_over_old_environment_in_current_session(tmp_path):
    env = {
        "OPENROUTER_API_KEY": "sk-or-v1-oldenvironment123456",
        "NEMOTRON_AGENT_MODEL": "1",
    }
    cfg = ConfigManager(config_dir=tmp_path / "cfg", env=env)
    creds = CredentialManager(
        config_dir=tmp_path / "cfg", env=env, keyring_adapter=FakeKeyring()
    )
    selected = cfg.resolve(cli_model="3")
    validated = []
    output = []
    outcome = run_interactive_setup(
        cfg,
        creds,
        initial_selection=selected,
        input_fn=lambda _: "n",  # decline the old env key
        secret_input_fn=lambda _: "sk-or-v1-newvalidatedkey123456",
        output_fn=output.append,
        client_factory=lambda _provider, _url, key, *_: (
            validated.append(key) or MockProviderClient()
        ),
    )
    assert outcome.completed and outcome.config.model_id == selected.model_id
    assert validated == ["sk-or-v1-newvalidatedkey123456"]
    assert creds.resolve_api_key("openrouter").api_key == validated[0]
    assert "Update or unset the old environment key" in "\n".join(output)
    assert cfg.resolve().model_id.endswith(
        "lightning:free"
    )  # env applies again on a future launch


def test_one_shot_setup_preserves_explicit_model_and_local_limits(tmp_path):
    cfg = ConfigManager(config_dir=tmp_path / "cfg", env={"NEMOTRON_AGENT_MODEL": "2"})
    creds = CredentialManager(
        config_dir=tmp_path / "cfg", env={}, keyring_adapter=FakeKeyring()
    )
    selected = cfg.resolve(cli_model="3", cli_max_turns=2, cli_max_tool_calls=3)
    outcome = ensure_ready_or_run_setup(
        cfg,
        creds,
        resolved=selected,
        input_fn=lambda _: pytest.fail("selected model should not be asked again"),
        secret_input_fn=lambda _: "sk-or-v1-oneshot123456789",
        output_fn=lambda _: None,
        client_factory=lambda *_: MockProviderClient(),
    )
    assert outcome.completed
    assert outcome.config.model_id == selected.model_id
    assert (outcome.config.max_turns, outcome.config.max_tool_calls) == (2, 3)


def test_auth_set_key_uses_new_key_in_current_session(tmp_path):
    env = {"OPENROUTER_API_KEY": "sk-or-v1-oldauth123456789"}
    cfg = ConfigManager(config_dir=tmp_path / "cfg", env=env)
    creds = CredentialManager(
        config_dir=tmp_path / "cfg", env=env, keyring_adapter=FakeKeyring()
    )
    args = build_subcommand_parser().parse_args(
        ["auth", "set-key", "--provider", "openrouter", "--skip-check"]
    )
    output = []
    rc = handle_config_or_auth_command(
        args,
        config_manager=cfg,
        credential_manager=creds,
        secret_input_fn=lambda _: "sk-or-v1-newauth123456789",
        output_fn=output.append,
    )
    assert rc == 0
    assert creds.resolve_api_key("openrouter").api_key == "sk-or-v1-newauth123456789"
    assert "https://openrouter.ai/settings/keys" in "\n".join(output)
    assert "Update or unset the old environment key" in "\n".join(output)


def test_setup_retry_after_preselected_failure_shows_presets_and_honors_new_choice(
    tmp_path,
):
    from nemotron_agent.providers.base import ConnectionTestResult

    cfg = ConfigManager(config_dir=tmp_path / "cfg", env={"NEMOTRON_AGENT_MODEL": "3"})
    creds = CredentialManager(
        config_dir=tmp_path / "cfg", env={}, keyring_adapter=FakeKeyring()
    )
    answers = iter(["y", "2", ""])
    attempts = []
    output = []

    def factory(*_):
        client = MockProviderClient()
        attempts.append(client)
        if len(attempts) == 1:
            client.validate_connection = lambda model: ConnectionTestResult(
                ok=False,
                provider="openrouter",
                model_id=model,
                latency_ms=1,
                message="Try another model",
            )
        return client

    outcome = run_interactive_setup(
        cfg,
        creds,
        initial_selection=cfg.resolve(),
        input_fn=lambda _: next(answers),
        secret_input_fn=lambda _: "sk-or-v1-retrynewmodel123456",
        output_fn=output.append,
        client_factory=factory,
    )
    assert outcome.completed and outcome.config.model_id.endswith("super-vl:free")
    assert "Available NVIDIA Nemotron presets" in "\n".join(output)


def test_auth_refuses_unapproved_insecure_endpoint_before_key_input(tmp_path):
    env = {"NEMOTRON_AGENT_BASE_URL": "http://openrouter.ai/api/v1"}
    cfg = ConfigManager(config_dir=tmp_path / "cfg", env=env)
    creds = CredentialManager(
        config_dir=tmp_path / "cfg", env=env, keyring_adapter=FakeKeyring()
    )
    args = build_subcommand_parser().parse_args(
        ["auth", "set-key", "--provider", "openrouter"]
    )
    output = []
    rc = handle_config_or_auth_command(
        args,
        config_manager=cfg,
        credential_manager=creds,
        input_fn=lambda _: "n",
        secret_input_fn=lambda _: pytest.fail("do not request key before approval"),
        output_fn=output.append,
        client_factory=lambda *_: pytest.fail("do not connect"),
    )
    assert rc == 1
    assert "not encrypted" in "\n".join(output)
    assert creds.resolve_api_key("openrouter") is None
