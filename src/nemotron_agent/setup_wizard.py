"""Interactive first-run setup wizard and provider/credential configuration helper."""

from __future__ import annotations

import getpass
import sys
import warnings
from dataclasses import dataclass
from typing import Callable
from urllib.parse import parse_qsl, urlsplit

from nemotron_agent.config import (
    DEFAULT_PROVIDERS,
    PROVIDER_TERMS_NOTICE,
    ConfigManager,
    ModelPreset,
    ResolvedConfig,
    UserConfig,
    get_provider_profile,
)
from nemotron_agent.credentials import CredentialManager
from nemotron_agent.logging_utils import (
    AgentError,
    FallbackConsentRequiredError,
    redact_text,
    register_secret,
    safe_terminal_text,
)
from nemotron_agent.providers.base import ProviderClient
from nemotron_agent.providers.mock import MockProviderClient
from nemotron_agent.providers.openrouter import OpenAICompatibleProviderClient

InputFn = Callable[[str], str]
SecretInputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
ClientFactory = Callable[[str, str, str, float, int], ProviderClient]


def secure_secret_input(prompt: str) -> str:
    """Refuse getpass's unmasked stdin fallback instead of echoing an API key."""
    if not sys.stdin.isatty():
        raise AgentError(
            "Hidden key entry requires an interactive terminal. Set the documented API key "
            "environment variable or run /setup in a terminal."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            return getpass.getpass(prompt)
    except getpass.GetPassWarning as exc:
        raise AgentError(
            "This terminal cannot hide key input. Use an environment variable instead."
        ) from exc


def approve_key_endpoint(
    provider_name: str, base_url: str, *, input_fn: InputFn, output_fn: OutputFn
) -> str | None:
    """Show where a credential will go and require consent for risky destinations."""
    try:
        endpoint = urlsplit(base_url)
        sensitive_query = any(
            key.lower()
            in {
                "api_key",
                "apikey",
                "key",
                "token",
                "access_token",
                "password",
                "secret",
            }
            for key, _ in parse_qsl(endpoint.query, keep_blank_values=True)
        )
        valid = (
            endpoint.scheme in {"https", "http"}
            and endpoint.hostname
            and not endpoint.username
            and not endpoint.password
            and not sensitive_query
        )
    except ValueError:
        valid = False
    if not valid:
        raise AgentError(
            "Invalid API endpoint. Use an HTTP(S) URL without credentials in the URL."
        )
    output_fn(f"API endpoint that receives your key: {redact_text(base_url)}")
    if endpoint.scheme == "http":
        output_fn(
            "WARNING: HTTP is not encrypted; your API key and prompts could be intercepted."
        )
        if input_fn(
            "Continue with this insecure endpoint? [y/N]: "
        ).strip().lower() not in {"y", "yes"}:
            return None
    if (
        provider_name == "openrouter"
        and (endpoint.hostname or "").lower() != "openrouter.ai"
    ):
        output_fn(
            "WARNING: This is not the official OpenRouter host. The OpenRouter key will be sent to this endpoint."
        )
        if input_fn(
            "Send the OpenRouter key to this custom host? [y/N]: "
        ).strip().lower() not in {"y", "yes"}:
            return None
    return endpoint.scheme


def default_client_factory(
    provider: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
) -> ProviderClient:
    if provider == "mock":
        return MockProviderClient(provider="mock")
    return OpenAICompatibleProviderClient(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


@dataclass(frozen=True)
class SetupOutcome:
    completed: bool
    config: ResolvedConfig | None
    message: str


def format_presets_table(presets: list[ModelPreset]) -> str:
    """Format the selectable Nemotron model presets for display."""
    lines = ["Selectable NVIDIA Nemotron model presets (initial provider: OpenRouter):"]
    for preset in presets:
        tool_mode = (
            "native tool-calling"
            if preset.capabilities.supports_tool_calling
            else "structured-prompt fallback"
        )
        vision_tag = ", vision-language" if preset.capabilities.supports_vision else ""
        lines.append(f"  {preset.index}. {preset.display_name}")
        lines.append(f"     Model ID : {preset.model_id}")
        lines.append(f"     Provider : {preset.provider} ({tool_mode}{vision_tag})")
        lines.append(f"     Summary  : {preset.description}")
    lines.append("")
    lines.append(PROVIDER_TERMS_NOTICE)
    return "\n".join(lines)


def ensure_ready_or_run_setup(
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    *,
    resolved: ResolvedConfig,
    interactive: bool = True,
    force_setup: bool = False,
    input_fn: InputFn = input,
    secret_input_fn: SecretInputFn = secure_secret_input,
    output_fn: OutputFn = print,
    client_factory: ClientFactory = default_client_factory,
) -> SetupOutcome:
    """Reuse saved/env configuration if complete, or launch first-run setup when needed."""
    if resolved.mock_mode and not force_setup:
        return SetupOutcome(completed=True, config=resolved, message="Mock mode ready.")

    existing_cred = credential_manager.resolve_api_key(resolved.provider)
    has_env_model_or_saved = (
        resolved.has_saved_config
        or resolved.sources.get("model_id", "").startswith("env:")
        or resolved.sources.get("model_id") == "cli"
    )

    if not force_setup and has_env_model_or_saved and existing_cred is not None:
        if existing_cred.warning:
            output_fn(f"Warning: {existing_cred.warning}")
        return SetupOutcome(
            completed=True,
            config=resolved,
            message=f"Using configured provider '{resolved.provider}' and model '{resolved.model_id}'.",
        )

    if not interactive:
        missing_parts: list[str] = []
        if not has_env_model_or_saved:
            missing_parts.append(
                f"saved user config ({config_manager.config_path}) or NEMOTRON_AGENT_MODEL"
            )
        if existing_cred is None:
            profile = get_provider_profile(resolved.provider)
            env_hint = (
                " / ".join(profile.env_key_names)
                if profile.env_key_names
                else "NEMOTRON_AGENT_API_KEY"
            )
            missing_parts.append(f"API key ({env_hint} or saved credentials)")
        return SetupOutcome(
            completed=False,
            config=None,
            message=(
                "Configuration incomplete for non-interactive run; missing "
                + " and ".join(missing_parts)
                + ". Run `nemotron-agent setup` interactively, pass `--mock`, or set the documented environment variables."
            ),
        )

    outcome = run_interactive_setup(
        config_manager=config_manager,
        credential_manager=credential_manager,
        input_fn=input_fn,
        secret_input_fn=secret_input_fn,
        output_fn=output_fn,
        client_factory=client_factory,
        initial_selection=resolved
        if has_env_model_or_saved and not force_setup
        else None,
    )
    if not outcome.completed or outcome.config is None:
        return outcome
    chosen = outcome.config
    # Preserve CLI/env limits while applying the model explicitly chosen in setup.
    effective = resolved.model_copy(
        update={
            "provider": chosen.provider,
            "model_id": chosen.model_id,
            "base_url": chosen.base_url,
            "mock_mode": chosen.mock_mode,
            "capabilities": chosen.capabilities,
            "has_saved_config": True,
            "sources": {
                **resolved.sources,
                "provider": "setup",
                "model_id": "setup",
                "base_url": "setup",
            },
        }
    )
    return SetupOutcome(True, effective, outcome.message)


def run_interactive_setup(
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    *,
    input_fn: InputFn = input,
    secret_input_fn: SecretInputFn = secure_secret_input,
    output_fn: OutputFn = print,
    client_factory: ClientFactory = default_client_factory,
    initial_selection: ResolvedConfig | None = None,
) -> SetupOutcome:
    """Run setup, optionally using a model the user already selected in the REPL."""
    sink = output_fn

    def safe_output(message: str) -> None:
        sink(safe_terminal_text(redact_text(str(message))))

    output_fn = safe_output
    user_cfg = config_manager.load_user_config() or UserConfig()
    presets = user_cfg.get_presets()

    output_fn("=== Nemotron Agent First-Run Setup: Connect Your Model ===")
    table_displayed = initial_selection is None
    if table_displayed:
        output_fn("")
        output_fn(format_presets_table(presets))
        output_fn("")

    try:
        while True:
            # A /models selection is already explicit; do not ask the user to pick it twice.
            if initial_selection is not None:
                provider_name = initial_selection.provider
                model_id = initial_selection.model_id
                base_url = initial_selection.base_url
                initial_selection = None
                output_fn(f"Selected model: {model_id} (provider: {provider_name})")
            else:
                if not table_displayed:
                    output_fn("")
                    output_fn(
                        "Available NVIDIA Nemotron presets on OpenRouter (:free):"
                    )
                    output_fn(format_presets_table(presets))
                    table_displayed = True
                _, provider_name, model_id, base_url = _prompt_model_and_provider(
                    user_cfg=user_cfg,
                    presets=presets,
                    input_fn=input_fn,
                    output_fn=output_fn,
                )

            profile = get_provider_profile(provider_name)
            endpoint_scheme = ""
            if profile.requires_api_key:
                try:
                    endpoint_scheme = approve_key_endpoint(
                        provider_name, base_url, input_fn=input_fn, output_fn=output_fn
                    )
                except AgentError as exc:
                    output_fn(str(exc))
                    if not _prompt_retry_or_exit(input_fn, output_fn):
                        return SetupOutcome(
                            False, None, "Setup cancelled: invalid API endpoint."
                        )
                    continue
                if endpoint_scheme is None:
                    return SetupOutcome(
                        False, None, "Setup cancelled: API endpoint was not approved."
                    )

            # 2. Resolve or prompt for API key
            api_key = ""
            key_came_from_prompt = False
            if profile.requires_api_key:
                existing = credential_manager.resolve_api_key(provider_name)
                use_existing = False
                if existing is not None:
                    reuse_ans = (
                        input_fn(
                            f"An API key for '{provider_name}' is already available via {existing.source}. "
                            "Reuse it? [Y/n]: "
                        )
                        .strip()
                        .lower()
                    )
                    use_existing = reuse_ans in ("", "y", "yes")
                    if use_existing:
                        api_key = existing.api_key

                if not use_existing:
                    output_fn("")
                    output_fn(f"Connect to {profile.display_name}")
                    if profile.key_management_url:
                        output_fn(
                            f"  1. Open this page to create a key: {profile.key_management_url}"
                        )
                    else:
                        output_fn(
                            "  1. Open your provider's account dashboard and create an API key."
                        )
                    output_fn(
                        "  2. Copy the key, then paste it at the hidden prompt below and press Enter."
                    )
                    output_fn(
                        "     Nothing will appear while you paste or type; this is normal."
                    )
                    output_fn(f"  {profile.key_obtain_instructions}")
                    if endpoint_scheme == "http":
                        output_fn(
                            "  Privacy: the key will be sent to the HTTP endpoint shown above without transport encryption."
                        )
                    else:
                        output_fn(f"  Privacy: {profile.key_usage_explanation}")
                    output_fn("")
                    raw_key = secret_input_fn(
                        f"Paste {profile.display_name} API key (hidden), then press Enter: "
                    ).strip()
                    if not raw_key:
                        output_fn("Error: API key was empty.")
                        if not _prompt_retry_or_exit(input_fn, output_fn):
                            return SetupOutcome(
                                completed=False,
                                config=None,
                                message="Setup cancelled: no API key entered.",
                            )
                        continue
                    register_secret(raw_key)
                    api_key = raw_key
                    key_came_from_prompt = True

            # 3. Validate connection WITHOUT sending repository files
            output_fn(
                f"Validating connection to '{provider_name}' ({base_url}) with model '{model_id}' "
                "(no repository files are sent)..."
            )
            client = client_factory(
                provider_name,
                base_url,
                api_key,
                user_cfg.request_timeout_seconds,
                user_cfg.max_retries,
            )
            try:
                test_res = client.validate_connection(model_id)
                if not test_res.ok:
                    raise AgentError(test_res.message)
            except AgentError as exc:
                output_fn(f"Connection test failed: {redact_text(str(exc))}")
                output_fn(
                    "Note: nemotron-agent never switches models or providers automatically."
                )
                if not _prompt_retry_or_exit(input_fn, output_fn):
                    return SetupOutcome(
                        completed=False,
                        config=None,
                        message="Setup aborted after connection validation failure.",
                    )
                continue
            finally:
                client.close()

            output_fn(
                f"Validation succeeded ({test_res.latency_ms} ms): {test_res.message}"
            )

            # 4. Save non-secret user configuration
            user_cfg.provider = provider_name
            user_cfg.model_id = model_id
            user_cfg.base_url = base_url
            saved_cfg_path = config_manager.save_user_config(user_cfg)
            output_fn(f"Saved non-secret settings to {saved_cfg_path}")

            # 5. Store API key securely (OS keyring -> consented 0600 fallback -> session-only)
            if profile.requires_api_key and key_came_from_prompt:
                _persist_or_hold_key(
                    credential_manager=credential_manager,
                    provider_name=provider_name,
                    api_key=api_key,
                    input_fn=input_fn,
                    output_fn=output_fn,
                )
                # The key just validated must win over older environment values
                # for this process, even when it was also stored in a keyring.
                credential_manager.set_session_key(provider_name, api_key)
                if existing and existing.source.startswith("env:"):
                    output_fn(
                        "The new key is active for this session. Update or unset the old environment key for future launches."
                    )

            resolved = config_manager.resolve()
            resolved = resolved.model_copy(
                update={
                    "provider": provider_name,
                    "model_id": model_id,
                    "base_url": base_url,
                    "mock_mode": provider_name == "mock",
                    "capabilities": user_cfg.capabilities_for(model_id),
                    "sources": {
                        **resolved.sources,
                        "provider": "setup",
                        "model_id": "setup",
                        "base_url": "setup",
                    },
                    "has_saved_config": True,
                }
            )
            return SetupOutcome(
                completed=True,
                config=resolved,
                message=f"Setup complete. Active provider='{resolved.provider}', model='{resolved.model_id}'.",
            )

    except (KeyboardInterrupt, EOFError):
        output_fn("\nSetup cancelled by user.")
        return SetupOutcome(
            completed=False,
            config=None,
            message="Setup cancelled by user.",
        )


def _prompt_model_and_provider(
    *,
    user_cfg: UserConfig,
    presets: list[ModelPreset],
    input_fn: InputFn,
    output_fn: OutputFn,
) -> tuple[ModelPreset | None, str, str, str]:
    while True:
        choice = input_fn(
            f"Choose a model preset [1-{len(presets)}] (default 1), or type 'custom' for another provider/model: "
        ).strip()
        if not choice:
            choice = "1"

        if choice.lower() == "custom":
            prov = (
                input_fn(
                    "Enter provider name [openrouter / openai_compatible] (default: openrouter): "
                )
                .strip()
                .lower()
                or "openrouter"
            )
            profile = get_provider_profile(prov)
            if prov != "openrouter":
                output_fn(
                    "Important: OpenRouter `:free` model IDs are provider-specific. Verify the "
                    "exact model ID supported by your chosen provider."
                )
            default_url = profile.default_base_url
            url_prompt = (
                f"Enter API base URL [{default_url}]: "
                if default_url
                else "Enter API base URL (e.g., https://api.example.com/v1): "
            )
            base_url = input_fn(url_prompt).strip().rstrip("/") or default_url
            if not base_url:
                output_fn("Error: A base URL is required for this provider.")
                continue
            model_id = input_fn("Enter exact model ID: ").strip()
            if not model_id:
                output_fn("Error: Model ID cannot be empty.")
                continue
            return None, prov, model_id, base_url

        matched = user_cfg.find_preset(choice)
        if matched is None:
            output_fn(
                f"Invalid selection '{choice}'. Enter a number 1-{len(presets)}, preset name, or 'custom'."
            )
            continue

        prov = matched.provider
        default_url = DEFAULT_PROVIDERS.get(
            prov, get_provider_profile(prov)
        ).default_base_url
        url_override = (
            input_fn(f"API base URL for {prov} [press Enter for {default_url}]: ")
            .strip()
            .rstrip("/")
        )
        base_url = url_override or default_url
        return matched, prov, matched.model_id, base_url


def _persist_or_hold_key(
    *,
    credential_manager: CredentialManager,
    provider_name: str,
    api_key: str,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> None:
    def _ask_fallback(explanation: str) -> bool:
        output_fn("")
        output_fn(explanation)
        ans = (
            input_fn(
                "Allow saving API key to this restricted (0600) local credentials file? [y/N]: "
            )
            .strip()
            .lower()
        )
        return ans in ("y", "yes")

    try:
        saved = credential_manager.save_api_key(
            provider_name,
            api_key,
            confirm_fallback=_ask_fallback,
        )
        output_fn(f"Stored API key in {saved.detail}.")
    except FallbackConsentRequiredError:
        credential_manager.set_session_key(provider_name, api_key)
        output_fn(
            "API key was NOT written to disk. Kept in memory for this session only. "
            "For future sessions, export OPENROUTER_API_KEY or NEMOTRON_AGENT_API_KEY."
        )


def _prompt_retry_or_exit(input_fn: InputFn, output_fn: OutputFn) -> bool:
    ans = (
        input_fn("Would you like to retry setup with updated settings? [y/N]: ")
        .strip()
        .lower()
    )
    if ans in ("y", "yes"):
        output_fn("")
        return True
    return False
