"""Configuration, model presets, provider metadata, and precedence resolution."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from nemotron_agent.logging_utils import ConfigError

APP_DIR_NAME = "nemotron-agent"
CONFIG_FILE_NAME = "config.json"

# Environment variable names (documented in README and CLI help)
ENV_CONFIG_DIR = "NEMOTRON_AGENT_CONFIG_DIR"
ENV_PROVIDER = "NEMOTRON_AGENT_PROVIDER"
ENV_MODEL = "NEMOTRON_AGENT_MODEL"
ENV_BASE_URL = "NEMOTRON_AGENT_BASE_URL"
ENV_API_KEY = "NEMOTRON_AGENT_API_KEY"
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
ENV_MOCK = "NEMOTRON_AGENT_MOCK"
ENV_MAX_TURNS = "NEMOTRON_AGENT_MAX_TURNS"
ENV_MAX_TOOL_CALLS = "NEMOTRON_AGENT_MAX_TOOL_CALLS"
ENV_TIMEOUT = "NEMOTRON_AGENT_TIMEOUT"

PROVIDER_TERMS_NOTICE = (
    "Note: The `:free` model IDs above are OpenRouter-specific identifiers and do not "
    "imply free direct endpoints from NVIDIA. Free-tier availability, rate limits, "
    "supported parameters (such as native tool calling), model IDs, and data/privacy "
    "policies may change at any time. Check your selected provider's current model "
    "documentation and terms before sending code."
)


class ModelCapabilities(BaseModel):
    """Per-model capability flags and context limits (configurable per model)."""

    model_config = ConfigDict(extra="forbid")

    supports_tool_calling: bool = True
    supports_structured_output: bool = False
    supports_vision: bool = False
    supports_temperature: bool = True
    context_window_tokens: int = Field(default=32768, ge=1024)
    max_context_chars: int = Field(default=48000, ge=2000)
    default_max_tokens: int = Field(default=2048, ge=64, le=32768)


class ModelPreset(BaseModel):
    """Selectable model preset stored in configuration."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1)
    slug: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    provider: str = Field(default="openrouter", min_length=1)
    description: str = Field(min_length=1)
    is_free_endpoint: bool = True
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)


class ProviderProfile(BaseModel):
    """Metadata and setup guidance for a supported API provider."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    default_base_url: str
    env_key_names: list[str]
    key_usage_explanation: str
    key_obtain_instructions: str
    key_management_url: str | None = None
    requires_api_key: bool = True


DEFAULT_PROVIDERS: dict[str, ProviderProfile] = {
    "openrouter": ProviderProfile(
        name="openrouter",
        display_name="OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        env_key_names=[ENV_API_KEY, ENV_OPENROUTER_API_KEY],
        key_usage_explanation=(
            "Your API key is sent in the HTTP Authorization header "
            "over HTTPS to the configured OpenRouter API base URL when validating your "
            "connection or sending prompts. It is never echoed or written to logs, "
            "prompts, conversation history, or repository files."
        ),
        key_obtain_instructions=(
            "Sign in to OpenRouter, create an API key, and copy it. The same OpenRouter "
            "key can be used with its supported models. Review the provider's terms, "
            "rate limits, and data-handling settings before use."
        ),
        key_management_url="https://openrouter.ai/settings/keys",
        requires_api_key=True,
    ),
    "openai_compatible": ProviderProfile(
        name="openai_compatible",
        display_name="OpenAI-Compatible Custom Endpoint",
        default_base_url="",
        env_key_names=[ENV_API_KEY],
        key_usage_explanation=(
            "Your API key is sent in the HTTP Authorization header "
            "to your configured OpenAI-compatible base URL. Note that OpenRouter `:free` "
            "model IDs are provider-specific and will not work on other providers unless "
            "independently supported."
        ),
        key_obtain_instructions=(
            "Obtain an API key and base URL from your chosen OpenAI-compatible API "
            "provider's account dashboard and verify the exact model IDs it supports."
        ),
        requires_api_key=True,
    ),
    "mock": ProviderProfile(
        name="mock",
        display_name="Local Mock Provider (Offline)",
        default_base_url="mock://local",
        env_key_names=[],
        key_usage_explanation="Mock mode runs completely offline and does not use an API key.",
        key_obtain_instructions="No API key is required for mock mode.",
        requires_api_key=False,
    ),
}

DEFAULT_MODEL_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        index=1,
        slug="nemotron-3.5-lightning",
        display_name="Nemotron 3.5 Lightning",
        model_id="nvidia/nemotron-3.5-lightning:free",
        provider="openrouter",
        description=(
            "Low-latency Nemotron 3.5 MoE preset (3B active / 30B total) via OpenRouter; "
            "native tool-calling default for fast interactive coding questions and edits."
        ),
        is_free_endpoint=True,
        capabilities=ModelCapabilities(
            supports_tool_calling=True,
            supports_structured_output=False,
            supports_vision=False,
            supports_temperature=True,
            context_window_tokens=1_000_000,
            max_context_chars=48000,
            default_max_tokens=2048,
        ),
    ),
    ModelPreset(
        index=2,
        slug="nemotron-3.5-super-vl",
        display_name="Nemotron 3.5 Super VL",
        model_id="nvidia/nemotron-3.5-super-vl:free",
        provider="openrouter",
        description=(
            "Vision-language Nemotron 3.5 preset via OpenRouter; defaults to validated "
            "structured-prompt tool parsing because VL endpoints may not expose native tool calling."
        ),
        is_free_endpoint=True,
        capabilities=ModelCapabilities(
            supports_tool_calling=False,
            supports_structured_output=False,
            supports_vision=True,
            supports_temperature=True,
            context_window_tokens=128_000,
            max_context_chars=32000,
            default_max_tokens=2048,
        ),
    ),
    ModelPreset(
        index=3,
        slug="nemotron-3-ultra",
        display_name="Nemotron 3 Ultra",
        model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        provider="openrouter",
        description=(
            "Large MoE Nemotron 3 model (550B total / 55B active) via OpenRouter for "
            "deeper multi-file reasoning and complex refactors."
        ),
        is_free_endpoint=True,
        capabilities=ModelCapabilities(
            supports_tool_calling=True,
            supports_structured_output=False,
            supports_vision=False,
            supports_temperature=True,
            context_window_tokens=1_000_000,
            max_context_chars=56000,
            default_max_tokens=2048,
        ),
    ),
    ModelPreset(
        index=4,
        slug="nemotron-3-nano-omni-reasoning",
        display_name="Nemotron 3 Nano Omni Reasoning",
        model_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        provider="openrouter",
        description=(
            "Compact multimodal Nemotron 3 Nano (30B total / 3B active) via OpenRouter; "
            "free, reasoning-oriented, text-only in this preset."
        ),
        is_free_endpoint=True,
        capabilities=ModelCapabilities(
            supports_tool_calling=False,
            supports_structured_output=False,
            supports_vision=False,
            supports_temperature=True,
            context_window_tokens=262_000,
            max_context_chars=40000,
            default_max_tokens=2048,
        ),
    ),
    ModelPreset(
        index=5,
        slug="llama-3.1-nemotron-70b",
        display_name="Llama 3.1 Nemotron 70B",
        model_id="nvidia/llama-3.1-nemotron-70b-instruct:free",
        provider="openrouter",
        description=(
            "Llama 3.1 Nemotron 70B instruct via OpenRouter; strong general-purpose "
            "coding assistant with native tool-calling."
        ),
        is_free_endpoint=True,
        capabilities=ModelCapabilities(
            supports_tool_calling=True,
            supports_structured_output=False,
            supports_vision=False,
            supports_temperature=True,
            context_window_tokens=128_000,
            max_context_chars=48000,
            default_max_tokens=2048,
        ),
    ),
    ModelPreset(
        index=6,
        slug="llama-3.1-nemotron-nano-8b",
        display_name="Llama 3.1 Nemotron Nano 8B",
        model_id="nvidia/llama-3.1-nemotron-nano-8b-v1:free",
        provider="openrouter",
        description=(
            "Small, fast Llama 3.1 Nemotron Nano 8B via OpenRouter for lightweight "
            "repository Q&A and quick edits."
        ),
        is_free_endpoint=True,
        capabilities=ModelCapabilities(
            supports_tool_calling=True,
            supports_structured_output=False,
            supports_vision=False,
            supports_temperature=True,
            context_window_tokens=128_000,
            max_context_chars=36000,
            default_max_tokens=2048,
        ),
    ),
)

_FORBIDDEN_SECRET_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "openrouter_api_key",
        "nemotron_agent_api_key",
        "secret",
        "secret_key",
        "access_token",
        "password",
    }
)


class UserConfig(BaseModel):
    """Non-secret user-level configuration persisted in the user's config directory."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "openrouter"
    model_id: str = DEFAULT_MODEL_PRESETS[0].model_id
    base_url: str = DEFAULT_PROVIDERS["openrouter"].default_base_url
    max_turns: int = Field(default=8, ge=1, le=50)
    max_tool_calls: int = Field(default=12, ge=1, le=100)
    request_timeout_seconds: float = Field(default=30.0, gt=1.0, le=300.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    command_timeout_seconds: float = Field(default=60.0, gt=1.0, le=600.0)
    max_output_bytes: int = Field(default=32768, ge=1024, le=524288)
    excluded_patterns: list[str] = Field(default_factory=list)
    extra_allowed_commands: list[list[str]] = Field(default_factory=list)
    presets: list[ModelPreset] = Field(default_factory=lambda: [p.model_copy(deep=True) for p in DEFAULT_MODEL_PRESETS])
    capability_overrides: dict[str, ModelCapabilities] = Field(default_factory=dict)
    prompt_color: str = "orange"

    @field_validator("prompt_color")
    @classmethod
    def _validate_prompt_color(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"orange", "blue", "neon_green"}:
            raise ValueError("prompt_color must be one of: orange, blue, neon_green")
        return cleaned

    @model_validator(mode="before")
    @classmethod
    def _reject_secret_fields(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            bad = [k for k in data if str(k).strip().lower() in _FORBIDDEN_SECRET_CONFIG_KEYS]
            if bad:
                raise ValueError(
                    f"Refusing to load or store secret fields ({', '.join(bad)}) in non-secret config.json. "
                    "Use the credential store (`auth set-key`) or environment variables instead."
                )
        return data

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("provider cannot be empty.")
        return cleaned

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model_id cannot be empty.")
        return cleaned

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if cleaned and not (
            cleaned.startswith("https://")
            or cleaned.startswith("http://")
            or cleaned.startswith("mock://")
        ):
            raise ValueError("base_url must start with https://, http://, or mock://")
        return cleaned

    def get_presets(self) -> list[ModelPreset]:
        """Return configured model presets (falling back to built-in defaults if empty)."""
        return list(self.presets) if self.presets else [p.model_copy(deep=True) for p in DEFAULT_MODEL_PRESETS]

    def find_preset(self, selector: str) -> ModelPreset | None:
        """Match a preset by 1-based index, slug, display name, or exact model_id."""
        needle = selector.strip()
        needle_lower = needle.lower()
        for preset in self.get_presets():
            if (
                str(preset.index) == needle
                or preset.slug.lower() == needle_lower
                or preset.display_name.lower() == needle_lower
                or preset.model_id == needle
            ):
                return preset
        return None

    def capabilities_for(self, model_id: str) -> ModelCapabilities:
        """Resolve capability flags for a model ID from overrides, presets, or safe defaults."""
        if model_id in self.capability_overrides:
            return self.capability_overrides[model_id].model_copy(deep=True)
        for preset in self.get_presets():
            if preset.model_id == model_id:
                return preset.capabilities.model_copy(deep=True)
        # Conservative default for custom/unknown models
        return ModelCapabilities(
            supports_tool_calling=False,
            supports_structured_output=False,
            supports_vision=False,
            supports_temperature=True,
        )


class ResolvedConfig(BaseModel):
    """Effective runtime configuration with provenance tracking."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str
    base_url: str
    mock_mode: bool = False
    max_turns: int
    max_tool_calls: int
    request_timeout_seconds: float
    max_retries: int
    command_timeout_seconds: float
    max_output_bytes: int
    excluded_patterns: list[str]
    extra_allowed_commands: list[list[str]]
    capabilities: ModelCapabilities
    sources: dict[str, str] = Field(default_factory=dict)
    config_file_path: str = ""
    has_saved_config: bool = False

    @property
    def is_free_endpoint(self) -> bool:
        return ":free" in self.model_id.lower()


def _default_os_config_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the standard OS user-level config directory (ignoring NEMOTRON_AGENT_CONFIG_DIR)."""
    environ = env if env is not None else os.environ
    if sys.platform == "win32":
        appdata = environ.get("APPDATA", "").strip()
        if appdata:
            return (Path(appdata) / APP_DIR_NAME).resolve()
        return (Path.home() / "AppData" / "Roaming" / APP_DIR_NAME).resolve()

    xdg = environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return (Path(xdg).expanduser() / APP_DIR_NAME).resolve()
    return (Path.home() / ".config" / APP_DIR_NAME).resolve()


def get_user_config_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve user-level config directory outside any target codebase."""
    environ = env if env is not None else os.environ
    override = environ.get(ENV_CONFIG_DIR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_os_config_dir(environ)


def _parse_bool_env(raw: str | None) -> bool | None:
    if raw is None or not raw.strip():
        return None
    norm = raw.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Invalid boolean environment value: {raw!r}. Expected 1/true or 0/false.")


def _ensure_not_inside_repo(config_dir: Path, project_root: Path | None) -> None:
    """Ensure user-level configuration is never stored inside a target project directory."""
    if project_root is None:
        return
    resolved_dir = config_dir.resolve()
    resolved_root = project_root.resolve()
    home_dir = Path.home().resolve()

    # If the user launched from their OS home directory (~) and config_dir is the standard
    # OS user-level config path (~/.config/nemotron-agent or %APPDATA%\nemotron-agent), allow it.
    if resolved_root == home_dir and resolved_dir == _default_os_config_dir():
        return

    if resolved_dir == resolved_root or resolved_root in resolved_dir.parents:
        raise ConfigError(
            f"Refusing to store user configuration or credentials inside project directory "
            f"({resolved_root}). Configure {ENV_CONFIG_DIR} to a directory outside the project."
        )


class ConfigManager:
    """Loads, validates, saves, and resolves user-level non-secret configuration."""

    def __init__(
        self,
        config_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._env = env if env is not None else os.environ
        self.config_dir = (config_dir.expanduser().resolve() if config_dir else get_user_config_dir(self._env))
        self.config_path = self.config_dir / CONFIG_FILE_NAME
        self._project_root = project_root.resolve() if project_root else None

    def set_project_root(self, project_root: Path) -> None:
        """Update the active repository used to guard user-level config writes."""
        root = project_root.expanduser().resolve()
        _ensure_not_inside_repo(self.config_dir, root)
        self._project_root = root

    def exists(self) -> bool:
        return self.config_path.is_file()

    def load_user_config(self) -> UserConfig | None:
        """Load saved user config if present; return None if not yet created."""
        if not self.config_path.exists():
            return None
        try:
            raw_text = self.config_path.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
            return UserConfig.model_validate(raw_data)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"Saved configuration file at {self.config_path} contains invalid JSON: {exc.msg}"
            ) from exc
        except ValidationError as exc:
            raise ConfigError(
                f"Saved configuration at {self.config_path} failed validation: {exc}"
            ) from exc
        except OSError as exc:
            raise ConfigError(
                f"Unable to read configuration file {self.config_path}: {exc}"
            ) from exc

    def save_user_config(self, config: UserConfig) -> Path:
        """Persist non-secret UserConfig to user-level config.json."""
        _ensure_not_inside_repo(self.config_dir, self._project_root)
        # Re-validate before writing
        validated = UserConfig.model_validate(config.model_dump())
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.config_dir, 0o700)
            except OSError:
                pass
            tmp_path = self.config_dir / f".{CONFIG_FILE_NAME}.tmp"
            payload = json.dumps(validated.model_dump(mode="json"), indent=2) + "\n"
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self.config_path)
            return self.config_path
        except OSError as exc:
            raise ConfigError(f"Failed to write configuration to {self.config_path}: {exc}") from exc

    def resolve(
        self,
        *,
        cli_provider: str | None = None,
        cli_model: str | None = None,
        cli_base_url: str | None = None,
        cli_mock: bool | None = None,
        cli_max_turns: int | None = None,
        cli_max_tool_calls: int | None = None,
        cli_timeout: float | None = None,
    ) -> ResolvedConfig:
        """Resolve effective configuration with precedence: CLI > Env > Saved Config > Default."""
        saved = self.load_user_config()
        has_saved = saved is not None
        base = saved if saved is not None else UserConfig()
        base_source = "config" if has_saved else "default"

        sources: dict[str, str] = {}

        # Mock mode resolution
        env_mock = _parse_bool_env(self._env.get(ENV_MOCK))
        if cli_mock is not None:
            mock_mode = cli_mock
            sources["mock_mode"] = "cli"
        elif env_mock is not None:
            mock_mode = env_mock
            sources["mock_mode"] = f"env:{ENV_MOCK}"
        else:
            mock_mode = base.provider == "mock"
            sources["mock_mode"] = base_source

        # Provider resolution
        env_provider = self._env.get(ENV_PROVIDER, "").strip()
        if cli_provider:
            provider = cli_provider.strip().lower()
            sources["provider"] = "cli"
        elif mock_mode and not env_provider:
            provider = "mock"
            sources["provider"] = sources["mock_mode"]
        elif env_provider:
            provider = env_provider.lower()
            sources["provider"] = f"env:{ENV_PROVIDER}"
        else:
            provider = base.provider
            sources["provider"] = base_source

        if provider == "mock":
            mock_mode = True

        # Model resolution (supports preset selector OR raw model ID)
        env_model = self._env.get(ENV_MODEL, "").strip()
        if cli_model:
            matched = base.find_preset(cli_model)
            model_id = matched.model_id if matched else cli_model.strip()
            sources["model_id"] = "cli"
        elif env_model:
            matched = base.find_preset(env_model)
            model_id = matched.model_id if matched else env_model
            sources["model_id"] = f"env:{ENV_MODEL}"
        else:
            model_id = base.model_id
            sources["model_id"] = base_source

        # Base URL resolution
        env_base_url = self._env.get(ENV_BASE_URL, "").strip()
        if cli_base_url:
            base_url = cli_base_url.strip().rstrip("/")
            sources["base_url"] = "cli"
        elif env_base_url:
            base_url = env_base_url.rstrip("/")
            sources["base_url"] = f"env:{ENV_BASE_URL}"
        elif has_saved and base.provider == provider and base.base_url:
            base_url = base.base_url
            sources["base_url"] = "config"
        elif provider in DEFAULT_PROVIDERS and DEFAULT_PROVIDERS[provider].default_base_url:
            base_url = DEFAULT_PROVIDERS[provider].default_base_url
            sources["base_url"] = "default"
        else:
            base_url = base.base_url
            sources["base_url"] = base_source

        # Bounded loop & timeout settings
        max_turns = base.max_turns
        sources["max_turns"] = base_source
        if self._env.get(ENV_MAX_TURNS, "").strip():
            try:
                max_turns = int(self._env[ENV_MAX_TURNS].strip())
                sources["max_turns"] = f"env:{ENV_MAX_TURNS}"
            except ValueError as exc:
                raise ConfigError(f"{ENV_MAX_TURNS} must be an integer.") from exc
        if cli_max_turns is not None:
            max_turns = cli_max_turns
            sources["max_turns"] = "cli"

        max_tool_calls = base.max_tool_calls
        sources["max_tool_calls"] = base_source
        if self._env.get(ENV_MAX_TOOL_CALLS, "").strip():
            try:
                max_tool_calls = int(self._env[ENV_MAX_TOOL_CALLS].strip())
                sources["max_tool_calls"] = f"env:{ENV_MAX_TOOL_CALLS}"
            except ValueError as exc:
                raise ConfigError(f"{ENV_MAX_TOOL_CALLS} must be an integer.") from exc
        if cli_max_tool_calls is not None:
            max_tool_calls = cli_max_tool_calls
            sources["max_tool_calls"] = "cli"

        timeout = base.request_timeout_seconds
        sources["request_timeout_seconds"] = base_source
        if self._env.get(ENV_TIMEOUT, "").strip():
            try:
                timeout = float(self._env[ENV_TIMEOUT].strip())
                sources["request_timeout_seconds"] = f"env:{ENV_TIMEOUT}"
            except ValueError as exc:
                raise ConfigError(f"{ENV_TIMEOUT} must be a number.") from exc
        if cli_timeout is not None:
            timeout = cli_timeout
            sources["request_timeout_seconds"] = "cli"

        capabilities = base.capabilities_for(model_id)

        try:
            return ResolvedConfig(
                provider=provider,
                model_id=model_id,
                base_url=base_url,
                mock_mode=mock_mode,
                max_turns=max_turns,
                max_tool_calls=max_tool_calls,
                request_timeout_seconds=timeout,
                max_retries=base.max_retries,
                command_timeout_seconds=base.command_timeout_seconds,
                max_output_bytes=base.max_output_bytes,
                excluded_patterns=list(base.excluded_patterns),
                extra_allowed_commands=[list(cmd) for cmd in base.extra_allowed_commands],
                capabilities=capabilities,
                sources=sources,
                config_file_path=str(self.config_path),
                has_saved_config=has_saved,
            )
        except ValidationError as exc:
            raise ConfigError(f"Resolved configuration is invalid: {exc}") from exc


def get_provider_profile(provider: str) -> ProviderProfile:
    """Return provider metadata, or a generic OpenAI-compatible profile for custom names."""
    norm = provider.strip().lower()
    if norm in DEFAULT_PROVIDERS:
        return DEFAULT_PROVIDERS[norm]
    return ProviderProfile(
        name=norm,
        display_name=f"Custom OpenAI-Compatible Provider ({norm})",
        default_base_url="",
        env_key_names=[ENV_API_KEY],
        key_usage_explanation=(
            f"Your API key is sent in the HTTP Authorization header "
            f"to the configured base URL for provider '{norm}'."
        ),
        key_obtain_instructions=(
            f"Obtain an API key and base URL from '{norm}' and verify which model IDs "
            f"and capabilities that provider supports."
        ),
        requires_api_key=True,
    )
