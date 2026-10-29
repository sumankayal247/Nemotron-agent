"""Command-line interface, one-shot runner, and interactive chat REPL for nemotron-agent."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from nemotron_agent import __version__
from nemotron_agent.agent import AgentSession
from nemotron_agent.update import InstallMethod, UpdateManager
from nemotron_agent.config import (
    DEFAULT_PROVIDERS,
    ConfigManager,
    ModelCapabilities,
    ResolvedConfig,
    UserConfig,
    get_provider_profile,
)
from nemotron_agent.credentials import CredentialManager
from nemotron_agent.logging_utils import (
    AgentError,
    FallbackConsentRequiredError,
    configure_logging,
    redact_text,
    register_secret,
)
from nemotron_agent.providers.base import ProviderClient
from nemotron_agent.setup_wizard import (
    ClientFactory,
    InputFn,
    OutputFn,
    SecretInputFn,
    approve_key_endpoint,
    default_client_factory,
    ensure_ready_or_run_setup,
    format_presets_table,
    run_interactive_setup,
    secure_secret_input,
)

KNOWN_SUBCOMMANDS = frozenset({"setup", "config", "auth", "ping", "chat", "ask", "update"})

_VALUE_FLAGS = frozenset(
    {
        "-p",
        "--path",
        "--dir",
        "--project",
        "--repo",
        "--provider",
        "--model",
        "--base-url",
        "--exclude",
        "--include",
        "--max-turns",
        "--max-tool-calls",
        "--timeout",
    }
)


def _normalize_cli_argv(raw_args: list[str]) -> list[str]:
    """Allow natural `nemotron-agent "prompt"` and bare `nemotron-agent` (REPL) invocations."""
    if any(flag in raw_args for flag in ("-h", "--help", "--version")):
        return raw_args

    i = 0
    n = len(raw_args)
    while i < n:
        tok = raw_args[i]
        if tok == "--":
            # Everything after `--` is positional prompt text
            return [*raw_args[:i], "ask", *raw_args[i:]]
        if tok in _VALUE_FLAGS:
            i += 2
            continue
        if any(tok.startswith(f"{vf}=") for vf in _VALUE_FLAGS):
            i += 1
            continue
        if tok.startswith("-"):
            i += 1
            continue
        # Found first positional token
        if tok in KNOWN_SUBCOMMANDS:
            return raw_args
        return [*raw_args[:i], "ask", *raw_args[i:]]

    # No positional token at all -> default to interactive `chat`
    return [*raw_args, "chat"]


def _add_global_flags(parser: argparse.ArgumentParser, *, is_subparser: bool = False) -> None:
    sup = argparse.SUPPRESS if is_subparser else None
    parser.add_argument(
        "-p",
        "--path",
        "--dir",
        "--project",
        "--repo",
        dest="repo",
        default=argparse.SUPPRESS if is_subparser else None,
        help="Local folder path of the codebase on your computer (default: current directory).",
    )
    parser.add_argument(
        "--select-dir",
        action="store_true",
        default=argparse.SUPPRESS if is_subparser else False,
        help="Prompt interactively to choose a local codebase directory on launch.",
    )
    parser.add_argument(
        "--provider",
        default=sup,
        help="Override provider for this invocation (e.g., openrouter, openai_compatible, mock).",
    )
    parser.add_argument(
        "--model",
        default=sup,
        help="Override model ID or preset name/number (1-4) for this invocation.",
    )
    parser.add_argument(
        "--base-url",
        default=sup,
        help="Override API base URL for this invocation.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=argparse.SUPPRESS if is_subparser else None,
        help="Run offline using the deterministic mock model client (no API key needed).",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        default=argparse.SUPPRESS if is_subparser else False,
        help="Enforce strict read-only mode (disable file edits and command execution).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=argparse.SUPPRESS if is_subparser else [],
        help="Additional file/directory glob pattern to exclude from repository context (repeatable).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=argparse.SUPPRESS if is_subparser else [],
        help="Explicit file glob pattern to re-include if excluded by non-secret rules (repeatable).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=sup,
        help="Maximum model turns per task.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=sup,
        help="Maximum tool calls per task.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=sup,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=argparse.SUPPRESS if is_subparser else False,
        help="Disable interactive prompts (for CI/scripts; denies writes and commands).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS if is_subparser else False,
        help="Enable informational diagnostic logging (with automatic secret redaction).",
    )


def build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemotron-agent",
        description=(
            "Bounded, repository-aware command-line coding agent for NVIDIA Nemotron models. "
            "Run `nemotron-agent` to start interactive chat, `nemotron-agent \"question\"` for a "
            "one-shot prompt, or `nemotron-agent setup` to configure your provider and model."
        ),
    )
    parser.add_argument("--version", action="version", version=f"nemotron-agent {__version__}")
    _add_global_flags(parser, is_subparser=False)

    subparsers = parser.add_subparsers(dest="subcommand")

    # chat (interactive REPL)
    sub_chat = subparsers.add_parser(
        "chat", help="Start an interactive repository-aware chat session."
    )
    _add_global_flags(sub_chat, is_subparser=True)
    sub_chat.add_argument(
        "--no-tui",
        action="store_true",
        default=False,
        help="Use the classic line-based REPL instead of the full-screen TUI.",
    )

    # ask (one-shot prompt)
    sub_ask = subparsers.add_parser(
        "ask", help="Run a single bounded repository-aware prompt and exit."
    )
    _add_global_flags(sub_ask, is_subparser=True)
    sub_ask.add_argument(
        "prompt",
        nargs="+",
        help="Prompt or question for the coding agent.",
    )

    # setup
    sub_setup = subparsers.add_parser(
        "setup", help="Run interactive model/provider and credential setup."
    )
    _add_global_flags(sub_setup, is_subparser=True)

    # ping
    sub_ping = subparsers.add_parser(
        "ping",
        help="Validate active provider connection and model without sending repository files.",
    )
    _add_global_flags(sub_ping, is_subparser=True)

    # config
    sub_cfg = subparsers.add_parser("config", help="Inspect or update non-secret configuration.")
    _add_global_flags(sub_cfg, is_subparser=True)
    cfg_sub = sub_cfg.add_subparsers(dest="config_action")

    cfg_show = cfg_sub.add_parser(
        "show", help="Show active provider, model, capabilities, and credential status."
    )
    _add_global_flags(cfg_show, is_subparser=True)

    cfg_sub.add_parser("presets", help="List selectable NVIDIA Nemotron model presets.")

    cfg_set_model = cfg_sub.add_parser(
        "set-model", help="Set the saved default model by preset (1-4 / slug / name) or model ID."
    )
    cfg_set_model.add_argument(
        "model",
        help="Preset number (1-4), preset slug/name, or exact provider model ID.",
    )
    cfg_set_model.add_argument(
        "--skip-check",
        action="store_true",
        help="Save without running an immediate connection check.",
    )

    cfg_set_prov = cfg_sub.add_parser(
        "set-provider", help="Set the saved provider and base URL."
    )
    cfg_set_prov.add_argument(
        "provider_name", help="Provider identifier (e.g., openrouter, openai_compatible)."
    )
    cfg_set_prov.add_argument("--base-url", dest="new_base_url", default=None, help="API base URL.")
    cfg_set_prov.add_argument(
        "--model", dest="new_model", default=None, help="Model ID supported by this provider."
    )

    cfg_cap = cfg_sub.add_parser(
        "set-capability", help="Configure capability overrides for a model."
    )
    cfg_cap.add_argument(
        "--for-model", default=None, help="Target model ID (defaults to saved model)."
    )
    cfg_cap.add_argument("--tool-calling", choices=["on", "off"], default=None)
    cfg_cap.add_argument("--structured-output", choices=["on", "off"], default=None)
    cfg_cap.add_argument("--vision", choices=["on", "off"], default=None)

    # update
    sub_update = subparsers.add_parser(
        "update", help="Check for and apply nemotron-agent updates."
    )
    _add_global_flags(sub_update, is_subparser=True)
    sub_update.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Only report current/latest version; do not install anything.",
    )
    sub_update.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Install the update without asking for confirmation.",
    )

    # auth
    sub_auth = subparsers.add_parser("auth", help="Manage provider API credentials.")
    _add_global_flags(sub_auth, is_subparser=True)
    auth_sub = sub_auth.add_subparsers(dest="auth_action")

    auth_set = auth_sub.add_parser(
        "set-key", help="Prompt for and store an API key for a provider."
    )
    auth_set.add_argument(
        "--provider",
        dest="target_provider",
        default=None,
        help="Provider name (default: active provider).",
    )
    auth_set.add_argument(
        "--allow-file-fallback",
        action="store_true",
        help="Permit 0600 local secrets file fallback without interactive prompt if OS keyring is unavailable.",
    )
    auth_set.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip connection validation after entering the key.",
    )

    auth_clear = auth_sub.add_parser(
        "clear", help="Clear saved API credentials from keyring and local storage."
    )
    auth_clear.add_argument(
        "--provider",
        dest="target_provider",
        default=None,
        help="Clear only this provider's key (default: clear all saved credentials).",
    )

    return parser


def format_active_config_report(
    resolved: ResolvedConfig,
    credential_manager: CredentialManager,
) -> str:
    cred = credential_manager.resolve_api_key(resolved.provider)
    kr_status = credential_manager.keyring_status()
    cred_desc = (
        "not required (mock mode)"
        if resolved.mock_mode
        else (f"configured [REDACTED] (source: {cred.source})" if cred else "not configured")
    )
    caps = resolved.capabilities
    lines = [
        "Active Nemotron Agent Configuration:",
        f"  Config file       : {resolved.config_file_path} ({'saved' if resolved.has_saved_config else 'not created yet'})",
        f"  Provider          : {resolved.provider} (source: {resolved.sources.get('provider', 'default')})",
        f"  Model ID          : {resolved.model_id} (source: {resolved.sources.get('model_id', 'default')})",
        f"  Base URL          : {resolved.base_url} (source: {resolved.sources.get('base_url', 'default')})",
        f"  Mock mode         : {resolved.mock_mode}",
        f"  API credential    : {cred_desc}",
        f"  OS keyring        : {'available' if kr_status.available else 'unavailable'} ({kr_status.backend_name})",
        (
            f"  Capabilities      : tool_calling={caps.supports_tool_calling}, "
            f"structured_output={caps.supports_structured_output}, "
            f"vision={caps.supports_vision}, "
            f"max_context_chars={caps.max_context_chars}"
        ),
        (
            f"  Execution bounds  : max_turns={resolved.max_turns}, "
            f"max_tool_calls={resolved.max_tool_calls}, "
            f"request_timeout={resolved.request_timeout_seconds}s"
        ),
    ]
    return "\n".join(lines)


def handle_config_or_auth_command(
    args: argparse.Namespace,
    *,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    input_fn: InputFn = input,
    secret_input_fn: SecretInputFn = secure_secret_input,
    output_fn: OutputFn = print,
    client_factory: ClientFactory = default_client_factory,
) -> int:
    """Execute setup, ping, config, and auth subcommands."""
    if args.subcommand == "setup":
        outcome = run_interactive_setup(
            config_manager=config_manager,
            credential_manager=credential_manager,
            input_fn=input_fn,
            secret_input_fn=secret_input_fn,
            output_fn=output_fn,
            client_factory=client_factory,
        )
        output_fn(outcome.message)
        return 0 if outcome.completed else 1

    if args.subcommand == "ping":
        resolved = config_manager.resolve(
            cli_provider=args.provider,
            cli_model=args.model,
            cli_base_url=args.base_url,
            cli_mock=True if args.mock else None,
            cli_timeout=getattr(args, "timeout", None),
        )
        if resolved.mock_mode:
            client = client_factory("mock", "mock://local", "", resolved.request_timeout_seconds, 0)
            res = client.validate_connection(resolved.model_id)
            output_fn(res.message)
            return 0
        cred = credential_manager.resolve_api_key(resolved.provider)
        if cred is None:
            output_fn(
                f"Error: No API key configured for provider '{resolved.provider}'. "
                "Run `nemotron-agent auth set-key` or export the provider's API key env var."
            )
            return 1
        client = client_factory(
            resolved.provider,
            resolved.base_url,
            cred.api_key,
            resolved.request_timeout_seconds,
            resolved.max_retries,
        )
        try:
            res = client.validate_connection(resolved.model_id)
            output_fn(f"OK ({res.latency_ms} ms): {res.message}")
            return 0
        finally:
            client.close()

    if args.subcommand == "config":
        action = args.config_action or "show"
        user_cfg = config_manager.load_user_config() or UserConfig()

        if action == "presets":
            output_fn(format_presets_table(user_cfg.get_presets()))
            return 0

        if action == "show":
            resolved = config_manager.resolve(
                cli_provider=args.provider,
                cli_model=args.model,
                cli_base_url=args.base_url,
                cli_mock=True if args.mock else None,
            )
            output_fn(format_active_config_report(resolved, credential_manager))
            return 0

        if action == "set-model":
            preset = user_cfg.find_preset(args.model)
            if preset is not None:
                user_cfg.model_id = preset.model_id
                user_cfg.provider = preset.provider
                if not user_cfg.base_url and preset.provider in DEFAULT_PROVIDERS:
                    user_cfg.base_url = DEFAULT_PROVIDERS[preset.provider].default_base_url
                label = f"{preset.display_name} ({preset.model_id})"
            else:
                user_cfg.model_id = args.model.strip()
                label = user_cfg.model_id

            if not args.skip_check and user_cfg.provider != "mock":
                cred = credential_manager.resolve_api_key(user_cfg.provider)
                if cred is not None:
                    client = client_factory(
                        user_cfg.provider,
                        user_cfg.base_url,
                        cred.api_key,
                        user_cfg.request_timeout_seconds,
                        user_cfg.max_retries,
                    )
                    try:
                        client.validate_connection(user_cfg.model_id)
                    finally:
                        client.close()

            path = config_manager.save_user_config(user_cfg)
            output_fn(f"Updated model to {label} (provider: {user_cfg.provider}) in {path}.")
            return 0

        if action == "set-provider":
            prov = args.provider_name.strip().lower()
            profile = get_provider_profile(prov)
            new_url = (args.new_base_url or "").strip().rstrip("/") or profile.default_base_url
            if not new_url:
                output_fn(
                    f"Error: Provider '{prov}' requires an explicit `--base-url https://...`."
                )
                return 1
            if prov != "openrouter" and user_cfg.model_id.endswith(":free") and not args.new_model:
                output_fn(
                    "Warning: Current model ID ends with `:free` (an OpenRouter-specific identifier). "
                    "Verify and set a model ID supported by your chosen provider via `--model`."
                )
            user_cfg.provider = prov
            user_cfg.base_url = new_url
            if args.new_model:
                user_cfg.model_id = args.new_model.strip()
            path = config_manager.save_user_config(user_cfg)
            output_fn(
                f"Updated provider='{user_cfg.provider}', base_url='{user_cfg.base_url}', "
                f"model='{user_cfg.model_id}' in {path}."
            )
            return 0

        if action == "set-capability":
            target_model = (args.for_model or user_cfg.model_id).strip()
            current_caps: ModelCapabilities = user_cfg.capabilities_for(target_model)
            if args.tool_calling is not None:
                current_caps.supports_tool_calling = args.tool_calling == "on"
            if args.structured_output is not None:
                current_caps.supports_structured_output = args.structured_output == "on"
            if args.vision is not None:
                current_caps.supports_vision = args.vision == "on"
            user_cfg.capability_overrides[target_model] = current_caps
            path = config_manager.save_user_config(user_cfg)
            output_fn(
                f"Saved capability overrides for '{target_model}' in {path}: "
                f"tool_calling={current_caps.supports_tool_calling}, "
                f"structured_output={current_caps.supports_structured_output}, "
                f"vision={current_caps.supports_vision}"
            )
            return 0

    if args.subcommand == "auth":
        action = args.auth_action
        resolved = config_manager.resolve(cli_provider=args.provider)

        if action == "clear":
            cleared = credential_manager.clear_credentials(args.target_provider)
            if cleared:
                output_fn("Cleared saved credentials from: " + ", ".join(cleared))
            else:
                output_fn("No saved credentials found to clear.")
            return 0

        if action == "set-key":
            target_provider = (
                (args.target_provider or resolved.provider).strip().lower()
            )
            profile = get_provider_profile(target_provider)
            output_fn(
                f"Updating API key for provider: {profile.display_name} ({target_provider})"
            )
            output_fn(f"Where your key is used: {profile.key_usage_explanation}")
            output_fn(f"How to obtain a key: {profile.key_obtain_instructions}")
            if profile.key_management_url:
                output_fn(f"Create a key: {profile.key_management_url}")
            base_url = ""
            if not args.skip_check and target_provider != "mock":
                base_url = (
                    resolved.base_url
                    if resolved.provider == target_provider and resolved.base_url
                    else profile.default_base_url
                )
                if not base_url:
                    output_fn(
                        "Error: Configure the provider base URL first, or explicitly use --skip-check."
                    )
                    return 1
                try:
                    approved = approve_key_endpoint(
                        target_provider,
                        base_url,
                        input_fn=input_fn,
                        output_fn=output_fn,
                    )
                except AgentError as exc:
                    output_fn(f"Error: {exc}")
                    return 1
                if approved is None:
                    output_fn(
                        "Key validation cancelled; the endpoint was not approved."
                    )
                    return 1
            previous_credential = credential_manager.resolve_api_key(target_provider)
            raw_key = secret_input_fn(
                f"Paste {profile.display_name} API key (hidden): "
            ).strip()
            if not raw_key:
                output_fn("Error: API key cannot be empty.")
                return 1
            register_secret(raw_key)

            if not args.skip_check and target_provider != "mock":
                client = client_factory(
                    target_provider,
                    base_url,
                    raw_key,
                    resolved.request_timeout_seconds,
                    resolved.max_retries,
                )
                try:
                    res = client.validate_connection(resolved.model_id)
                    if not res.ok:
                        output_fn(f"Validation failed: {redact_text(res.message)}")
                        return 1
                    output_fn(
                        f"Validation succeeded ({res.latency_ms} ms): {res.message}"
                    )
                finally:
                    client.close()

            def _ask_consent(explanation: str) -> bool:
                output_fn("")
                output_fn(explanation)
                ans = (
                    input_fn(
                        "Save API key to this restricted (0600) local credentials file? [y/N]: "
                    )
                    .strip()
                    .lower()
                )
                return ans in ("y", "yes")

            try:
                saved = credential_manager.save_api_key(
                    target_provider,
                    raw_key,
                    allow_file_fallback=bool(args.allow_file_fallback),
                    confirm_fallback=_ask_consent,
                )
                credential_manager.set_session_key(target_provider, raw_key)
                output_fn(f"Stored API key in {saved.detail}.")
                if previous_credential and previous_credential.source.startswith(
                    "env:"
                ):
                    output_fn(
                        "The new key is active for this session. Update or unset the old environment key for future launches."
                    )
                return 0
            except FallbackConsentRequiredError as exc:
                output_fn(redact_text(str(exc)))
                return 1

    if args.subcommand == "update":
        return _handle_update_command(
            args,
            output_fn=output_fn,
        )

    output_fn("Unknown command. Run `nemotron-agent --help` for usage.")
    return 1


def _handle_update_command(args: argparse.Namespace, *, output_fn: OutputFn = print) -> int:
    """`nemotron-agent update`: report and optionally apply updates."""

    mgr = UpdateManager()
    info = mgr.check_for_update()
    lines: list[str] = [
        f"nemotron-agent {info.current}",
        f"Install method: {info.method.value}",
    ]
    if info.latest:
        lines.append(
            f"Latest version: {info.latest} "
            + ("(update available)" if info.available else "(up to date)")
        )
    else:
        lines.append("Latest version: unable to reach PyPI")
    if info.error:
        lines.append(f"Note: {info.error}")
    if info.dirty:
        lines.append("Warning: uncommitted changes detected in the working tree.")

    if args.check or not info.available:
        output_fn("\n".join(lines))
        return 0

    if info.dirty:
        result = mgr.apply_update(
            check_only=False, yes=args.yes, restart_on_success=bool(args.yes)
        )
        output_fn(result.message)
        return 0 if result.ok else 1

    if not args.yes:
        output_fn("\n".join(lines))
        output_fn(
            f"Update {info.latest} is available ({info.method.value}). "
            f"Run `nemotron-agent update --yes` to install and auto-restart."
        )
        return 0

    result = mgr.apply_update(check_only=False, yes=True, restart_on_success=True)
    output_fn(result.message)
    return 0 if result.ok else 1


def _build_provider_client(
    resolved: ResolvedConfig,
    credential_manager: CredentialManager,
    client_factory: ClientFactory,
) -> ProviderClient:
    if resolved.mock_mode:
        return client_factory(
            "mock", "mock://local", "", resolved.request_timeout_seconds, 0
        )
    cred = credential_manager.resolve_api_key(resolved.provider)
    api_key = cred.api_key if cred is not None else ""
    return client_factory(
        resolved.provider,
        resolved.base_url,
        api_key,
        resolved.request_timeout_seconds,
        resolved.max_retries,
    )


def run_repl(
    session: AgentSession,
    *,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    input_fn: InputFn = input,
    secret_input_fn: SecretInputFn = secure_secret_input,
    output_fn: OutputFn = print,
    client_factory: ClientFactory = default_client_factory,
    use_tui: bool | None = None,
) -> int:
    """Run the registry-driven interactive environment around the existing agent.

    By default (a real terminal) the full-screen prompt_toolkit TUI is used;
    pass ``use_tui=False`` (or ``--no-tui``) for the classic line-based REPL.
    """
    from nemotron_agent.repl.session import ReplEnvironment

    env = ReplEnvironment(
        session,
        config_manager=config_manager,
        credential_manager=credential_manager,
        input_fn=input_fn,
        secret_input_fn=secret_input_fn,
        output_fn=output_fn,
        client_factory=client_factory,
    )

    if use_tui is None:
        use_tui = (
            sys.stdin.isatty()
            and sys.stdout.isatty()
            and os.environ.get("PYTEST_CURRENT_TEST") is None
        )

    if use_tui:
        from nemotron_agent.tui.app import NemotronTuiApp

        app = NemotronTuiApp(env)
        env.input_fn = app.request_input
        env.secret_input_fn = lambda prompt: app.request_input(prompt, secret=True)
        env._output_fn = app._append_output
        return app.run()

    return env.run()


def _prompt_local_codebase_dir(
    *,
    default_hint: Path,
    from_home_dir: bool,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> Path | None:
    """Interactively ask the user which local computer folder they want to inspect."""
    if from_home_dir:
        output_fn(
            f"Note: Current working directory is your home folder ({default_hint})."
        )
    while True:
        try:
            prompt = (
                "Enter the local folder path of the codebase on your computer to inspect "
                "(or 'q' to quit): "
                if from_home_dir
                else f"Enter local codebase folder path [{default_hint}]: "
            )
            ans = input_fn(prompt).strip().strip("\"'")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nCancelled.")
            return None

        if ans.lower() in ("q", "quit", "exit"):
            return None
        if not ans and not from_home_dir:
            return default_hint

        if not ans:
            output_fn("Please enter a local folder path (for example: ~/projects/my-app).")
            continue

        candidate = Path(ans).expanduser().resolve()
        if not candidate.exists():
            output_fn(f"Directory does not exist: {candidate}")
            continue
        if not candidate.is_dir():
            output_fn(f"Path is a file, not a folder: {candidate}")
            continue
        return candidate


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    config_dir: Path | None = None,
    env: dict[str, str] | None = None,
    input_fn: InputFn = input,
    secret_input_fn: SecretInputFn = secure_secret_input,
    output_fn: OutputFn = print,
    client_factory: ClientFactory = default_client_factory,
) -> int:
    """Testable entrypoint supporting full dependency injection."""
    raw_args = _normalize_cli_argv(list(argv if argv is not None else sys.argv[1:]))
    parser = build_subcommand_parser()
    parsed = parser.parse_args(raw_args)

    configure_logging(level=logging.INFO if getattr(parsed, "verbose", False) else logging.WARNING)

    explicit_repo: str | None = getattr(parsed, "repo", None)

    if parsed.subcommand in {"setup", "config", "auth", "ping", "update"}:
        explicit_root = Path(explicit_repo).expanduser().resolve() if explicit_repo else None
        cfg_mgr = ConfigManager(config_dir=config_dir, env=env, project_root=explicit_root)
        cred_mgr = CredentialManager(config_dir=config_dir, env=env, project_root=explicit_root)
        return handle_config_or_auth_command(
            parsed,
            config_manager=cfg_mgr,
            credential_manager=cred_mgr,
            input_fn=input_fn,
            secret_input_fn=secret_input_fn,
            output_fn=output_fn,
            client_factory=client_factory,
        )

    interactive = not bool(getattr(parsed, "non_interactive", False))
    select_dir = bool(getattr(parsed, "select_dir", False))
    cwd_resolved = Path.cwd().expanduser().resolve()
    home_resolved = Path.home().expanduser().resolve()

    if explicit_repo:
        repo_root = Path(explicit_repo).expanduser().resolve()
        if not repo_root.exists() or not repo_root.is_dir():
            output_fn(
                f"Error: Local codebase folder '{repo_root}' does not exist or is not a directory."
            )
            return 1
        if select_dir and interactive:
            chosen = _prompt_local_codebase_dir(
                default_hint=repo_root,
                from_home_dir=False,
                input_fn=input_fn,
                output_fn=output_fn,
            )
            if chosen is None:
                return 1
            repo_root = chosen
    elif select_dir or cwd_resolved == home_resolved:
        if not interactive:
            output_fn(
                f"Error: Current directory is your home folder ({home_resolved}). "
                "cd into a local project folder or pass `--path /path/to/local/codebase`."
            )
            return 1
        chosen = _prompt_local_codebase_dir(
            default_hint=cwd_resolved,
            from_home_dir=(cwd_resolved == home_resolved),
            input_fn=input_fn,
            output_fn=output_fn,
        )
        if chosen is None:
            return 1
        repo_root = chosen
    else:
        repo_root = cwd_resolved

    cfg_mgr = ConfigManager(config_dir=config_dir, env=env, project_root=repo_root)
    cred_mgr = CredentialManager(config_dir=config_dir, env=env, project_root=repo_root)
    resolved = cfg_mgr.resolve(
        cli_provider=getattr(parsed, "provider", None),
        cli_model=getattr(parsed, "model", None),
        cli_base_url=getattr(parsed, "base_url", None),
        cli_mock=True if getattr(parsed, "mock", None) else None,
        cli_max_turns=getattr(parsed, "max_turns", None),
        cli_max_tool_calls=getattr(parsed, "max_tool_calls", None),
        cli_timeout=getattr(parsed, "timeout", None),
    )

    # Enter the interactive environment even on first run. The existing setup
    # wizard is deferred until the first model request (or explicit /setup), so
    # /help and other local commands remain usable without credentials.
    defer_setup = parsed.subcommand == "chat" and interactive
    ready = ensure_ready_or_run_setup(
        cfg_mgr,
        cred_mgr,
        resolved=resolved,
        interactive=interactive and not defer_setup,
        input_fn=input_fn,
        secret_input_fn=secret_input_fn,
        output_fn=output_fn,
        client_factory=client_factory,
    )
    if not ready.completed or ready.config is None:
        if not defer_setup:
            output_fn(f"Error: {ready.message}")
            return 1
        from nemotron_agent.providers.unconfigured import UnconfiguredProviderClient

        active_cfg = resolved
        client = UnconfiguredProviderClient(resolved.provider)
        output_fn("Welcome! Choose a model with /models, or use /setup to connect a provider.")
    else:
        active_cfg = ready.config
        client = _build_provider_client(active_cfg, cred_mgr, client_factory)
    session: AgentSession | None = None
    try:
        session = AgentSession(
            project_root=repo_root,
            config=active_cfg,
            provider_client=client,
            interactive=interactive,
            read_only=bool(getattr(parsed, "read_only", False)),
            extra_exclusions=list(getattr(parsed, "exclude", []) or []),
            explicit_inclusions=list(getattr(parsed, "include", []) or []),
            input_fn=input_fn,
            output_fn=output_fn,
        )

        if parsed.subcommand == "ask":
            from nemotron_agent.repl.instructions import InstructionFiles

            instructions = InstructionFiles(cfg_mgr.config_dir)
            session.set_guidance(instructions.guidance(session, warn=output_fn))
            prompt_text = " ".join(parsed.prompt).strip()
            result = session.run_task(prompt_text, show_progress=True)
            output_fn("")
            output_fn(result.answer)
            output_fn("")
            output_fn(result.format_summary())
            return 0 if result.status == "completed" else 1

        if parsed.subcommand == "chat":
            return run_repl(
                session,
                config_manager=cfg_mgr,
                credential_manager=cred_mgr,
                input_fn=input_fn,
                secret_input_fn=secret_input_fn,
                output_fn=output_fn,
                client_factory=client_factory,
                use_tui=(False if getattr(parsed, "no_tui", False) else None),
            )

        parser.print_help()
        return 0
    finally:
        # The interactive environment may have replaced the original client.
        (session.provider_client if session is not None else client).close()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except AgentError as exc:
        print(f"Error: {redact_text(str(exc))}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
