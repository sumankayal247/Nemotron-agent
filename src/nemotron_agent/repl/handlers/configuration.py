"""Handlers for configuration slash commands."""

from __future__ import annotations

import contextlib
import io

from nemotron_agent.config import ResolvedConfig, UserConfig, get_provider_profile
from nemotron_agent.logging_utils import AgentError
from nemotron_agent.repl.commands import (
    Invocation,
)
from nemotron_agent.setup_wizard import run_interactive_setup


class ConfigurationCommands:
    def _compat_command(self, name: str, args: list[str]) -> int:
        from nemotron_agent.cli import (
            build_subcommand_parser,
            handle_config_or_auth_command,
        )

        parser = build_subcommand_parser()
        errors = io.StringIO()
        try:
            with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(errors):
                parsed = parser.parse_args([name, *args])
        except SystemExit as exc:
            self.output(errors.getvalue().strip() or f"Invalid /{name} arguments.")
            return int(exc.code or 0)
        # These flags make delegated auth validation use the active provider rather
        # than accidentally falling back to a different saved/default provider.
        if name == "auth":
            parsed.provider = self.session.config.provider
            if (
                getattr(parsed, "auth_action", None) == "set-key"
                and not self.session.permissions.policy.interactive
            ):
                raise AgentError(
                    "Setting credentials requires interactive hidden input."
                )
        return handle_config_or_auth_command(
            parsed,
            config_manager=self.config_manager,
            credential_manager=self.credential_manager,
            input_fn=self.input_fn,
            secret_input_fn=self.secret_input_fn,
            output_fn=self.output,
            client_factory=self.client_factory,
        )

    def cmd_config(self, inv: Invocation) -> None:
        args = list(inv.args)
        if not args or args == ["show"]:
            from nemotron_agent.cli import format_active_config_report

            self.output(
                format_active_config_report(
                    self.session.config, self.credential_manager
                )
            )
            return
        rc = self._compat_command("config", args)
        if rc != 0:
            return
        saved = self.user_config()
        if args[0] in {"set-model", "set-provider", "set-capability"}:
            self.session.config = self.session.config.model_copy(
                update={"has_saved_config": self.config_manager.exists()}
            )
        if args[0] == "set-model":
            self._apply_model(saved.model_id)
            self.output(f"Active session model: {saved.model_id}.")
        elif args[0] == "set-provider":
            candidate = self.session.config.model_copy(
                update={
                    "provider": saved.provider,
                    "base_url": saved.base_url,
                    "model_id": saved.model_id,
                    "mock_mode": saved.provider == "mock",
                    "capabilities": saved.capabilities_for(saved.model_id),
                }
            )
            self._replace_provider(candidate)
            self.output(f"Active session provider: {candidate.provider}.")
        elif args[0] == "set-capability":
            self._apply_model(self.session.config.model_id)
        self.save()

    def cmd_auth(self, inv: Invocation) -> None:
        if not inv.args:
            self.output("Usage: /auth [set-key|clear] (use hidden input for keys)")
            return
        rc = self._compat_command("auth", list(inv.args))
        if (
            rc == 0
            and inv.args[0] in {"set-key", "clear"}
            and not self.session.config.mock_mode
        ):
            self._refresh_credentials()

    def cmd_key(self, inv: Invocation) -> None:
        self.cmd_auth(
            Invocation(
                "auth", "auth", ("set-key", "--provider", self.session.config.provider)
            )
        )

    def cmd_clear_key(self, inv: Invocation) -> None:
        self.cmd_auth(
            Invocation(
                "auth", "auth", ("clear", "--provider", self.session.config.provider)
            )
        )

    def cmd_setup(self, inv: Invocation) -> None:
        self._run_setup()

    def _run_setup(self, initial_selection: ResolvedConfig | None = None) -> bool:
        if not self.session.permissions.policy.interactive:
            raise AgentError("Setup requires interactive input.")
        outcome = run_interactive_setup(
            config_manager=self.config_manager,
            credential_manager=self.credential_manager,
            input_fn=self.input_fn,
            secret_input_fn=self.secret_input_fn,
            output_fn=self.output,
            client_factory=self.client_factory,
            initial_selection=initial_selection,
        )
        self.output(outcome.message)
        if outcome.completed and outcome.config:
            configured = outcome.config
            candidate = self.session.config.model_copy(
                update={
                    "provider": configured.provider,
                    "base_url": configured.base_url,
                    "model_id": configured.model_id,
                    "mock_mode": configured.mock_mode,
                    "capabilities": configured.capabilities,
                }
            )
            self._replace_provider(candidate)
            self.save()
            return True
        return False

    def cmd_ping(self, inv: Invocation) -> None:
        self._ensure_client()
        result = self.session.provider_client.validate_connection(
            self.session.config.model_id
        )
        self.output(
            f"{'OK' if result.ok else 'Failed'} ({result.latency_ms} ms): {result.message}"
        )

    def cmd_provider(self, inv: Invocation) -> None:
        if not inv.args:
            self.output(
                f"Active provider: {self.session.config.provider} ({self.session.config.base_url})"
            )
            return
        provider = inv.args[0].lower()
        profile = get_provider_profile(provider)
        url = inv.args[1].rstrip("/") if len(inv.args) > 1 else profile.default_base_url
        if not url:
            raise AgentError(f"Usage: /provider {provider} <https://base-url/v1>")
        # Re-validate the URL using the existing config model's validation rules.
        try:
            checked = UserConfig(
                provider=provider, base_url=url, model_id=self.session.config.model_id
            )
        except ValueError as exc:
            raise AgentError(f"Invalid provider settings: {exc}") from exc
        candidate = self.session.config.model_copy(
            update={
                "provider": provider,
                "base_url": checked.base_url,
                "mock_mode": provider == "mock",
            }
        )
        self._replace_provider(candidate)
        self.output(f"Switched active provider to '{provider}' ({url}).")
        self.save()

    def cmd_readonly(self, inv: Invocation) -> None:
        if not inv.args:
            self.output(
                f"Strict read-only mode is {'ON' if self.session.permissions.policy.read_only else 'OFF'}."
            )
            return
        value = inv.args[0].lower()
        if value in {"on", "true", "1"}:
            self.session.set_read_only(True)
            self.output(
                "Strict read-only mode enabled (file edits and commands are disabled)."
            )
        elif value in {"off", "false", "0"}:
            if self._read_only_locked:
                raise AgentError(
                    "The startup read-only lock cannot be disabled in this session."
                )
            if not self.session.permissions.policy.interactive:
                raise AgentError(
                    "Read-only mode cannot be disabled in non-interactive mode."
                )
            self.session.set_read_only(False)
            self.output(
                "Default approval-gated mode restored (edits and allowlisted commands ask permission)."
            )
        else:
            self.output("Usage: /readonly [on|off]")
            return
        self.save()
