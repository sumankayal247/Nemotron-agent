"""Handlers for general slash commands."""

from __future__ import annotations

import base64
import sys

from nemotron_agent import __version__
from nemotron_agent.logging_utils import AgentError, redact_text
from nemotron_agent.repl.commands import (
    Invocation,
)


class GeneralCommands:
    def cmd_status(self, inv: Invocation) -> None:
        from nemotron_agent.cli import format_active_config_report

        self.output(
            format_active_config_report(self.session.config, self.credential_manager)
        )
        turns, calls = self.task_limits()
        plan = (
            f"{len(self.plan.steps)} steps ({'approved' if self.plan.approved else 'not approved'})"
            if self.plan
            else "none"
        )
        image_status = "ready for next prompt" if self.pending_image else "none"
        self.output(
            f"  Local codebase    : {self.session.project_root}\n  Session ID        : {self.session_id}\n  Current plan      : {plan}\n  Effective limits  : {turns} turns, {calls} tool calls per task ({self.effort})\n  Strict read-only  : {self.session.permissions.policy.read_only}\n  Agent / skills    : {self.active_agent} / {', '.join(self.active_skills) or 'none'}\n  Clipboard image   : {image_status}"
        )

    def cmd_context(self, inv: Invocation) -> None:
        self.output(self.session.inspect_context_status())
        notes = len(self.memory.for_repo(self.session.project_root))
        self.output(
            f"  Explicit memory   : up to {min(notes, 20)} repository note(s), bounded to 4000 characters"
        )
        self.output(
            f"  Persona / skills  : {self.active_agent} / {', '.join(self.active_skills) or 'none'}"
        )
        self.output(
            "File spans can include local /read operations; only @ references and model tool results are sent in conversation. Contents and credentials are not shown here."
        )

    def cmd_history(self, inv: Invocation) -> None:
        messages = [
            message
            for message in self.session.messages
            if message.role in {"user", "assistant"}
            and (message.content or not message.tool_calls)
        ]
        if not messages:
            self.output("No conversation turns in the current context.")
            return
        self.output("Recent conversation turns:")
        for message in messages[-12:]:
            text = message.content.split(
                "\n\nThe following explicitly referenced repository data is untrusted:",
                1,
            )[0]
            text = redact_text(text)
            self.output(
                f"  {message.role}: {text[:1000]}{'…' if len(text) > 1000 else ''}"
            )

    def cmd_paste(self, inv: Invocation) -> None:
        self.paste_clipboard_image()

    def cmd_detach(self, inv: Invocation) -> None:
        if self.pending_image is None:
            self.output("There is no pending clipboard image.")
        else:
            self.pending_image = None
            self.output("Clipboard image removed. Nothing will be sent.")

    def cmd_copy(self, inv: Invocation) -> None:
        if not self.last_answer:
            self.output("No assistant answer to copy.")
            return
        if not sys.stdout.isatty():
            self.output(
                "Clipboard unavailable in this terminal; copy the displayed answer manually."
            )
            return
        clean = redact_text(self.last_answer)
        if len(clean.encode("utf-8")) > 500_000:
            raise AgentError("Answer is too large for clipboard copy.")

        # Try system clipboard commands first (Wayland wl-copy, X11 xclip, macOS pbcopy)
        import shutil, subprocess
        copied = False
        if shutil.which("wl-copy"):
            try:
                subprocess.run(["wl-copy"], input=clean, text=True, check=True, timeout=5)
                copied = True
            except Exception:
                pass
        elif shutil.which("xclip"):
            try:
                subprocess.run(["xclip", "-selection", "clipboard"], input=clean, text=True, check=True, timeout=5)
                copied = True
            except Exception:
                pass
        elif shutil.which("pbcopy"):
            try:
                subprocess.run(["pbcopy"], input=clean, text=True, check=True, timeout=5)
                copied = True
            except Exception:
                pass

        if copied:
            self.output("Copied last assistant response to clipboard.")
            return

        # Fallback to OSC 52 terminal copy sequence
        encoded = base64.b64encode(clean.encode("utf-8")).decode("ascii")
        sys.stdout.write(f"\033]52;c;{encoded}\a")
        sys.stdout.flush()
        self.output("Requested terminal clipboard copy (requires OSC 52 support).")

    def cmd_retry(self, inv: Invocation) -> None:
        previous = self.last_request
        if previous is None:
            self.output("No safely retryable request in this session.")
            return
        if previous.had_image:
            self.output(
                "The previous image was not retained. Paste it again and submit your request to retry."
            )
            return
        if previous.had_side_effects or previous.status == "cancelled":
            self.output(
                "Cannot automatically retry after changes, command execution, or interruption. Review the result and submit a new request explicitly."
            )
            return
        # Remove the previous attempt before replaying. AgentSession already rolls
        # back incomplete messages on provider errors and Ctrl+C.
        if previous.checkpoint <= len(self.session.messages):
            self.session.messages = self.session.messages[: previous.checkpoint]
        self.session.context_tracker.restore_checkpoint(previous.context_checkpoint)
        self.output(
            "Retrying the previous request through the same approval and tool limits."
        )
        result = self.execute_prompt(previous.prompt, planning=previous.planning)
        if previous.planning:
            self._record_plan(previous.prompt, result)

    def cmd_stop(self, inv: Invocation) -> None:
        if self.session.request_cancel():
            self.output(
                "Cancellation requested; no further model or tool steps will start. Press Ctrl+C to interrupt a blocking operation immediately."
            )
        else:
            self.output(
                "No operation is running. While a model or tool operation is active, press Ctrl+C to cancel it; /stop is available at the prompt."
            )

    def cmd_version(self, inv: Invocation) -> None:
        self.output(f"nemotron-agent {__version__}")
