"""Handlers for session slash commands."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from nemotron_agent.config import get_provider_profile
from nemotron_agent.logging_utils import AgentError, redact_text
from nemotron_agent.providers.base import ChatMessage, CompletionRequest
from nemotron_agent.repl.capabilities import Skill
from nemotron_agent.repl.commands import (
    ArgumentSpec,
    Invocation,
    SlashCommand,
    create_registry,
)
from nemotron_agent.repl.recap import session_recap
from nemotron_agent.repl.storage import Plan, SessionSnapshot
from nemotron_agent.repo.context import ContextSpan
from nemotron_agent.repo.discovery import is_secret_like_path

if TYPE_CHECKING:
    from nemotron_agent.agent import TaskRunResult
    from nemotron_agent.repl.session import ReplEnvironment


class SessionCommands:
    def cmd_help(self, inv: Invocation) -> None:
        self.output(self.registry.help_text(inv.args[0] if inv.args else None))

    def cmd_exit(self, inv: Invocation) -> bool:
        self.output("Goodbye.")
        return True

    def cmd_clear(self, inv: Invocation) -> None:
        # Archive the old snapshot before starting a new conversation, so /clear
        # never overwrites or deletes the previously resumable transcript.
        self.save()
        self.session.clear_context()
        self._new_identity()
        self.refresh_guidance()
        self.output("Conversation context cleared.")
        self.output("Cleared conversation history and tracked repository file context.")

    def cmd_resume(self, inv: Invocation) -> None:
        self._check_private_paths(self.session.project_root)
        if not inv.args:
            available = self.sessions.list(exclude_id=self.session_id)
            self._resume_candidates = available[:50]
            if not self._resume_candidates:
                self.output("Recent Sessions\n  No previous sessions found.")
                return
            self.output("Recent Sessions")
            for index, snapshot in enumerate(self._resume_candidates, 1):
                self.output(
                    f"  {index}. {snapshot.title}\n     {snapshot.repo_path}\n     {snapshot.updated_at} | {snapshot.provider} / {snapshot.model} | {snapshot.id[:8]}"
                )
            if len(available) > len(self._resume_candidates):
                self.output(
                    f"  Showing 50 of {len(available)} sessions; older sessions can be selected by unique ID prefix."
                )
            return
        selector = inv.args[0]
        candidates = (
            self._resume_candidates
            if self._resume_candidates is not None
            else self.sessions.list(exclude_id=self.session_id)[:50]
        )
        number = (
            int(selector)
            if selector.isascii() and selector.isdecimal() and len(selector) <= 32
            else 0
        )
        if 1 <= number <= len(candidates):
            snapshot = candidates[number - 1]
        else:
            matching = [
                item
                for item in self.sessions.list(exclude_id=self.session_id)
                if item.id.startswith(selector.lower())
            ]
            if len(matching) != 1:
                raise AgentError(
                    "Session number or ID prefix is missing or ambiguous. Run /resume to list sessions."
                )
            snapshot = matching[0]
        self._resume(snapshot)

    def _resume(self, snapshot: SessionSnapshot) -> None:
        from nemotron_agent.repl.session import _safe_endpoint, _valid_history

        try:
            root = Path(snapshot.repo_path).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise AgentError(f"Invalid saved repository path: {exc}") from exc
        if not root.is_dir():
            raise AgentError(f"Saved repository is not available: {root}")
        self._check_private_paths(root)
        if not _valid_history(snapshot.messages):
            raise AgentError(
                "Saved session has an incomplete or invalid tool-message sequence; it was not resumed."
            )
        caps = self.user_config().capabilities_for(snapshot.model)
        # A session file is data, not authority to redirect credentials. Only
        # use an endpoint already configured for the provider (or its built-in
        # default), and reject a snapshot that names a different endpoint.
        saved_config = self.config_manager.load_user_config()
        if snapshot.provider == self.session.config.provider:
            base_url = self.session.config.base_url
        elif saved_config and saved_config.provider == snapshot.provider:
            base_url = saved_config.base_url
        else:
            base_url = get_provider_profile(snapshot.provider).default_base_url
        if not base_url:
            raise AgentError(
                "The saved provider endpoint is unavailable; configure that provider before resuming."
            )
        if snapshot.base_url and snapshot.base_url != _safe_endpoint(base_url):
            raise AgentError(
                "Saved session endpoint differs from the configured provider endpoint. Select the intended provider explicitly before resuming."
            )
        candidate = self.session.config.model_copy(
            update={
                "provider": snapshot.provider,
                "model_id": snapshot.model,
                "base_url": base_url,
                "mock_mode": snapshot.provider == "mock",
                "capabilities": caps,
                "sources": {
                    **self.session.config.sources,
                    "provider": "resumed session",
                    "model_id": "resumed session",
                    "base_url": "configured",
                },
            }
        )
        spans: list[ContextSpan] = []
        sensitive_spans = False
        for span in snapshot.spans:
            try:
                path = (root / span.path).resolve()
            except (OSError, RuntimeError):
                continue
            if path.is_relative_to(root) and span.end_line >= span.start_line:
                relative = path.relative_to(root).as_posix()
                sensitive_spans |= is_secret_like_path(
                    span.path
                ) or is_secret_like_path(relative)
                spans.append(ContextSpan(relative, span.start_line, span.end_line))
        if candidate.is_free_endpoint and sensitive_spans:
            raise AgentError(
                "Refusing to resume sensitive-file context on a :free model."
            )
        self._check_free_transition(candidate)
        # Verify and construct the target client before altering the current session.
        new_client = self._client_for(candidate)
        self.save()
        old = self.session.provider_client
        old_config = self.session.config
        try:
            self.session.config = candidate
            self.session.switch_project_root(root)
        except BaseException:
            self.session.config = old_config
            new_client.close()
            raise
        self.session.provider_client = new_client
        self._client_available = True
        self._needs_setup = False
        self._pending_setup_selection = None
        self.session.permissions.policy.is_free_endpoint = candidate.is_free_endpoint
        old.close()
        self.config_manager.set_project_root(root)
        self.credential_manager.set_project_root(root)
        self.session.set_read_only(
            self._read_only_locked
            or self.session.permissions.policy.read_only
            or snapshot.read_only
        )
        self.session.messages.extend(snapshot.messages)
        self.session.context_tracker.restore_summary(spans, snapshot.context_chars)
        if self.pending_image is not None:
            self.output("Discarded the unsent image while switching sessions.")
            self.pending_image = None
        self.session_id, self.created_at, self.title = (
            snapshot.id,
            snapshot.created_at,
            snapshot.title,
        )
        self.last_user_prompt = snapshot.last_prompt
        self.last_request = (
            None  # Never automatically replay operations restored from disk.
        )
        self.last_answer = next(
            (
                m.content
                for m in reversed(snapshot.messages)
                if m.role == "assistant" and not m.tool_calls
            ),
            "",
        )
        self.plan, self.artifacts, self.effort = (
            snapshot.plan,
            list(snapshot.artifacts),
            snapshot.effort,
        )
        self.active_agent = (
            snapshot.agent if self.personas.get(snapshot.agent) else "coder"
        )
        self.active_skills = []
        self.registry = create_registry()
        for name in snapshot.skills:
            skill = self.skills.get(name)
            if skill:
                try:
                    self._activate_skill(skill)
                except AgentError as exc:
                    self.output(f"Warning: could not restore skill {name}: {exc}")
        self.resume_recap = session_recap(snapshot)
        self.refresh_guidance()
        self._resume_candidates = None
        self.output(
            f"Resumed session {snapshot.id[:8]}: {snapshot.title}\nRepository: {root}\nProvider/model: {snapshot.provider} / {snapshot.model}"
        )
        self.output(
            "Previously read files must be re-read before editing; prior approvals were not restored."
        )
        self._send_resume_recap()

    def _send_resume_recap(self) -> None:
        """Send one small, tool-free reconnect request; retain the recap for later turns."""
        self.output(
            "Sending a brief session recap and your current instructions to the selected model; no tools will run."
        )
        request = CompletionRequest(
            model_id=self.session.config.model_id,
            messages=[
                self.session.messages[0],
                ChatMessage(
                    role="user",
                    content=(
                        "A previous session was resumed. Briefly acknowledge the recap above "
                        "and say you are ready to continue. Do not request or perform any action."
                    ),
                ),
            ],
            tools=[],
            capabilities=self.session.config.capabilities,
            max_tokens=128,
        )
        try:
            response = self.session.provider_client.complete(request)
        except (AgentError, KeyboardInterrupt) as exc:
            detail = redact_text(str(exc)) or "cancelled"
            self.output(
                f"Could not send the recap now ({detail}). The restored session is still available; the recap will be included in your next request."
            )
            return
        if response.tool_calls:
            self.output(
                "The model returned an unexpected tool request; no tools were run. You can continue normally."
            )
        elif response.content.strip():
            self.output(
                "Ready to continue: " + redact_text(response.content.strip())[:800]
            )
        else:
            self.output("Session recap sent. You can continue where you left off.")

    def cmd_agents(self, inv: Invocation) -> None:
        args = inv.args
        if not args or args == ("list",):
            self.output("Available agents (personas; no autonomous workers):")
            for persona in self.personas.all():
                self.output(
                    f"  {persona.name}{' (active)' if persona.name == self.active_agent else ''} — {persona.description}"
                )
            return
        if len(args) != 2 or args[0] not in {"use", "info"}:
            self.output("Usage: /agents [list|use <name>|info <name>]")
            return
        persona = self.personas.get(args[1])
        if not persona:
            raise AgentError(f"Unknown agent {args[1]!r}.")
        if args[0] == "info":
            self.output(
                f"{persona.name}: {persona.description}\n{persona.instructions}"
            )
        else:
            self.active_agent = persona.name
            self.refresh_guidance()
            self.output(
                f"Active agent: {persona.name}. Existing tool and approval rules still apply."
            )
            self.save()

    def _activate_skill(self, skill: Skill) -> None:
        if skill.name in self.active_skills:
            return
        available = {
            tool.name
            for tool in self.session.tool_registry.get_tool_specs(read_only=False)
        }
        unknown = set(skill.tools) - available
        if unknown:
            raise AgentError(
                f"Skill names unavailable tools: {', '.join(sorted(unknown))}."
            )
        collisions = [name for name in skill.commands if self.registry.get(name)]
        if collisions:
            raise AgentError(
                f"Skill command conflicts with existing commands: {', '.join(collisions)}."
            )
        for command_name, workflow_name in skill.commands.items():
            workflow = skill.workflows[workflow_name]

            def run_workflow(
                env: ReplEnvironment,
                inv: Invocation,
                *,
                instruction: str = workflow,
                name: str = skill.name,
            ) -> None:
                task = " ".join(inv.args).strip() or "Inspect the current repository."
                env.execute_prompt(
                    f"Use the {name} skill. {instruction}\nUser task: {task}"
                )

            self.registry.register(
                SlashCommand(
                    name=command_name,
                    description=f"Run {skill.name} / {workflow_name} workflow",
                    usage=f"/{command_name} [task]",
                    handler=run_workflow,
                    arguments=ArgumentSpec("Optional workflow task", 0, None),
                    category="Skill workflows",
                )
            )
        self.active_skills.append(skill.name)

    def cmd_skills(self, inv: Invocation) -> None:
        args = inv.args
        if not args or args == ("list",):
            self.output("Available skills:")
            for skill in self.skills.all():
                self.output(
                    f"  {skill.name}{' (active)' if skill.name in self.active_skills else ''} — {skill.description}"
                )
            return
        if len(args) != 2 or args[0] not in {"info", "use"}:
            self.output("Usage: /skills [list|info <skill>|use <skill>]")
            return
        skill = self.skills.get(args[1])
        if not skill:
            raise AgentError(f"Unknown skill {args[1]!r}.")
        if args[0] == "info":
            self.output(
                f"{skill.name}: {skill.description}\nInstructions: {skill.instructions}\nExisting tools: {', '.join(skill.tools) or 'none'}\nWorkflows: {', '.join(skill.workflows) or 'none'}\nCommands: {', '.join('/' + c for c in skill.commands) or 'none'}"
            )
        else:
            self._activate_skill(skill)
            self.refresh_guidance()
            self.output(
                f"Activated skill {skill.name}; it cannot bypass tool or approval rules."
            )
            if skill.commands:
                self.output(
                    "Skill commands: "
                    + ", ".join("/" + name for name in skill.commands)
                )
            self.save()

    def cmd_models(self, inv: Invocation) -> None:
        args = list(inv.args)
        if inv.invoked == "presets":
            if args:
                self.output("Usage: /presets")
                return
            from nemotron_agent.setup_wizard import format_presets_table

            self.output(format_presets_table(self.user_config().get_presets()))
            return
        if inv.invoked == "model":
            args = ["use", *args] if args else ["current"]
        elif len(args) == 1 and args[0] not in {"list", "current", "use"}:
            args = ["use", *args]
        if not args or args == ["list"]:
            self.output("Configured model presets:")
            cfg = self.user_config()
            for preset in cfg.get_presets():
                marker = (
                    " (current)"
                    if preset.model_id == self.session.config.model_id
                    else ""
                )
                vision = (
                    ", image input"
                    if cfg.capabilities_for(preset.model_id).supports_vision
                    else ""
                )
                self.output(
                    f"  {preset.index}. {preset.display_name} — {preset.model_id}{marker} [{preset.provider}{vision}]"
                )
            self.output("Choose one with /models <number> (for example, /models 3).")
            return
        if args == ["current"]:
            self.output(
                f"Active model: {self.session.config.model_id} (provider: {self.session.config.provider})"
            )
            return
        if len(args) < 2 or args[0] != "use":
            self.output("Usage: /models [list|current|use <model>]")
            return
        selector = " ".join(args[1:])
        cfg = self.user_config()
        preset = cfg.find_preset(selector)
        model = preset.model_id if preset else selector.strip()
        if not model:
            raise AgentError("Model ID cannot be empty.")
        current = self.session.config
        selection = current.model_copy(
            update={"model_id": model, "capabilities": cfg.capabilities_for(model)}
        )
        if preset and preset.provider != current.provider and not current.mock_mode:
            profile = get_provider_profile(preset.provider)
            if not profile.default_base_url:
                raise AgentError(
                    "Configure this preset's provider endpoint with /setup first."
                )
            selection = selection.model_copy(
                update={
                    "provider": preset.provider,
                    "base_url": profile.default_base_url,
                    "mock_mode": preset.provider == "mock",
                }
            )
        self._check_free_transition(selection)
        profile = get_provider_profile(selection.provider)
        missing_key = (
            not selection.mock_mode
            and profile.requires_api_key
            and self.credential_manager.resolve_api_key(selection.provider) is None
        )
        if selection.provider != current.provider:
            self.output(
                f"This preset uses {selection.provider}, rather than {current.provider}."
            )
        if self._needs_setup or missing_key:
            self.output(
                f"Let's connect {model} using {selection.provider}. "
                "You can paste your provider API key privately."
            )
            self._pending_setup_selection = selection
            if not self._run_setup(selection):
                self.output(
                    "Model setup was not completed; the previous active model is unchanged."
                )
                return
            model = (
                self.session.config.model_id
            )  # The user may have chosen another model on retry.
        elif selection.provider != current.provider:
            self.output(
                f"This preset uses {selection.provider}; switching provider as well."
            )
            self._replace_provider(selection)
        else:
            self._apply_model(model)
        if inv.invoked == "model":  # Preserve the old /model saved-default behavior.
            cfg = (
                self.user_config()
            )  # Setup may have saved provider details since the selection.
            cfg.model_id = model
            try:
                self.config_manager.save_user_config(cfg)
            except AgentError as exc:
                self.output(
                    f"Warning: active model changed but default was not saved: {exc}"
                )
        self.output(f"Switched active model to '{model}'.")
        self.save()

    def cmd_efforts(self, inv: Invocation) -> None:
        if inv.args:
            level = inv.args[0].lower()
            if level not in {"low", "medium", "high"}:
                self.output("Usage: /efforts [low|medium|high]")
                return
            self.effort = level
        turns, calls = self.task_limits()
        self.output(
            f"Execution effort: {self.effort} (local caps: {turns} turns, {calls} tool calls per task)."
        )
        self.output(
            "Provider reasoning-effort control is not exposed by this provider interface; no reasoning parameter is sent."
        )
        self.save()

    def _show_plan(self) -> None:
        if not self.plan:
            self.output(
                "No plan yet. Use /plan create <goal> to create one without making changes."
            )
            return
        self.output(
            f"Plan — {self.plan.goal} [{'approved' if self.plan.approved else 'not approved'}]"
        )
        for index, step in enumerate(self.plan.steps, 1):
            self.output(f"  {index}. {step}")

    def cmd_plan(self, inv: Invocation) -> None:
        args = list(inv.args)
        action = (
            args.pop(0).lower()
            if args
            else ("create" if self.plan is None and self.last_user_prompt else "show")
        )
        if action == "show" and not args:
            self._show_plan()
            return
        if action == "clear" and not args:
            self.plan = None
            self.output("Plan cleared; conversation and saved sessions are unchanged.")
            self.save()
            return
        if action == "approve" and not args:
            if not self.plan:
                raise AgentError("No plan to approve.")
            self.plan.approved = True
            self.output("Plan approved. No changes or commands were executed.")
            self.save()
            return
        if action != "create":
            self.output("Usage: /plan [create [goal]|show|clear|approve]")
            return
        goal = " ".join(args).strip() or self.last_user_prompt
        if not goal:
            if not self.session.permissions.policy.interactive:
                raise AgentError("Specify a goal: /plan create <goal>.")
            goal = self.input_fn("Plan goal (empty to cancel): ").strip()
        if not goal:
            self.output("Plan creation cancelled.")
            return
        self.output(
            "Creating a read-only plan; edits and command execution are disabled."
        )
        result = self.execute_prompt(goal, planning=True)
        self._record_plan(goal, result)

    def _record_plan(self, goal: str, result: TaskRunResult | None) -> None:
        if (
            result
            and result.status in {"completed", "tool_limit"}
            and result.answer.strip()
        ):
            steps = []
            for line in result.answer.splitlines():
                match = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
                if match:
                    steps.append(match.group(1).strip()[:5000])
            if not steps:
                steps = [result.answer.strip()[:5000]]
            self.plan = Plan(goal=redact_text(goal)[:4000], steps=steps[:50])
            self.save()
            self._show_plan()

    def cmd_learn(self, inv: Invocation) -> None:
        args = list(inv.args)
        action = args.pop(0).lower() if args else "list"
        if action not in {
            "show",
            "paths",
            "global",
            "project",
            "list",
            "add",
            "search",
        }:
            if not self.session.permissions.policy.interactive:
                raise AgentError(
                    "Choose a scope explicitly: /learn project <instruction> or /learn global <instruction>."
                )
            choice = (
                self.input_fn(
                    "Save this instruction for [p]roject, [g]lobal, or [c]ancel? "
                )
                .strip()
                .lower()
            )
            if choice in {"p", "project"}:
                action = "project"
            elif choice in {"g", "global"}:
                action = "global"
            else:
                self.output("No instruction was saved.")
                return
            args = list(inv.args)
        if action == "paths" and not args:
            self.output(f"Global instructions:  {self.instructions.global_path}")
            self.output(
                f"Project instructions: {self.instructions.project_path(self.session)}"
            )
            self.output(
                "Edit these Markdown files directly, or use /learn global ... or /learn project ...."
            )
            return
        if action == "show" and not args:
            global_text = self.instructions.read_global(self.session.project_root)
            project_text = self.instructions.read_project(self.session)
            self.output(
                f"Global instructions ({self.instructions.global_path}):\n{global_text}"
            )
            if project_text is None:
                self.output(
                    "No project instruction file yet. Use /learn project <instruction> to create one with approval."
                )
            else:
                self.output(
                    f"Project instructions ({self.instructions.project_path(self.session)}):\n{project_text}"
                )
            self.output(
                "Project instructions take priority over global instructions, but never over safety rules."
            )
            return
        if action in {"global", "project"} and args:
            instruction = " ".join(args)
            if action == "global":
                message = self.instructions.add_global(self.session, instruction)
            else:
                message = self.instructions.add_project(self.session, instruction)
            self.output(message)
            self.refresh_guidance()
            return
        if action == "add" and args:
            entry = self.memory.add(self.session.project_root, " ".join(args))
            self.refresh_guidance()
            self.output(f"Saved repository-scoped knowledge: {entry.text}")
            self.output(
                "Explicit notes are included as untrusted context in future requests for this repository."
            )
            return
        if action in {"list", "search"} and (action != "search" or args):
            if action == "list" and args:
                self.output("Usage: /learn list")
                return
            query = " ".join(args).lower()
            notes = [
                entry
                for entry in self.memory.for_repo(self.session.project_root)
                if query in entry.text.lower()
            ]
            self.output(
                "Persistent repository knowledge:"
                if notes
                else "No matching persistent knowledge for this repository."
            )
            for index, entry in enumerate(notes[:100], 1):
                self.output(f"  {index}. {entry.text} ({entry.created_at})")
            if len(notes) > 100:
                self.output(
                    f"  ... {len(notes) - 100} additional notes; narrow the search."
                )
            self.output(
                "Save an instruction: /learn global <instruction> or /learn project <instruction>. "
                "Use /learn show or /learn paths to inspect the files. Notes are saved only with /learn add."
            )
            return
        self.output(
            "Usage: /learn [show|paths|global <instruction>|project <instruction>|list|add <note>|search <query>]"
        )
