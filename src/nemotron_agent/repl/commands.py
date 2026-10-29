"""Single source of truth for slash execution, help, validation and completion."""

from __future__ import annotations

import difflib
import shlex
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

from nemotron_agent.logging_utils import AgentError, redact_text

if TYPE_CHECKING:
    from nemotron_agent.repl.session import ReplEnvironment


@dataclass(frozen=True)
class ArgumentSpec:
    synopsis: str = ""
    minimum: int = 0
    maximum: int | None = 0
    first_choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class Invocation:
    name: str
    invoked: str
    args: tuple[str, ...]


Handler = Callable[["ReplEnvironment", Invocation], bool | None]
CompletionProvider = Callable[["ReplEnvironment", list[str]], Iterable[str]]


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    usage: str
    handler: Handler
    aliases: tuple[str, ...] = ()
    arguments: ArgumentSpec = field(default_factory=ArgumentSpec)
    requires_session: bool = True
    requires_repository: bool = False
    category: str = "General"
    details: str = ""
    completer: CompletionProvider | None = None
    quote_completions: bool = False

    @property
    def accepts_arguments(self) -> bool:
        return self.arguments.maximum != 0


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._names: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        names = (command.name, *command.aliases)
        if any(
            not name or name != name.lower() or not name.replace("-", "").isalnum()
            for name in names
        ):
            raise ValueError(
                "Command names must be lowercase letters, numbers, or hyphens."
            )
        if len(set(names)) != len(names) or any(name in self._names for name in names):
            raise ValueError(f"Duplicate slash command or alias: {names}")
        self._commands[command.name] = command
        for name in names:
            self._names[name] = command

    def get(self, name: str) -> SlashCommand | None:
        return self._names.get(name.lower().lstrip("/"))

    def commands(self) -> list[SlashCommand]:
        return list(self._commands.values())

    def names(self) -> list[tuple[str, SlashCommand]]:
        # Prefer canonical commands over compatibility aliases when Tab selects
        # the first match (e.g. /mo completes to /models rather than /model).
        return sorted(
            self._names.items(), key=lambda item: (item[0] != item[1].name, item[0])
        )

    def help_text(self, name: str | None = None) -> str:
        if name:
            command = self.get(name)
            if not command:
                return f"Unknown command '/{name.lstrip('/')}'. Type /help for available commands."
            lines = [
                f"/{command.name} — {command.description}",
                f"Usage: {command.usage}",
            ]
            if command.aliases:
                lines.append(
                    "Aliases: " + ", ".join(f"/{alias}" for alias in command.aliases)
                )
            if command.arguments.synopsis:
                lines.append(f"Arguments: {command.arguments.synopsis}")
            if command.arguments.first_choices:
                lines.append("Choices: " + ", ".join(command.arguments.first_choices))
            if command.details:
                lines.append(command.details)
            return "\n".join(lines)
        lines = [
            "Nemotron Agent Commands",
            "",
            "Quick start:",
            "  Type a question to talk to the agent.",
            "  /models          Choose a model; first-time setup guides you to an API key.",
            "  Ctrl+V or /paste Attach a clipboard image (vision models, max 5 MB).",
            "  /learn           Manage instructions for every session or this project.",
            "  /resume          Continue a saved session.",
        ]
        for category in dict.fromkeys(
            command.category for command in self._commands.values()
        ):
            lines.append(f"\n{category}")
            for command in self._commands.values():
                if command.category != category:
                    continue
                for name in (command.name, *command.aliases):
                    suffix = (
                        f" (alias for /{command.name})" if name != command.name else ""
                    )
                    lines.append(f"  /{name:<14} {command.description}{suffix}")
        return "\n".join(lines)

    def dispatch(self, environment: "ReplEnvironment", line: str) -> bool:
        try:
            words = shlex.split(line, posix=sys.platform != "win32")
            if sys.platform == "win32":
                # Non-POSIX shlex keeps quotes but preserves Windows backslashes.
                words = [
                    word[1:-1]
                    if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'"
                    else word
                    for word in words
                ]
        except ValueError as exc:
            environment.output(f"Invalid command arguments: {redact_text(str(exc))}")
            return False
        if not words or not words[0].startswith("/"):
            return False
        invoked = words[0][1:].lower()
        command = self.get(invoked)
        if not command:
            names = [name for name, _ in self.names()]
            suggestions = (
                [name for name in names if name.startswith(invoked)][:4]
                if invoked
                else []
            )
            if not suggestions:
                suggestions = difflib.get_close_matches(
                    invoked, names, n=3, cutoff=0.55
                )
            hint = (
                " Did you mean " + ", ".join(f"/{name}" for name in suggestions) + "?"
                if suggestions
                else ""
            )
            environment.output(
                f"Unknown slash command '/{invoked}'.{hint} Type /help for available commands."
            )
            return False
        args = tuple(words[1:])
        if len(args) < command.arguments.minimum or (
            command.arguments.maximum is not None
            and len(args) > command.arguments.maximum
        ):
            environment.output(f"Usage: {command.usage}")
            return False
        if (
            args
            and command.arguments.first_choices
            and args[0].lower() not in command.arguments.first_choices
        ):
            environment.output(f"Usage: {command.usage}")
            return False
        if command.requires_session and environment.session is None:
            environment.output("This command requires an active session.")
            return False
        if (
            command.requires_repository
            and not environment.session.project_root.is_dir()
        ):
            environment.output("This command requires an available repository.")
            return False
        try:
            return bool(
                command.handler(environment, Invocation(command.name, invoked, args))
            )
        except AgentError as exc:
            environment.output(f"Error: {redact_text(str(exc))}")
            return False
        except (KeyboardInterrupt, EOFError):
            environment.output("\nCommand cancelled.")
            return False


def _method(name: str) -> Handler:
    def call(environment: "ReplEnvironment", invocation: Invocation) -> bool | None:
        return getattr(environment, name)(invocation)

    return call


def _choices(*choices: str) -> CompletionProvider:
    return lambda _env, _words: choices


def _help_completion(env: "ReplEnvironment", _words: list[str]) -> Iterable[str]:
    return [name for name, _ in env.registry.names()]


def _models_completion(env: "ReplEnvironment", words: list[str]) -> Iterable[str]:
    if not words:
        presets = env.user_config().get_presets()
        return ["list", "current", "use", *[str(p.index) for p in presets]]
    if words[0] == "use":
        presets = env.user_config().get_presets()
        return (
            [str(p.index) for p in presets]
            + [p.model_id for p in presets]
            + [p.slug for p in presets]
        )
    return ()


def _skills_completion(env: "ReplEnvironment", words: list[str]) -> Iterable[str]:
    if not words:
        return ("list", "info", "use", *(s.name for s in env.skills.all()))
    return [s.name for s in env.skills.all()] if words[0] in {"info", "use"} else ()


def _agents_completion(env: "ReplEnvironment", words: list[str]) -> Iterable[str]:
    if not words:
        return ("list", "info", "use")
    return [p.name for p in env.personas.all()] if words[0] in {"info", "use"} else ()


def _artifacts_completion(env: "ReplEnvironment", words: list[str]) -> Iterable[str]:
    if not words:
        return ("list", "open", "path")
    return [record.path for record in env.artifacts] if words[0] == "open" else ()


def _files_completion(env: "ReplEnvironment", _words: list[str]) -> Iterable[str]:
    files, _ = env.session.indexer.iter_allowed_files()
    return [path for _, path in files[:500]]


def _directories_completion(env: "ReplEnvironment", _words: list[str]) -> Iterable[str]:
    files, _ = env.session.indexer.iter_allowed_files()
    directories: set[str] = set()
    for _, path in files[:500]:
        parts = path.split("/")
        directories.update("/".join(parts[:i]) for i in range(1, len(parts)))
    return sorted(directories)[:500]


def create_registry() -> CommandRegistry:
    registry = CommandRegistry()
    # name, description, usage, handler, aliases, argument spec, group, detail, completion
    entries = [
        SlashCommand(
            "clear",
            "Clear the current conversation context",
            "/clear",
            _method("cmd_clear"),
            category="Session",
            details="Preserves saved sessions, plans, memory, configuration, skills, and repository files.",
        ),
        SlashCommand(
            "resume",
            "List or resume previous sessions",
            "/resume [number|id-prefix]",
            _method("cmd_resume"),
            arguments=ArgumentSpec("Session number or unique ID prefix", 0, 1),
            category="Session",
        ),
        SlashCommand(
            "plan",
            "Create or inspect a read-only execution plan",
            "/plan [create [goal]|show|clear|approve]",
            _method("cmd_plan"),
            arguments=ArgumentSpec(
                "Action and optional goal",
                0,
                None,
                ("create", "show", "clear", "approve"),
            ),
            requires_repository=True,
            category="Session",
            details="Planning uses read-only tools; approval does not execute the plan.",
        ),
        SlashCommand(
            "agents",
            "Inspect or select agent personas",
            "/agents [list|use <name>|info <name>]",
            _method("cmd_agents"),
            arguments=ArgumentSpec("Action and persona name", 0, 2),
            category="AI",
            completer=_agents_completion,
        ),
        SlashCommand(
            "skills",
            "List, inspect, or activate installed skills",
            "/skills [list|info <skill>|use <skill>]",
            _method("cmd_skills"),
            arguments=ArgumentSpec("Action and skill name", 0, 2),
            category="AI",
            completer=_skills_completion,
        ),
        SlashCommand(
            "models",
            "List or select the active model",
            "/models [list|current|<model>|use <model>]",
            _method("cmd_models"),
            aliases=("model", "presets"),
            arguments=ArgumentSpec(
                "Action and model ID, preset name, or number", 0, None
            ),
            category="AI",
            completer=_models_completion,
        ),
        SlashCommand(
            "efforts",
            "Set local execution effort and limits",
            "/efforts [low|medium|high]",
            _method("cmd_efforts"),
            arguments=ArgumentSpec(
                "Local effort level", 0, 1, ("low", "medium", "high")
            ),
            category="AI",
            details="Controls local turn/tool-call caps only; it does not claim to set provider reasoning effort.",
        ),
        SlashCommand(
            "learn",
            "Save global/project instructions or repository notes",
            "/learn [show|paths|global <instruction>|project <instruction>|list|add <note>|search <query>]",
            _method("cmd_learn"),
            arguments=ArgumentSpec(
                "Action and text; or write an instruction directly", 0, None
            ),
            completer=_choices(
                "show", "paths", "global", "project", "list", "add", "search"
            ),
            requires_repository=True,
            category="AI",
            details="Instructions are written only after approval; project instructions take priority over global instructions. Repository notes are persisted only when explicitly added.",
        ),
        SlashCommand(
            "search",
            "Search repository files",
            "/search <query>",
            _method("cmd_search"),
            arguments=ArgumentSpec("Literal search text", 1, None),
            requires_repository=True,
            category="Repository",
        ),
        SlashCommand(
            "read",
            "Read a repository text file",
            "/read <file>",
            _method("cmd_read"),
            arguments=ArgumentSpec("Repository-relative file path", 1, 1),
            requires_repository=True,
            category="Repository",
            completer=_files_completion,
            quote_completions=True,
        ),
        SlashCommand(
            "files",
            "List repository files",
            "/files [directory]",
            _method("cmd_files"),
            arguments=ArgumentSpec("Repository-relative directory", 0, 1),
            requires_repository=True,
            category="Repository",
            completer=_directories_completion,
            quote_completions=True,
        ),
        SlashCommand(
            "tools",
            "List available bounded agent tools",
            "/tools",
            _method("cmd_tools"),
            category="Repository",
        ),
        SlashCommand(
            "project",
            "Show or switch the local project folder",
            "/project [path]",
            _method("cmd_project"),
            aliases=("cd",),
            arguments=ArgumentSpec("Local folder path; /cd requires one", 0, None),
            category="Repository",
        ),
        SlashCommand(
            "artifacts",
            "Inspect agent-created files",
            "/artifacts [list|open <name>|path]",
            _method("cmd_artifacts"),
            arguments=ArgumentSpec("Action and generated file path", 0, 2),
            requires_repository=True,
            category="Output",
            details="Only files created by this agent session are listed; opening uses the safe read tool.",
            completer=_artifacts_completion,
            quote_completions=True,
        ),
        SlashCommand(
            "config",
            "Inspect or modify saved configuration",
            "/config [show|presets|set-model|set-provider|set-capability]",
            _method("cmd_config"),
            arguments=ArgumentSpec("Configuration action and arguments", 0, None),
            category="Configuration",
            completer=_choices(
                "show", "presets", "set-model", "set-provider", "set-capability"
            ),
        ),
        SlashCommand(
            "color",
            "Choose the user prompt accent color",
            "/color [orange|blue|neon_green]",
            _method("cmd_color"),
            arguments=ArgumentSpec("Prompt color", 0, 1, ("orange", "blue", "neon_green")),
            category="Configuration",
            completer=_choices("orange", "blue", "neon_green"),
        ),
        SlashCommand(
            "update",
            "Check for and install nemotron-agent updates",
            "/update [--check|--yes]",
            _method("cmd_update"),
            arguments=ArgumentSpec("Update flags", 0, 2, ("--check", "--yes")),
            category="Configuration",
            details="Never touches sessions, config, or credentials. Refuses to update a dirty git working tree. --yes installs and restarts nemotron-agent.",
        ),
        SlashCommand(
            "auth",
            "Manage provider credentials",
            "/auth [set-key|clear]",
            _method("cmd_auth"),
            arguments=ArgumentSpec("Credential action and flags", 0, None),
            category="Configuration",
            completer=_choices("set-key", "clear"),
        ),
        SlashCommand(
            "setup",
            "Run the provider setup wizard",
            "/setup",
            _method("cmd_setup"),
            category="Configuration",
        ),
        SlashCommand(
            "ping",
            "Test the active provider without repository data",
            "/ping",
            _method("cmd_ping"),
            category="Configuration",
        ),
        SlashCommand(
            "provider",
            "Show or change the active provider",
            "/provider [name [base-url]]",
            _method("cmd_provider"),
            arguments=ArgumentSpec("Provider and optional base URL", 0, 2),
            category="Configuration",
        ),
        SlashCommand(
            "key",
            "Set the active provider's API key",
            "/key",
            _method("cmd_key"),
            category="Configuration",
        ),
        SlashCommand(
            "clear-key",
            "Clear the active provider's stored key",
            "/clear-key",
            _method("cmd_clear_key"),
            category="Configuration",
        ),
        SlashCommand(
            "readonly",
            "Inspect or change strict read-only mode",
            "/readonly [on|off]",
            _method("cmd_readonly"),
            arguments=ArgumentSpec(
                "on or off", 0, 1, ("on", "off", "true", "false", "1", "0")
            ),
            category="Configuration",
        ),
        SlashCommand(
            "status",
            "Show session, model, plan, and safety limits",
            "/status",
            _method("cmd_status"),
            category="General",
        ),
        SlashCommand(
            "context",
            "Summarize current context without exposing its contents",
            "/context",
            _method("cmd_context"),
            category="General",
        ),
        SlashCommand(
            "history",
            "Show recent conversation turns",
            "/history",
            _method("cmd_history"),
            category="General",
        ),
        SlashCommand(
            "paste",
            "Attach a clipboard image to your next prompt",
            "/paste",
            _method("cmd_paste"),
            category="General",
            details="Equivalent to Ctrl+V where the terminal sends it to NA. Image input needs a vision-capable model and is limited to 5 MB.",
        ),
        SlashCommand(
            "detach",
            "Remove the unsent clipboard image",
            "/detach",
            _method("cmd_detach"),
            category="General",
        ),
        SlashCommand(
            "copy",
            "Copy the last answer when the terminal supports it",
            "/copy",
            _method("cmd_copy"),
            category="General",
        ),
        SlashCommand(
            "retry",
            "Safely retry the last request",
            "/retry",
            _method("cmd_retry"),
            category="General",
            details="Refuses automatic retries after changes, command execution, interruption, or an image request whose image was discarded.",
        ),
        SlashCommand(
            "stop",
            "Explain how to cancel the current operation",
            "/stop",
            _method("cmd_stop"),
            category="General",
            details="While a synchronous model/tool operation is running, press Ctrl+C to cancel it.",
        ),
        SlashCommand(
            "version",
            "Show the Nemotron Agent version",
            "/version",
            _method("cmd_version"),
            category="General",
        ),
        SlashCommand(
            "help",
            "Show command documentation",
            "/help [command]",
            _method("cmd_help"),
            arguments=ArgumentSpec("Optional command name", 0, 1),
            category="General",
            completer=_help_completion,
        ),
        SlashCommand(
            "exit",
            "Exit the interactive environment",
            "/exit",
            _method("cmd_exit"),
            aliases=("quit",),
            category="General",
        ),
    ]
    for command in entries:
        registry.register(command)
    return registry
