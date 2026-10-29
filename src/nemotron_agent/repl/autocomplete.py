"""prompt_toolkit input and dynamic slash completion driven by the registry."""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.shortcuts import CompleteStyle

from nemotron_agent.logging_utils import AgentError, safe_terminal_text

if TYPE_CHECKING:
    from nemotron_agent.repl.session import ReplEnvironment


def _argument_tokens(text: str) -> list[tuple[str, int]]:
    """Split input into spans, retaining incomplete quoted arguments."""
    tokens: list[tuple[str, int]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        start = index
        quote = ""
        while index < len(text):
            char = text[index]
            if (
                char == "\\"
                and sys.platform != "win32"
                and quote != "'"
                and index + 1 < len(text)
            ):
                index += 2
                continue
            if quote:
                if char == quote:
                    quote = ""
            elif char in "'\"":
                quote = char
            elif char.isspace():
                break
            index += 1
        if index > start:
            tokens.append((text[start:index], index))
    return tokens


def _argument_value(token: str) -> str:
    try:
        words = shlex.split(token, posix=sys.platform != "win32")
        value = words[0] if words else ""
        if (
            sys.platform == "win32"
            and len(value) >= 2
            and value[0] == value[-1]
            and value[0] in "\"'"
        ):
            return value[1:-1]
        return value
    except ValueError:
        return token[1:] if token.startswith(("'", '"')) else token


def _quote_completion(value: str) -> str:
    if sys.platform == "win32":
        return (
            f'"{value}"' if any(ch.isspace() or ch in "'\"" for ch in value) else value
        )
    return shlex.quote(value)


class SlashCompleter(Completer):
    def __init__(self, environment: "ReplEnvironment") -> None:
        self.environment = environment

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        if " " not in text and "\t" not in text:
            prefix = text[1:].lower()
            for name, command in self.environment.registry.names():
                if name.startswith(prefix):
                    yield Completion(
                        "/" + name,
                        start_position=-len(text),
                        display_meta=safe_terminal_text(command.description).replace(
                            "\n", " "
                        ),
                    )
            return
        first, _, rest = text.partition(" ")
        command = self.environment.registry.get(first)
        if not command:
            return
        tokens = _argument_tokens(rest)
        if tokens and tokens[-1][1] == len(rest):
            raw_prefix = tokens.pop()[0]
        else:
            raw_prefix = ""
        prefix = _argument_value(raw_prefix)
        parts = [_argument_value(token) for token, _ in tokens]
        try:
            if command.completer:
                candidates = command.completer(self.environment, parts)
            else:
                candidates = command.arguments.first_choices if not parts else ()
            for candidate in dict.fromkeys(candidates):
                if candidate.isprintable() and candidate.lower().startswith(
                    prefix.lower()
                ):
                    replacement = (
                        _quote_completion(candidate)
                        if command.quote_completions
                        else candidate
                    )
                    yield Completion(
                        replacement, start_position=-len(raw_prefix), display=candidate
                    )
        except Exception:
            # Completion must never break the prompt if the repo/config disappears.
            return


def terminal_input(environment: "ReplEnvironment") -> Callable[[str], str]:
    """Return a prompt function with in-memory history, tab cycling, and Alt+Enter."""
    bindings = KeyBindings()

    @bindings.add("tab")
    def _tab(event: KeyPressEvent) -> None:
        buffer = event.app.current_buffer
        if buffer.complete_state:
            buffer.complete_next()
        else:
            buffer.start_completion(select_first=True)

    @bindings.add("escape", "enter")
    def _newline(event: KeyPressEvent) -> None:
        event.app.current_buffer.insert_text("\n")

    @bindings.add("c-v")
    def _paste_image(event: KeyPressEvent) -> None:
        def capture() -> None:
            try:
                environment.paste_clipboard_image()
            except AgentError as exc:
                environment.output(f"Could not attach image: {exc}")

        run_in_terminal(capture)

    prompt = PromptSession(
        completer=SlashCompleter(environment),
        complete_while_typing=True,
        complete_style=CompleteStyle.COLUMN,
        history=InMemoryHistory(),
        key_bindings=bindings,
    )
    return lambda label: prompt.prompt(label)
