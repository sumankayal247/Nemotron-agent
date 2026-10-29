"""Full-screen terminal UI for Nemotron Agent.

Built on prompt_toolkit (already a project dependency). The TUI drives the
existing :class:`~nemotron_agent.repl.session.ReplEnvironment` -- agent logic,
sessions, permissions, and providers are completely untouched. All blocking
work (model requests, slash commands, interactive approvals) runs on worker
threads; the event loop only renders and handles keys, so the terminal stays
responsive.

The alternate screen is used while the app runs and is fully restored on
exit (Ctrl+D, /exit, or a crash).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Callable, TypeAlias

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition, to_filter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window, ConditionalContainer
from prompt_toolkit.document import Document
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.processors import BeforeInput, ConditionalProcessor, AfterInput
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style

from nemotron_agent.logging_utils import AgentError, safe_terminal_text
from nemotron_agent.repl.session import ReplEnvironment
from nemotron_agent.tui.logo import DIM_FOR_ACCENT, logo_lines, plain_logo
from nemotron_agent.tui.markdown import render_markdown
from nemotron_agent.tui.theme import theme_for
from nemotron_agent.tui.widgets import (
    ConfirmModal,
    FilePickerModal,
    InputModal,
    MessageModal,
    PaletteModal,
    SelectionModal,
)

# A rendered transcript block: list of lines; each line is a list of
# (style, text) segments.
Block: TypeAlias = list[list[tuple[str, str]]]

# Fixed overhead lines outside the scrolling panes (input, borders, status,
# plus a little breathing room). Used to clamp scroll offsets.
_OVERHEAD_LINES = 5


def _pane_height(rows: int) -> int:
    return max(5, rows - _OVERHEAD_LINES)


def _flatten_block(lines: list, mouse_handler: Callable[[MouseEvent], object] | None = None) -> list[tuple]:
    """Flatten a list of lines (where each line is a list of (style, text) tuples)

    into a prompt_toolkit FormattedText fragment list with newline separators,
    optionally attaching a mouse_handler to each segment.
    """
    flat: list[tuple] = []
    for i, line in enumerate(lines):
        if i > 0:
            if mouse_handler is not None:
                flat.append(("", "\n", mouse_handler))
            else:
                flat.append(("", "\n"))
        if isinstance(line, list):
            for seg in line:
                if len(seg) == 2 and mouse_handler is not None:
                    flat.append((seg[0], seg[1], mouse_handler))
                else:
                    flat.append(seg)
        elif isinstance(line, tuple):
            if len(line) == 2 and mouse_handler is not None:
                flat.append((line[0], line[1], mouse_handler))
            else:
                flat.append(line)
    return flat


def _icon_for_output(text: str) -> tuple[str, str]:
    """Return (icon, remainder) for agent progress lines printed during a task."""
    if text.startswith("[Turn "):
        return ("● ", text)
    if text.startswith("[Tool "):
        # Antigravity style: ● ToolName(args)
        return ("● ", text)
    if text.startswith("Agent plan:"):
        return ("● ", text)
    if text.startswith("-> "):
        return ("✓ ", text[3:])
    if text.startswith("Warning:"):
        return ("! ", text)
    if text.startswith("Error:"):
        return ("✕ ", text)
    return ("● ", text)


class NemotronTuiApp(Application):
    """The Nemotron Agent full-screen TUI."""

    def __init__(self, env: ReplEnvironment) -> None:
        self.env = env
        self.theme = env.theme

        self._modal: object | None = None
        self._modal_scroll = 0
        self.busy = False

        self._blocks: list[Block] = []
        self._conv_scroll: int | None = None
        self._thinking = False

        # ------------------------------------------------------------------
        # Controls
        # ------------------------------------------------------------------
        self._input_buffer = Buffer()

        def _get_input_prompt():
            if self._modal is not None:
                return []
            return [(f"bold {self.env.theme.user_prompt}", "> ")]

        self._input_control = BufferControl(
            buffer=self._input_buffer,
            input_processors=[
                BeforeInput(_get_input_prompt),
                ConditionalProcessor(
                    AfterInput("Type your message...", style=self.env.theme.muted),
                    filter=Condition(lambda: self._modal is None and not self._input_buffer.text),
                ),
            ],
        )

        self._input_window = Window(
            self._input_control, height=1, style=self.theme.border
        )
        self._top_border = Window(char="─", height=1, style="fg:#ffffff")
        self._bottom_border = Window(char="─", height=1, style="fg:#ffffff")
        self._header_window = Window(
            FormattedTextControl(text=lambda: _flatten_block(self._render_header())),
            height=9,
            style=self.theme.border,
        )
        self._conversation_window = Window(
            FormattedTextControl(
                text=lambda: self._render_conversation_fragments()
            ),
            style=self.theme.border,
        )
        self._modal_window = Window(
            FormattedTextControl(text=lambda: _flatten_block(self._render_modal())),
            style=self.theme.border,
        )
        self._status_window = Window(
            FormattedTextControl(text=lambda: _flatten_block(self._render_status())), height=1
        )

        root = HSplit(
            [
                ConditionalContainer(
                    content=self._conversation_window,
                    filter=Condition(lambda: self._modal is None),
                ),
                ConditionalContainer(
                    content=self._modal_window,
                    filter=Condition(lambda: self._modal is not None),
                ),
                self._top_border,
                self._input_window,
                self._bottom_border,
                self._status_window,
            ]
        )

        bindings = KeyBindings()
        self._bind_keys(bindings)

        super().__init__(
            layout=Layout(root),
            key_bindings=bindings,
            full_screen=True,
            mouse_support=True,
            style=Style.from_dict({"": self.theme.ai_text}),
            erase_when_done=True,
        )

    # ------------------------------------------------------------------ #
    # Application life cycle
    # ------------------------------------------------------------------ #
    async def run_async(self, *args, **kwargs):
        self._main_loop = asyncio.get_running_loop()
        self._show_empty_state()
        return await super().run_async(*args, **kwargs)

    def _main_loop_or_none(self):
        return getattr(self, "_main_loop", None)

    def _safe_invalidate(self) -> None:
        """Trigger a UI redraw safely from the main thread or worker threads."""
        loop = self._main_loop_or_none()
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self.invalidate)
            except RuntimeError:
                pass
        else:
            try:
                self.invalidate()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Transcript rendering
    # ------------------------------------------------------------------ #
    def _add_block(self, kind: str, lines: Block) -> None:
        self._blocks.append((kind, lines))
        self._conv_scroll_to_bottom()

    def _conv_scroll_to_bottom(self) -> None:
        self._conv_scroll = None
        self._safe_invalidate()

    def _conv_scroll_delta(self, delta: int) -> None:
        try:
            rows = self.output.size.rows
        except Exception:
            rows = 24
        pane = _pane_height(rows)
        all_lines = self._get_all_conversation_lines()
        total = len(all_lines)
        max_scroll = max(0, total - pane)

        current = max_scroll if self._conv_scroll is None else self._conv_scroll
        new_scroll = current + delta
        if new_scroll >= max_scroll:
            self._conv_scroll = None
        else:
            self._conv_scroll = max(0, new_scroll)
        self._safe_invalidate()

    def _get_all_conversation_lines(self) -> Block:
        lines: Block = []
        for _kind, block in self._blocks:
            lines.extend(block)
            lines.append([])  # gap between blocks
        if lines:
            lines.pop()
        return lines

    def _on_conversation_mouse(self, mouse_event: MouseEvent) -> object:
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._conv_scroll_delta(-3)
            return None
        elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._conv_scroll_delta(3)
            return None
        return NotImplemented

    def _render_conversation_fragments(self) -> list[tuple]:
        try:
            rows = self.output.size.rows
        except Exception:
            rows = 24
        pane = _pane_height(rows)
        all_lines = self._get_all_conversation_lines()
        total = len(all_lines)
        max_scroll = max(0, total - pane)

        if self._conv_scroll is None:
            start = max_scroll
        else:
            start = max(0, min(max_scroll, self._conv_scroll))

        visible_lines = all_lines[start : start + pane]
        # Pad with empty lines if fewer than pane height to fill conversation area
        if len(visible_lines) < pane:
            visible_lines = list(visible_lines)
            while len(visible_lines) < pane:
                visible_lines.append([("", " ")])

        return _flatten_block(visible_lines, mouse_handler=self._on_conversation_mouse)

    def _render_conversation(self) -> list[Block]:
        return self._get_all_conversation_lines()

    def _render_header(self) -> Block:
        theme = self.env.theme
        accent = theme.user_prompt
        dim = DIM_FOR_ACCENT.get(accent, "#6b7280")
        try:
            art = logo_lines(accent, dim)
        except Exception:
            art = [[("#ffffff bold", line)] for line in plain_logo()]

        cfg = self.env.session.config
        lines: Block = list(art)
        lines.append([])
        lines.append([("#ffffff bold", "NEMOTRON"), (theme.muted, "   AI Coding Agent")])
        lines.append([])
        lines.append([(theme.muted, "  Repository: "), ("#ffffff", str(self.env.session.project_root))])
        lines.append([(theme.muted, "  Model: "), ("#ffffff", f"{cfg.provider} / {cfg.model_id}")])
        session_state = "Resumed" if getattr(self.env, "resume_recap", "") else "New session"
        lines.append([(theme.muted, "  Session: "), (theme.ok, session_state)])
        lines.append([])
        lines.append([(theme.muted, "  / commands   @ files   ? help   ↑↓ scroll")])
        return lines

    def _render_status(self) -> Block:
        theme = self.env.theme
        cfg = self.env.session.config
        policy = self.env.session.permissions.policy
        mode = "read-only" if policy.read_only else "accept-edits"

        try:
            cols = self.output.size.columns
        except Exception:
            cols = 80

        left_text = "? for shortcuts"
        model_display = cfg.model_id
        right_text = f"{mode}  ·  {model_display}  ·  {self.env.effort}"

        gap = max(2, cols - len(left_text) - len(right_text))
        line = [
            (theme.muted, left_text),
            ("", " " * gap),
            (theme.muted, f"{mode}  ·  "),
            ("#ffffff", model_display),
            (theme.muted, f"  ·  {self.env.effort}"),
        ]
        return [line]

    def _append_output(self, text: str) -> None:
        """Append agent/tool progress output (called from any thread)."""
        text = safe_terminal_text(text or "")
        if not text:
            return
        icon, rest = _icon_for_output(text)
        if icon == "✓ ":
            style = self.theme.ok
        elif icon == "! ":
            style = self.theme.warn
        elif icon == "✕ " or rest.startswith("Error:"):
            style = self.theme.error
        else:
            style = self.theme.muted
        self._add_block("tool", [[(style, icon), (style, rest)]])

    def _show_thinking(self) -> None:
        self._thinking = True
        self._add_block("tool", [[(self.theme.muted, "⟳ Nemotron is thinking…")]])
        self._safe_invalidate()

    def _hide_thinking(self) -> None:
        self._thinking = False
        if self._blocks and self._blocks[-1][0] == "tool":
            last = self._blocks[-1][1]
            if last and last[0] and last[0][0][1] == "⟳ Nemotron is thinking…":
                self._blocks.pop()
        self._safe_invalidate()

    def _show_error(self, text: str) -> None:
        self._add_block(
            "error",
            [
                [(self.theme.error, "✕ "), (self.theme.error, text)],
                [(self.theme.muted, "  Press Enter to continue or /help for commands.")],
            ],
        )

    def _show_empty_state(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Modals
    # ------------------------------------------------------------------ #
    def _open_modal(self, modal, focus_control=None) -> None:
        self._modal = modal
        self._modal_scroll = 0
        if focus_control is not None and hasattr(focus_control, "buffer"):
            self._input_control.buffer = focus_control.buffer
        self.invalidate()

    def _close_modal(self) -> None:
        self._modal = None
        self._input_control.buffer = self._input_buffer
        self.invalidate()

    def _focus(self, control) -> None:
        try:
            self.layout.focus(control)
        except Exception:
            pass

    def _check_modal_resolved(self) -> None:
        modal = self._modal
        if modal is None or not getattr(modal, "resolved", False):
            return
        self._close_modal()

        if isinstance(modal, PaletteModal):
            if modal.chosen:
                self._input_buffer.text = f"/{modal.chosen} "
        elif isinstance(modal, FilePickerModal):
            if modal.chosen:
                self._input_buffer.text = f"@{modal.chosen} "
        elif isinstance(modal, (ConfirmModal, InputModal, SelectionModal, MessageModal)):
            callback = getattr(modal, "_on_resolve", None)
            if callback:
                try:
                    value = None
                    if isinstance(modal, ConfirmModal):
                        value = bool(modal.chosen)
                    elif isinstance(modal, InputModal):
                        value = modal.submitted
                    elif isinstance(modal, SelectionModal):
                        value = modal.chosen
                    callback(value)
                except Exception as exc:  # pragma: no cover - defensive
                    self._show_error(str(exc))

    def _render_modal(self) -> Block:
        modal = self._modal
        if modal is None:
            return []
        body = modal.body()
        try:
            rows = self.output.size.rows
        except Exception:
            rows = 24
        pane = _pane_height(rows)
        total = len(body)
        target = None
        for index, line in enumerate(body):
            if any(style.startswith("#111827 #ffffff") for style, _ in line):
                target = index
                break
        if target is not None:
            if target < self._modal_scroll:
                self._modal_scroll = target
            elif target >= self._modal_scroll + pane:
                self._modal_scroll = target - pane + 1
        self._modal_scroll = max(0, min(max(0, total - pane), self._modal_scroll))
        return body[self._modal_scroll : self._modal_scroll + pane]

    # ------------------------------------------------------------------ #
    # Specific modal flows
    # ------------------------------------------------------------------ #
    def _open_palette(self) -> None:
        items = [
            (command.name, command.description)
            for command in self.env.registry.commands()
        ]
        modal = PaletteModal(items=sorted(items, key=lambda item: item[0]))
        buffer = Buffer(
            document=Document(text="/"),
            on_text_changed=lambda b: self._palette_filter_changed(modal, b),
        )
        self._open_modal(modal, focus_control=BufferControl(buffer=buffer))

    def _palette_filter_changed(self, modal: PaletteModal, buffer: Buffer) -> None:
        if self._modal is not modal:
            return
        text = buffer.text[1:] if buffer.text.startswith("/") else buffer.text
        modal.set_filter(text)

    def _open_file_picker(self) -> None:
        modal = FilePickerModal(root=str(self.env.session.project_root))
        buffer = Buffer(
            document=Document(text="@"),
            on_text_changed=lambda b: self._picker_filter_changed(modal, b),
        )
        self._open_modal(modal, focus_control=BufferControl(buffer=buffer))

    def _picker_filter_changed(self, modal: FilePickerModal, buffer: Buffer) -> None:
        if self._modal is not modal:
            return
        text = buffer.text
        if not text.startswith("@"):
            text = "@" + text
            buffer.text = text
        modal.filter_text = text
        modal.message = ""

        from pathlib import Path

        root = Path(modal.root).resolve()
        path_text = text[1:].strip()
        if not path_text:
            modal.cwd = root
            modal._reload()
            modal.selected_index = 0
            return
        candidate = (root / path_text)
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            modal.message = "Invalid path."
            return
        if not resolved.is_relative_to(root):
            modal.message = "Path escapes the project."
            return
        if resolved.is_dir():
            modal.cwd = resolved
            modal._reload()
            modal.selected_index = 0
        elif resolved.exists():
            modal.message = "Press Enter to select this file."
        else:
            # Fuzzy-filter the current directory entries while typing.
            needle = path_text.lower()
            matches = [
                entry
                for entry in modal.entries
                if needle in entry[0].lower()
            ]
            if matches:
                modal.entries = matches
                modal.selected_index = 0
            else:
                modal.message = "No match."

    def _show_help(self) -> None:
        theme = self.env.theme
        body = "\n".join(
            [
                "Nemotron Agent -- quick reference",
                "",
                "Type a message and press Enter to talk to the agent.",
                "  /          Command palette",
                "  @          File picker",
                "  ?          This help",
                "  /exit      Quit (Ctrl+D also quits)",
                "  ↑↓         Scroll conversation",
                "  Ctrl+C     Stop current operation",
                "",
                "Commands: /help /clear /resume /rewind /models /agents /skills",
                "          /efforts /artifacts /learn /color /update /status /exit",
            ]
        )
        self._open_modal(
            MessageModal(title="Help", body_text=body),
        )

    # ------------------------------------------------------------------ #
    # Input submission
    # ------------------------------------------------------------------ #
    def _submit_input(self) -> None:
        if self.busy or self._modal is not None:
            return
        text = (self._input_buffer.text or "").strip()
        self._input_buffer.text = ""
        if not text:
            return
        if text == "/":
            self._open_palette()
            return
        if text == "@":
            self._open_file_picker()
            return
        if text == "?":
            self._show_help()
            return
        if text.startswith("/"):
            asyncio.ensure_future(self._do_dispatch(text))
        else:
            asyncio.ensure_future(self._do_message(text))

    async def _do_message(self, text: str) -> None:
        # Echo user prompt in conversation stream
        accent = self.env.theme.user_prompt
        self._add_block("user", [[(accent, "> "), ("#ffffff bold", text)]])
        self.busy = True
        self._input_buffer.read_only = to_filter(True)
        self._show_thinking()
        self.invalidate()
        result = None
        try:
            result = await asyncio.to_thread(self.env.execute_prompt, text, print_result=False)
        except AgentError as exc:
            self._show_error(exc.safe_message)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._show_error(f"Unexpected error: {exc}")
        finally:
            self.busy = False
            self._input_buffer.read_only = to_filter(False)
            self._hide_thinking()
            self.invalidate()
        if result is not None:
            self._render_result(result)

    async def _do_dispatch(self, line: str) -> None:
        self.busy = True
        self._input_buffer.read_only = to_filter(True)
        self.invalidate()
        exit_requested = False
        try:
            exit_requested = bool(
                await asyncio.to_thread(self.env.registry.dispatch, self.env, line)
            )
        except AgentError as exc:
            self._show_error(exc.safe_message)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._show_error(f"Unexpected error: {exc}")
        finally:
            self.busy = False
            self._input_buffer.read_only = to_filter(False)
            self.invalidate()
        if exit_requested:
            try:
                self.env.save()
            except AgentError as exc:
                self._show_error(exc.safe_message)
            self.exit(0)

    def _render_result(self, result) -> None:
        theme = self.env.theme
        answer = (result.answer or "").strip()
        if answer:
            question_style = theme.ai_question if answer.endswith("?") else theme.ai_text
            document = render_markdown(answer, ai_text_style=question_style)
            self._add_block("ai", document)
        summary = (result.format_summary() or "").strip()
        if summary:
            # Format concise summary like Antigravity stats
            turn_stat = f"{result.turns_used} turns" if hasattr(result, "turns_used") else ""
            tool_stat = f"{result.tool_calls_used} tools" if hasattr(result, "tool_calls_used") else ""
            stat_line = f"● {result.status.upper()} ({', '.join(filter(None, [turn_stat, tool_stat]))})"
            self._add_block("summary", [[(theme.ok if result.status == 'completed' else theme.warn, stat_line)]])

    # ------------------------------------------------------------------ #
    # Key bindings
    # ------------------------------------------------------------------ #
    def _bind_keys(self, bindings: KeyBindings) -> None:
        app = self

        @bindings.add("c-c")
        def _(event) -> None:
            if app._modal is not None:
                app._modal.on_cancel()
                app._check_modal_resolved()
            elif app.busy:
                app.env.session.request_cancel()
                app._append_output("Cancelling current operation…")
            else:
                quit_modal = ConfirmModal(
                    title="Quit Nemotron?",
                    text="Quit the application?",
                    yes_label="Quit",
                    no_label="Stay",
                )

                def _quit(value: bool) -> None:
                    if value:
                        app.exit(0)

                quit_modal._on_resolve = _quit  # type: ignore[attr-defined]
                app._open_modal(quit_modal)

        @bindings.add("c-d")
        def _(event) -> None:
            if app.busy:
                app.env.session.request_cancel()
            else:
                app.exit(0)

        @bindings.add("enter")
        def _(event) -> None:
            if app._modal is not None:
                app._modal.on_enter()
                app._check_modal_resolved()
            elif app.busy:
                return
            else:
                app._submit_input()

        def _shift_enter(event) -> None:
            if app._modal is None and not app.busy:
                app._input_buffer.insert_text("\n")

        try:
            bindings.add("s-enter")(_shift_enter)
        except ValueError:
            # Older prompt_toolkit builds don't parse "s-enter"; shift+enter
            # simply inserts a newline via the terminal's default binding.
            pass

        def _scroll(delta: int) -> None:
            if app._modal is not None:
                modal = app._modal
                if isinstance(modal, (PaletteModal, SelectionModal, FilePickerModal)):
                    if delta < 0:
                        modal.on_up()
                    else:
                        modal.on_down()
                app.invalidate()
            else:
                app._conv_scroll_delta(delta)
                app.invalidate()

        @bindings.add("up")
        def _(event) -> None:
            _scroll(-1)

        @bindings.add("down")
        def _(event) -> None:
            _scroll(1)

        @bindings.add("pageup")
        def _(event) -> None:
            _scroll(-_pane_height(app.output.size.rows))

        @bindings.add("pagedown")
        def _(event) -> None:
            _scroll(_pane_height(app.output.size.rows))

        @bindings.add("home")
        def _(event) -> None:
            app._conv_scroll = 0
            app.invalidate()

        @bindings.add("end")
        def _(event) -> None:
            app._conv_scroll_to_bottom()
            app.invalidate()

        @bindings.add("/")
        def _(event) -> None:
            if app._modal is None and not app.busy and not app._input_buffer.text:
                app._open_palette()
            elif app._modal is None:
                app._input_buffer.insert_text("/")

        @bindings.add("@")
        def _(event) -> None:
            if app._modal is None and not app.busy and not app._input_buffer.text:
                app._open_file_picker()
            elif app._modal is None:
                app._input_buffer.insert_text("@")

        @bindings.add("?")
        def _(event) -> None:
            if app._modal is None and not app.busy and not app._input_buffer.text:
                app._show_help()
            elif app._modal is None:
                app._input_buffer.insert_text("?")

    # ------------------------------------------------------------------ #
    # TUI-facing input hooks used by ReplEnvironment from worker threads
    # ------------------------------------------------------------------ #
    def request_input(self, prompt: str, *, secret: bool = False) -> str | None:
        """Blocking modal input used by ``input_fn``/``secret_input_fn``.

        Called from worker threads: shows a modal on the main loop and waits
        for the user's answer without freezing the terminal.
        """
        loop = self._main_loop_or_none()
        if loop is None:  # pragma: no cover - defensive
            raise RuntimeError("TUI event loop is not running")
        result: dict[str, str | None] = {}
        event = threading.Event()

        def on_submit(text: str | None) -> None:
            result["value"] = text
            event.set()

        self._main_loop.call_soon_threadsafe(self._open_input_modal, prompt, secret, on_submit)
        event.wait()
        return result.get("value")

    def _open_input_modal(self, prompt: str, secret: bool, on_submit: Callable[[str | None], None]) -> None:
        modal = InputModal(title="Input", prompt=prompt, secret=secret)
        modal._on_resolve = on_submit  # type: ignore[attr-defined]
        buffer = Buffer(
            document=Document(text=""),
            on_text_changed=lambda b: setattr(modal, "value", b.text),
        )
        self._open_modal(modal, focus_control=BufferControl(buffer=buffer))

    # ------------------------------------------------------------------ #
    # Application entry point used by cli.py
    # ------------------------------------------------------------------ #
    def run(self) -> int:
        return super().run()
