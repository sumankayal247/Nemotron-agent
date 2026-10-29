"""Reusable, keyboard-first interactive primitives for the Nemotron TUI.

Every modal is event-driven: it renders formatted text, handles arrow/Enter/Esc
input through the application's event loop, and resolves through a callback.
Nothing here blocks the terminal or runs its own event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeAlias

# A multi-line formatted body: list of lines; each line is a list of
# (style, text) segments.
ModalBody: TypeAlias = list[list[tuple[str, str]]]


def _selected_line(style: str, text: str) -> list[tuple[str, str]]:
    return [(style, "❯ "), ("bold #ffffff", text)]


def _unselected_line(style: str, text: str) -> list[tuple[str, str]]:
    return [(style, "  "), ("#e5e7eb", text)]


@dataclass
class SelectionModal:
    """A single-choice list: arrows navigate, Enter selects, Esc cancels."""

    title: str
    items: list[tuple[str, object, str]]  # (display text, value, detail)
    footer: str = "↑↓ Navigate    Enter Select    Esc Cancel"
    selected_index: int = 0
    cancelled: bool = field(default=False, init=False)
    chosen: object | None = field(default=None, init=False)
    resolved: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("SelectionModal needs at least one item")
        self.selected_index = max(0, min(self.selected_index, len(self.items) - 1))

    def body(self) -> ModalBody:
        lines: ModalBody = [[("#ffffff bold", self.title)], []]
        for index, (text, _value, detail) in enumerate(self.items):
            is_selected = index == self.selected_index
            base = "#111827 #ffffff" if is_selected else "#6b7280 #e5e7eb"
            lines.append(_selected_line(base, text) if is_selected else _unselected_line(base, text))
            if detail:
                lines.append([("#9ca3af", "    " + detail)])
        lines.append([])
        lines.append([("#9ca3af", self.footer)])
        return lines

    def on_up(self) -> None:
        self.selected_index = (self.selected_index - 1) % len(self.items)

    def on_down(self) -> None:
        self.selected_index = (self.selected_index + 1) % len(self.items)

    def on_enter(self) -> None:
        self.chosen = self.items[self.selected_index][1]
        self.resolved = True

    def on_cancel(self) -> None:
        self.cancelled = True
        self.resolved = True


@dataclass
class ConfirmModal:
    """Yes/no confirmation. Returns True/False via resolve."""

    title: str
    text: str
    yes_label: str = "Yes"
    no_label: str = "No"
    footer: str = "↑↓ Navigate    Enter Select    Esc Cancel"
    selected_yes: bool = field(default=True, init=False)
    cancelled: bool = field(default=False, init=False)
    chosen: bool | None = field(default=None, init=False)
    resolved: bool = field(default=False, init=False)

    def body(self) -> ModalBody:
        yes_style = "#111827 #ffffff" if self.selected_yes else "#6b7280 #e5e7eb"
        no_style = "#111827 #ffffff" if not self.selected_yes else "#6b7280 #e5e7eb"
        lines: ModalBody = [
            [("#ffffff bold", self.title)],
            [],
            [("#e5e7eb", self.text)],
            [],
            _selected_line(yes_style, f"{self.yes_label}        "),
            _unselected_line(no_style, f"{self.no_label}        "),
            [],
            [("#9ca3af", self.footer)],
        ]
        return lines

    def on_up(self) -> None:
        self.selected_yes = True

    def on_down(self) -> None:
        self.selected_yes = False

    def on_enter(self) -> None:
        self.chosen = self.selected_yes
        self.resolved = True

    def on_cancel(self) -> None:
        self.cancelled = True
        self.resolved = True


@dataclass
class InputModal:
    """Free-form text input. Optional hidden (secret) input."""

    title: str
    prompt: str
    initial: str = ""
    secret: bool = False
    footer: str = "Enter Submit    Esc Cancel"
    value: str = ""
    cancelled: bool = field(default=False, init=False)
    resolved: bool = field(default=False, init=False)
    submitted: str | None = field(default=None, init=False)

    def body(self) -> ModalBody:
        show = "•" * len(self.value) if self.secret else self.value
        lines: ModalBody = [
            [("#ffffff bold", self.title)],
            [],
            [("#e5e7eb", self.prompt + " "), ("#ffffff", show)],
            [],
            [("#9ca3af", self.footer)],
        ]
        return lines

    def on_enter(self) -> None:
        self.submitted = self.value
        self.resolved = True

    def on_cancel(self) -> None:
        self.cancelled = True
        self.resolved = True


@dataclass
class MessageModal:
    """Read-only message with an OK dismissal."""

    title: str
    body_text: str
    footer: str = "Enter OK    Esc Dismiss"
    resolved: bool = field(default=False, init=False)

    def body(self) -> ModalBody:
        return [
            [("#ffffff bold", self.title)],
            [],
            [("#e5e7eb", self.body_text)],
            [],
            [("#9ca3af", self.footer)],
        ]

    def on_enter(self) -> None:
        self.resolved = True

    def on_cancel(self) -> None:
        self.resolved = True


@dataclass
class PaletteModal:
    """Slash command palette: filterable, navigable list of commands."""

    title: str = "Commands"
    items: list[tuple[str, str]] = field(default_factory=list)  # (name, description)
    filter_text: str = field(default="", init=False)
    selected_index: int = field(default=0, init=False)
    cancelled: bool = field(default=False, init=False)
    chosen: str | None = field(default=None, init=False)
    resolved: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        filter_lower = self.filter_text.lower()
        self.matches = [
            (name, desc)
            for name, desc in self.items
            if filter_lower in name.lower() or filter_lower in desc.lower()
        ]
        if not self.matches:
            self.matches = [("(no matches)", "")]
        self.selected_index = max(0, min(self.selected_index, len(self.matches) - 1))

    def set_filter(self, text: str) -> None:
        self.filter_text = text
        self._rebuild()

    def body(self) -> ModalBody:
        lines: ModalBody = [
            [("#ffffff bold", self.title)],
            [("#9ca3af", "Type to filter   ↑↓ Navigate   Enter Select   Esc Cancel")],
            [],
        ]
        for index, (name, desc) in enumerate(self.matches):
            if index == self.selected_index:
                lines.append(_selected_line("#111827 #ffffff", f"/{name}"))
            else:
                lines.append(_unselected_line("#6b7280 #e5e7eb", f"/{name}"))
            if desc:
                lines.append([("#9ca3af", "    " + desc)])
        lines.append([])
        return lines

    def on_up(self) -> None:
        self.selected_index = (self.selected_index - 1) % len(self.matches)

    def on_down(self) -> None:
        self.selected_index = (self.selected_index + 1) % len(self.matches)

    def on_enter(self) -> None:
        name, _desc = self.matches[self.selected_index]
        if name != "(no matches)":
            self.chosen = name
        self.resolved = True

    def on_cancel(self) -> None:
        self.cancelled = True
        self.resolved = True


@dataclass
class FilePickerModal:
    """Interactive project file/directory picker backing the ``@`` command.

    The filter text doubles as a path: ``@`` lists the project root,
    ``@src/`` descends into ``src``, and typing a filename filters matches.
    Enter on a file selects it; Enter on a directory descends; Esc cancels.
    """

    title: str = "Files"
    root: str = "."
    cwd: str = field(default=".", init=False)
    entries: list[tuple[str, bool]] = field(default_factory=list, init=False)  # (name, is_dir)
    filter_text: str = field(default="@", init=False)
    selected_index: int = field(default=0, init=False)
    cancelled: bool = field(default=False, init=False)
    chosen: str | None = field(default=None, init=False)
    resolved: bool = field(default=False, init=False)
    message: str = ""

    def __post_init__(self) -> None:
        self.cwd = self.root
        self._reload()

    # -- filesystem -------------------------------------------------------

    def _reload(self) -> None:
        from pathlib import Path

        base = Path(self.root)
        target = Path(self.cwd)
        try:
            target = target.resolve()
            base = base.resolve()
            if not target.is_relative_to(base):
                target = base
        except (OSError, RuntimeError):
            target = base
        self.cwd = target
        self.message = ""
        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except OSError:
            entries = []
        shown: list[tuple[str, bool]] = []
        for entry in entries:
            name = entry.name
            if name.startswith(".") and name not in {".env.example"}:
                continue
            shown.append((name + ("/" if entry.is_dir() else ""), entry.is_dir()))
        self.entries = shown
        self.selected_index = max(0, min(self.selected_index, len(self.entries) - 1))

    def _current_filter_path(self) -> str:
        text = self.filter_text
        if text.startswith("@"):
            text = text[1:]
        return text.strip()

    def _apply_filter(self) -> None:
        from pathlib import Path

        path_text = self._current_filter_path()
        if not path_text:
            self.cwd = Path(self.root).resolve()
            self._reload()
            return
        base = Path(self.root).resolve()
        candidate = (base / path_text)
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            self.message = "Invalid path."
            return
        if not resolved.is_relative_to(base):
            self.message = "Path escapes the project."
            return
        if resolved.is_dir():
            self.cwd = resolved
            self._reload()
        elif resolved.exists():
            self.chosen = resolved.relative_to(base).as_posix()
            self.resolved = True
        else:
            self.message = "No such file or directory."

    def on_up(self) -> None:
        if self.entries:
            self.selected_index = (self.selected_index - 1) % len(self.entries)

    def on_down(self) -> None:
        if self.entries:
            self.selected_index = (self.selected_index + 1) % len(self.entries)

    def on_char(self, ch: str) -> None:
        self.filter_text += ch
        self.message = ""

    def on_backspace(self) -> None:
        if self.filter_text == "@":
            return
        self.filter_text = self.filter_text[:-1]
        self.message = ""

    def on_enter(self) -> None:
        # If the typed text names a directory shown in the list, descend;
        # if it names a file, select it; otherwise select the highlighted entry.
        path_text = self._current_filter_path()
        if path_text:
            self._apply_filter()
            if self.resolved or self.message:
                return
        if self.entries:
            name, is_dir = self.entries[self.selected_index]
            if is_dir:
                from pathlib import Path

                self.cwd = (self.cwd / name.rstrip("/")).resolve()
                self._reload()
            else:
                self.chosen = name.rstrip("/")
                self.resolved = True

    def on_cancel(self) -> None:
        self.cancelled = True
        self.resolved = True

    def body(self) -> ModalBody:
        lines: ModalBody = [
            [("#ffffff bold", self.title)],
            [("#9ca3af", "↑↓ Navigate   Enter Select/Open   Type to filter   Tab Complete   Esc Cancel")],
            [],
        ]
        if self.message:
            lines.append([("#ef4444", "  " + self.message)])
            lines.append([])
        rel_cwd = self.cwd
        try:
            from pathlib import Path

            rel_cwd = self.cwd.relative_to(Path(self.root).resolve())
        except ValueError:
            pass
        lines.append([("#9ca3af", f"  📂 {rel_cwd or '.'}")])
        lines.append([])
        if not self.entries:
            lines.append([("#9ca3af", "  (empty directory)")])
        for index, (name, is_dir) in enumerate(self.entries):
            icon = "📁" if is_dir else "📄"
            if index == self.selected_index:
                lines.append(_selected_line("#111827 #ffffff", f"{icon} {name}"))
            else:
                lines.append(_unselected_line("#6b7280 #e5e7eb", f"{icon} {name}"))
        lines.append([])
        return lines
