"""Markdown rendering for AI responses.

Converts markdown into styled document lines (lists of ``(style, text)``
segments) that the scrollable conversation pane displays. Code blocks are
syntax highlighted with Pygments when a language is known; otherwise they
render as plain, visually separated code.

Only a small, readable subset of markdown is supported: headings, paragraphs,
bullet and numbered lists, bold/italic, inline code, fenced code blocks,
blockquotes, tables, horizontal rules, and links. The goal is structured,
easy-to-read terminal output -- not a full HTML engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TypeAlias

from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

# A styled line: list of (prompt_toolkit style string, text) segments.
StyledLine: TypeAlias = list[tuple[str, str]]
Document: TypeAlias = list[StyledLine]


# Inline markup -----------------------------------------------------------

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"\*(.+?)\*")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _inline_segments(text: str, base_style: str) -> StyledLine:
    """Turn one paragraph line into styled segments honoring **bold**, *italic*,
    `` `code` ``, and links."""
    segments: StyledLine = []
    # Tokenize preserving order of the first matching inline marker.
    pattern = re.compile(r"\*\*[^*]+?\*\*|\*[^*]+?\*|`[^`\n]+`|\[[^\]]*\]\([^)]*\)|.")
    pos = 0
    for match in pattern.finditer(text):
        chunk = match.group(0)
        start = match.start()
        if start > pos:
            segments.append((base_style, text[pos:start]))
        pos = match.end()
        if chunk.startswith("**") and chunk.endswith("**"):
            segments.append((f"bold {base_style}", chunk[2:-2]))
        elif chunk.startswith("*") and chunk.endswith("*"):
            segments.append((f"italic {base_style}", chunk[1:-1]))
        elif chunk.startswith("`") and chunk.endswith("`"):
            segments.append((f"bold {base_style} #a5f3fc", chunk[1:-1]))
        elif chunk.startswith("[") and "(" in chunk:
            label = chunk[1 : chunk.index("]")]
            segments.append((base_style, label))
            segments.append((f"italic {base_style} #9ca3af", " (link)"))
        else:
            segments.append((base_style, chunk))
    if pos < len(text):
        segments.append((base_style, text[pos:]))
    return segments


# Block parsing -----------------------------------------------------------

_FENCE = re.compile(r"^(\s*)```(\w*)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^(\s*)([*\-+])\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)\d+\.\s+(.*)$")
_BLOCKQUOTE = re.compile(r"^(\s*)>\s?(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-+(:?-+)*:?\s*\|?\s*$")
_HR = re.compile(r"^(\s*)([*\-_])(\s*\2){2,}\s*$")


@dataclass
class _Block:
    kind: str
    lines: list[str] = field(default_factory=list)
    language: str = ""
    level: int = 0
    number: int = 0


def _parse_blocks(text: str) -> list[_Block]:
    blocks: list[_Block] = []
    current: _Block | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            blocks.append(current)
            current = None

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        fence = _FENCE.match(line)
        if fence:
            if current is not None and current.kind == "code":
                # Closing fence.
                flush()
            else:
                flush()
                current = _Block(kind="code", language=fence.group(2).strip())
            continue
        if current is not None and current.kind == "code":
            current.lines.append(line)
            continue

        if line.strip() == "":
            flush()
            continue
        if _HR.match(line):
            flush()
            blocks.append(_Block(kind="hr"))
            continue
        if _TABLE_SEP.match(line):
            # Table separator row: attach to a pending table block or start one.
            if current is not None and current.kind == "table":
                current.lines.append(line)
                continue
            flush()
            current = _Block(kind="table")
            current.lines.append(line)
            continue
        if current is not None and current.kind == "table":
            if "|" in line:
                current.lines.append(line)
                continue
            flush()
        m = _HEADING.match(line)
        if m:
            flush()
            blocks.append(_Block(kind="heading", lines=[m.group(2)], level=len(m.group(1))))
            continue
        m = _BULLET.match(line)
        if m:
            flush()
            blocks.append(
                _Block(kind="bullet", lines=[m.group(3)], level=len(m.group(1)) // 2)
            )
            continue
        m = _NUMBERED.match(line)
        if m:
            flush()
            blocks.append(
                _Block(kind="numbered", lines=[m.group(3)], level=len(m.group(1)) // 2, number=len(blocks) + 1)
            )
            continue
        m = _BLOCKQUOTE.match(line)
        if m:
            flush()
            blocks.append(_Block(kind="blockquote", lines=[m.group(2)], level=len(m.group(1))))
            continue
        # Paragraph continuation or start.
        if current is not None and current.kind == "paragraph":
            current.lines.append(line)
        else:
            flush()
            blocks.append(_Block(kind="paragraph", lines=[line]))

    flush()
    return blocks


# Pygments integration ----------------------------------------------------

_PYGMENTS_TO_STYLE: dict[str, str] = {
    "keyword": "bold",
    "keyword.declaration": "bold",
    "string": "#a5f3fc",
    "string.escape": "bold #a5f3fc",
    "comment": "italic #9ca3af",
    "function": "#f59e0b",
    "function.name": "#f59e0b",
    "class.name": "bold #f59e0b",
    "number": "#d8b4fe",
    "operator": "#cbd5e1",
    "punctuation": "#94a3b8",
    "name": "#e5e7eb",
    "name.tag": "#f472b6",
    "name.attribute": "#fcd34d",
    "name.constant": "#f472b6",
    "name.variable": "#cbd5e1",
    "literal.string": "#a5f3fc",
    "text": "#e5e7eb",
}


def _pygment_style(token_type) -> str:
    style = _PYGMENTS_TO_STYLE.get(token_type.group)
    if style is None:
        parent = token_type.parent
        if parent is not None:
            style = _PYGMENTS_TO_STYLE.get(parent.group)
    return style or "#e5e7eb"


def _highlight_code(code: str, language: str) -> Document:
    document: Document = []
    if not code.strip():
        return document
    try:
        lexer = get_lexer_by_name(language)
    except ClassNotFound:
        try:
            lexer = guess_lexer(code)
        except ClassNotFound:
            lexer = None
    formatter = TerminalFormatter()
    if lexer is None:
        return [[("#a5f3fc", line)] for line in code.split("\n")]
    highlighted = highlight(code, lexer, formatter)
    # TerminalFormatter emits ANSI escapes; parse them into segments.
    ansi_split = re.compile(r"(\x1b\[[0-9;]*m)")
    for raw_line in highlighted.split("\n"):
        segments: StyledLine = []
        current_style = "#e5e7eb"
        for piece in ansi_split.split(raw_line):
            if not piece:
                continue
            if piece.startswith("\x1b["):
                current_style = _ansi_to_style(piece)
                continue
            if piece:
                segments.append((current_style, piece))
        document.append(segments if segments else [("#e5e7eb", "")])
    return document


_ANSI_256_OR_RGB = re.compile(r"\x1b\[(?:[0-9;]*?(?:;[0-9;]*?)*)m")


def _ansi_to_style(code: str) -> str:
    """Best-effort mapping of ANSI SGR sequences to prompt_toolkit styles."""
    body = code[2:-1]
    if not body or body == "0":
        return "#e5e7eb"
    parts = body.split(";")
    style_parts: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part in ("1", "bold"):
            style_parts.append("bold")
        elif part in ("3", "italic"):
            style_parts.append("italic")
        elif part == "4":
            style_parts.append("underline")
        elif part in ("30", "37"):
            style_parts.append("#9ca3af")
        elif part == "31":
            style_parts.append("#ef4444")
        elif part == "32":
            style_parts.append("#22c55e")
        elif part == "33":
            style_parts.append("#f59e0b")
        elif part == "34":
            style_parts.append("#2563eb")
        elif part == "35":
            style_parts.append("#f472b6")
        elif part == "36":
            style_parts.append("#67e8f9")
        elif part == "38" and i + 2 < len(parts) and parts[i + 1] == "5":
            style_parts.append(f"#{int(parts[i + 2]):06x}")
            i += 2
        elif part == "39":
            style_parts.append("#e5e7eb")
        i += 1
    return " ".join(style_parts) or "#e5e7eb"


# Rendering ---------------------------------------------------------------


def render_markdown(text: str, ai_text_style: str = "#ffffff") -> Document:
    """Render markdown text into a styled document.

    Each returned line is a list of ``(style, text)`` segments suitable for a
    prompt_toolkit ``FormattedTextControl``/lexer.
    """
    if not text:
        return []
    document: Document = []
    blocks = _parse_blocks(text)

    for block in blocks:
        if block.kind == "heading":
            level = max(1, min(6, block.level))
            size_style = {
                1: "bold #ffffff",
                2: "bold #ffffff",
                3: "bold #e5e7eb",
                4: "bold #e5e7eb",
                5: "italic #e5e7eb",
                6: "italic #9ca3af",
            }[level]
            prefix = "  " * block.level
            document.append(
                _inline_segments(prefix + block.lines[0], size_style)
            )
            document.append([])  # spacing
        elif block.kind == "paragraph":
            for line in block.lines:
                document.append(_inline_segments(line, ai_text_style))
            document.append([])
        elif block.kind == "bullet":
            for line in block.lines:
                document.append(
                    [("#9ca3af", "  ▪ ")]
                    + _inline_segments(line, ai_text_style)
                )
            document.append([])
        elif block.kind == "numbered":
            document.append(
                [("#9ca3af", f"  {block.number}. ")]
                + _inline_segments(block.lines[0], ai_text_style)
            )
        elif block.kind == "blockquote":
            for line in block.lines:
                document.append([("#6b7280 italic", "▌ ")] + _inline_segments(line, f"italic {ai_text_style}"))
        elif block.kind == "hr":
            document.append([("#6b7280", "━" * 40)])
        elif block.kind == "table":
            document.extend(_render_table(block.lines))
            document.append([])
        elif block.kind == "code":
            document.append([("#9ca3af", f"─ code block{f' ({block.language})' if block.language else ''} ─")])
            document.extend(_highlight_code("\n".join(block.lines), block.language))
            document.append([("#9ca3af", "─" * 40)])
            document.append([])

    # Trailing blank lines are harmless; drop empty trailing spacer.
    while document and document[-1] == []:
        document.pop()
    return document


def _render_table(lines: list[str]) -> Document:
    """Very small markdown table renderer."""
    document: Document = []
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return document
    widths = [max(len(cell) for cell in col) for col in zip(*rows)]
    for index, row in enumerate(rows):
        segments: StyledLine = [("#9ca3af", "│ ")]
        for cell, width in zip(row, widths):
            if index == 1:
                segments.append(("bold #e5e7eb", cell.ljust(width)))
            else:
                segments.append(("#e5e7eb", cell.ljust(width)))
            segments.append(("#9ca3af", " │ "))
        document.append(segments)
        if index == 0:
            separator: StyledLine = [("#9ca3af", "├")]
            for i, w in enumerate(widths):
                separator.append(("#9ca3af", "─" * (w + 2)))
                separator.append(("#9ca3af", "┼" if i < len(widths) - 1 else "┤"))
            document.append(separator)
    return document
