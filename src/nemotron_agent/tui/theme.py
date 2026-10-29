"""Centralized color theme for the Nemotron Agent TUI.

Colors are defined here in one place and referenced by name everywhere else.
Never scatter hardcoded ANSI/color codes through the UI code.

User prompt accent (applied to the user prompt indicator, user-entered text
accents, selected user-side UI elements, and the input cursor area):
    * Orange (default)
    * Blue
    * Neon Green

AI text is primarily white and visually distinct from user text.
AI questions the user must answer are bold white so they can never be missed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Named colors used by the TUI.

    All values are prompt_toolkit-compatible inline style color strings
    (hex triples or ANSI names).
    """

    # ---- User side (configurable accent) ----
    user_prompt: str  # the ">" prompt indicator
    user_text: str  # user-entered message text where appropriate
    user_accent: str  # selected user-side UI accents / cursor area

    # ---- AI side ----
    ai_label: str  # "Nemotron:" speaker label
    ai_text: str  # primary AI response text (white)
    ai_question: str  # bold white for AI questions/requests for input

    # ---- Structural / neutral ----
    border: str  # panel and box borders
    muted: str  # secondary text, hints, timestamps
    status: str  # status bar background/text
    status_text: str  # status bar text
    highlight: str  # selected item background
    selected_text: str  # text on selected items
    error: str  # error rendering
    warn: str  # warnings
    ok: str  # success indicators
    code: str  # inline code / code block base
    logo: str  # brand logo color (tints with user accent when possible)
    input_bg: str  # input area background


# Palette definitions -------------------------------------------------------

ORANGE = "#f59e0b"   # warm, friendly default accent
BLUE = "#2563eb"     # calm, professional accent
NEON_GREEN = "#22c55e"  # fresh, energetic accent

AI_WHITE = "#ffffff"
BORDER_GRAY = "#6b7280"
MUTED_GRAY = "#9ca3af"
STATUS_BG = "#111827"
ERROR_RED = "#ef4444"
WARN_AMBER = "#f59e0b"
OK_GREEN = "#22c55e"


def orange_theme() -> Theme:
    """Default theme: orange user accent."""
    return Theme(
        user_prompt=ORANGE,
        user_text=ORANGE,
        user_accent=ORANGE,
        ai_label="bold #ffffff",
        ai_text="#ffffff",
        ai_question="bold #ffffff",
        border=BORDER_GRAY,
        muted=MUTED_GRAY,
        status=STATUS_BG,
        status_text="#e5e7eb",
        highlight="#374151",
        selected_text="#ffffff",
        error=ERROR_RED,
        warn=WARN_AMBER,
        ok=OK_GREEN,
        code="#a5f3fc",
        logo=ORANGE,
        input_bg="#0b0c10",
    )


def blue_theme() -> Theme:
    return Theme(
        user_prompt=BLUE,
        user_text=BLUE,
        user_accent=BLUE,
        ai_label="bold #ffffff",
        ai_text="#ffffff",
        ai_question="bold #ffffff",
        border=BORDER_GRAY,
        muted=MUTED_GRAY,
        status=STATUS_BG,
        status_text="#e5e7eb",
        highlight="#374151",
        selected_text="#ffffff",
        error=ERROR_RED,
        warn=WARN_AMBER,
        ok=OK_GREEN,
        code="#67e8f9",
        logo=BLUE,
        input_bg="#0b0c10",
    )


def neon_green_theme() -> Theme:
    return Theme(
        user_prompt=NEON_GREEN,
        user_text=NEON_GREEN,
        user_accent=NEON_GREEN,
        ai_label="bold #ffffff",
        ai_text="#ffffff",
        ai_question="bold #ffffff",
        border=BORDER_GRAY,
        muted=MUTED_GRAY,
        status=STATUS_BG,
        status_text="#e5e7eb",
        highlight="#374151",
        selected_text="#ffffff",
        error=ERROR_RED,
        warn=WARN_AMBER,
        ok=OK_GREEN,
        code="#6ee7b7",
        logo=NEON_GREEN,
        input_bg="#0b0c10",
    )


THEMES: dict[str, callable] = {
    "orange": orange_theme,
    "blue": blue_theme,
    "neon_green": neon_green_theme,
}

DEFAULT_THEME_NAME = "orange"


def theme_for(name: str | None) -> Theme:
    factory = THEMES.get((name or DEFAULT_THEME_NAME).lower())
    if factory is None:
        available = ", ".join(sorted(THEMES))
        raise ValueError(f"Unknown prompt color '{name}'. Available: {available}")
    return factory()


def theme_names() -> list[str]:
    return ["orange", "blue", "neon_green"]


def theme_display(name: str | None) -> str:
    return {
        "orange": "Orange",
        "blue": "Blue",
        "neon_green": "Neon Green",
    }.get((name or DEFAULT_THEME_NAME).lower(), str(name))
