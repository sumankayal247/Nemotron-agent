"""Original Nemotron brand logo: a compact, colored, 3D-looking ``N``.

The design is intentionaly small (seven lines) so it fits the startup header
without wasting terminal space. Every line is a list of
``(style, text)`` segments prompt_toolkit can render directly.

A plain ASCII fallback is provided for terminals with limited Unicode or
color support. Do not copy Antigravity's branding: this is an original
Nemotron identity built from extruded block geometry.
"""

from __future__ import annotations

from typing import TypeAlias

# A single logo line: list of (style, text) segments.
LogoLine: TypeAlias = list[tuple[str, str]]
LogoArt: TypeAlias = list[LogoLine]

# The 3D N is drawn as a bright front face with a dimmed extrusion shifted
# down-right, plus a light top-depth connector. Color is applied by the theme
# (orange / blue / neon green), so the logo always matches the user accent.


def _front(color: str, text: str) -> LogoLine:
    return [(f"bold {color}", text)]


def _depth(color: str, text: str) -> LogoLine:
    return [(color, text)]


def _top(color: str, text: str) -> LogoLine:
    return [(f"{color}", text)]


def colored_logo(accent: str, dim: str) -> LogoArt:
    """Return the colored 3D ``N`` logo as styled lines."""
    # Front face (bright accent) and its extrusion (dimmed accent).
    return [
        [(f"bold {accent}", "  ████╗     ████╗"), (dim, " ╗")],
        [(f"bold {accent}", "  █████╗    ████║"), (dim, " ║")],
        [(f"bold {accent}", "  ██║╚██╗   ████║"), (dim, " ║")],
        [(f"bold {accent}", "  ██║ ╚██╗  ████║"), (dim, " ║")],
        [(f"bold {accent}", "  ██║  ╚██╗ ████║"), (dim, " ║")],
        [(f"bold {accent}", "  ██║   ╚███████║"), (dim, " ║")],
        [(f"bold {accent}", "  ╚═╝    ╚══════╝"), (dim, " ╝")],
    ]


def logo_lines(accent: str, dim: str) -> LogoArt:
    """Colored logo lines, ready to render."""
    return colored_logo(accent, dim)


def plain_logo() -> list[str]:
    """ASCII-only fallback for limited terminals."""
    return [
        "N   N",
        "NN  N",
        "N N N",
        "N  NN",
        "N   N",
        "",
        "NEMOTRON",
    ]


DIM_FOR_ACCENT: dict[str, str] = {
    "#f59e0b": "#b45309",   # orange -> darker amber
    "#2563eb": "#1d4ed8",   # blue -> darker blue
    "#22c55e": "#15803d",   # neon green -> darker green
}
