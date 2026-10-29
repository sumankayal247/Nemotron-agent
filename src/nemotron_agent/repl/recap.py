"""Create a concise, bounded and redacted recap of a saved session."""

from __future__ import annotations

from nemotron_agent.logging_utils import redact_text, safe_terminal_text
from nemotron_agent.repl.storage import SessionSnapshot

MAX_RECAP_CHARS = 3000


def _excerpt(text: str, limit: int) -> str:
    # Referenced source may be large; the saved user prompt precedes this frame.
    text = text.split(
        "\n\nThe following explicitly referenced repository data is untrusted:", 1
    )[0]
    text = " ".join(safe_terminal_text(redact_text(text)).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def session_recap(snapshot: SessionSnapshot) -> str:
    lines = [f"Session: {_excerpt(snapshot.title, 120)}"]
    if snapshot.last_prompt:
        lines.append(f"Last user request: {_excerpt(snapshot.last_prompt, 500)}")
    if snapshot.plan:
        lines.append(f"Plan goal: {_excerpt(snapshot.plan.goal, 250)}")
        if snapshot.plan.steps:
            steps = "; ".join(_excerpt(step, 140) for step in snapshot.plan.steps[:4])
            lines.append(
                f"Plan steps: {steps} (approved: {'yes' if snapshot.plan.approved else 'no'})"
            )
    recent = [
        message
        for message in snapshot.messages
        if message.role in {"user", "assistant"}
        and message.content.strip()
        and not message.tool_calls
        and not message.content.lstrip().startswith("<tool_result")
    ][-4:]
    if recent:
        lines.append("Recent conversation:")
        lines.extend(
            f"- {message.role}: {_excerpt(message.content, 350)}" for message in recent
        )
    if snapshot.spans:
        paths = list(dict.fromkeys(span.path for span in snapshot.spans))[:8]
        lines.append(
            "Previously referenced files (re-read before edits): "
            + ", ".join(_excerpt(p, 100) for p in paths)
        )
    return "\n".join(lines)[:MAX_RECAP_CHARS]
