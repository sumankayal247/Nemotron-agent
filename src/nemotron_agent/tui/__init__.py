"""Terminal user interface for Nemotron Agent.

Built on prompt_toolkit (already a project dependency). The TUI is a
presentation layer around the existing :class:`~nemotron_agent.repl.session.ReplEnvironment`;
agent logic, sessions, permissions, and providers are untouched.
"""

from __future__ import annotations

__all__ = ["theme", "logo", "markdown", "widgets", "app"]
