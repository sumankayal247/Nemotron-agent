"""Simple personas and declarative skills, with no autonomous worker processes.

Skill manifests are data, not executable plug-ins. They can name existing bounded
agent tools and define slash-command workflows; neither grants new permissions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

_NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    instructions: str


class PersonaRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Persona] = {}

    def register(self, persona: Persona) -> None:
        if not _NAME.fullmatch(persona.name) or persona.name in self._items:
            raise ValueError(f"Invalid or duplicate persona: {persona.name}")
        self._items[persona.name] = persona

    def get(self, name: str) -> Persona | None:
        return self._items.get(name.lower())

    def all(self) -> list[Persona]:
        return list(self._items.values())


def default_personas() -> PersonaRegistry:
    registry = PersonaRegistry()
    for persona in (
        Persona(
            "coder",
            "General repository-aware coding assistant",
            "Make focused, reviewable changes and verify them.",
        ),
        Persona(
            "researcher",
            "Investigate and explain repository behavior",
            "Investigate evidence first and distinguish facts from assumptions.",
        ),
        Persona(
            "reviewer",
            "Review code for defects and missing tests",
            "Prioritize actionable defects, regressions, and missing tests.",
        ),
        Persona(
            "debugger",
            "Diagnose failures and propose focused fixes",
            "Reproduce or trace the failure before proposing a minimal fix.",
        ),
        Persona(
            "planner",
            "Outline implementation and verification steps",
            "Identify dependencies, risks, and verification steps before changes.",
        ),
    ):
        registry.register(persona)
    return registry


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=300)
    instructions: str = Field(min_length=1, max_length=6000)
    context: str = Field(default="", max_length=4000)
    tools: list[str] = Field(default_factory=list, max_length=30)
    workflows: dict[str, str] = Field(default_factory=dict, max_length=30)
    # Slash command name -> workflow name. Each workflow is an ordinary agent
    # request and goes through the same tool registry and approval gates.
    commands: dict[str, str] = Field(default_factory=dict, max_length=30)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError(
                "Skill names must be lowercase letters, digits, and hyphens."
            )
        return value

    @model_validator(mode="after")
    def _workflow_commands(self) -> "Skill":
        if any(not _NAME.fullmatch(name) for name in self.workflows):
            raise ValueError("Invalid workflow name.")
        if any(
            len(text) > 4000 or not text.strip() for text in self.workflows.values()
        ):
            raise ValueError(
                "Workflow instructions must be nonempty and at most 4000 characters."
            )
        if any(
            not _NAME.fullmatch(name) or target not in self.workflows
            for name, target in self.commands.items()
        ):
            raise ValueError(
                "Skill commands must have valid names and reference an existing workflow."
            )
        return self


class SkillRegistry:
    """Register built-ins or load user-installed JSON manifests from config/skills/."""

    def __init__(self) -> None:
        self._items: dict[str, Skill] = {}
        self.warnings: list[str] = []

    def register(self, skill: Skill) -> None:
        if skill.name in self._items:
            raise ValueError(f"Duplicate skill: {skill.name}")
        self._items[skill.name] = skill

    def load_directory(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json"))[:100]:
            try:
                if path.is_symlink() or path.stat().st_size > 32_768:
                    raise ValueError("symlink or oversized manifest")
                skill = Skill.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                self.register(skill)
            except (OSError, UnicodeError, ValueError, ValidationError) as exc:
                self.warnings.append(f"Skipped skill manifest {path.name}: {exc}")

    def get(self, name: str) -> Skill | None:
        return self._items.get(name.lower())

    def all(self) -> list[Skill]:
        return sorted(self._items.values(), key=lambda item: item.name)


def default_skills(config_dir: Path) -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(
        Skill(
            name="code-review",
            description="Review code for actionable defects and missing tests",
            instructions="Review for correctness, security, regressions, and missing tests. Cite evidence; do not change files unless the user asks.",
            tools=["list_files", "search_files", "read_file"],
            workflows={
                "review": "Review the requested code or change. Report actionable findings with file references and tests."
            },
            commands={"review-code": "review"},
        )
    )
    registry.load_directory(config_dir / "skills")
    return registry
