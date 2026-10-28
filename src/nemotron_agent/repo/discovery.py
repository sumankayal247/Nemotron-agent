"""Repository discovery, path confinement, .gitignore matching, and secret-file filtering."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from nemotron_agent.logging_utils import PathViolationError
from nemotron_agent.permissions import PermissionManager

DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".config",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "build",
        "dist",
        "target",
        "out",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".parcel-cache",
        ".turbo",
        "coverage",
        "htmlcov",
        ".eggs",
        ".terraform",
        ".gradle",
        "vendor",
    }
)

DEFAULT_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
        ".obj",
        ".o",
        ".a",
        ".lib",
        ".class",
        ".jar",
        ".war",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".ico",
        ".tiff",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".mp3",
        ".mp4",
        ".wav",
        ".ogg",
        ".mov",
        ".avi",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".wasm",
        ".parquet",
        ".pkl",
        ".pickle",
        ".pt",
        ".safetensors",
        ".onnx",
    }
)

_SAFE_ENV_TEMPLATES: frozenset[str] = frozenset(
    {
        ".env.example",
        ".env.sample",
        ".env.template",
        ".env.dist",
    }
)

_SECRET_EXACT_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        "credentials.json",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
        "secrets.toml",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)

_SECRET_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".jks",
        ".keystore",
        ".kdbx",
        ".ovpn",
        ".secret",
    }
)

_SECRET_NAME_GLOBS: tuple[str, ...] = (
    "id_rsa.*",
    "id_dsa.*",
    "id_ecdsa.*",
    "id_ed25519.*",
    "*client_secret*.json",
    "*service-account*.json",
    "*service_account*.json",
    "*secret_key*",
    "*.pem.*",
)

_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".kt": "Kotlin",
    ".c": "C",
    ".h": "C/C++",
    ".cpp": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".sh": "Shell",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
}

_KNOWN_MANIFESTS: tuple[str, ...] = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "CMakeLists.txt",
    "pom.xml",
    "build.gradle",
    "Dockerfile",
)


def is_secret_like_path(rel_posix_path: str) -> bool:
    """Return True if a relative POSIX path matches secret, key, or credential conventions."""
    norm = rel_posix_path.strip("/").replace("\\", "/")
    if not norm:
        return False
    parts = norm.split("/")
    filename = parts[-1]
    lower_name = filename.lower()

    if lower_name in _SAFE_ENV_TEMPLATES:
        return False
    if lower_name in _SECRET_EXACT_NAMES:
        return True
    if lower_name.startswith(".env."):
        return True

    # Secret directories such as .aws/credentials or .ssh/*
    lower_parts = [p.lower() for p in parts[:-1]]
    if any(seg in {".ssh", ".aws", ".gnupg"} for seg in lower_parts):
        return True

    _, ext = os.path.splitext(lower_name)
    if ext in _SECRET_EXTENSIONS:
        return True

    for pattern in _SECRET_NAME_GLOBS:
        if fnmatch.fnmatch(lower_name, pattern):
            return True

    return False


def is_binary_file(path: Path) -> bool:
    """Fast extension + null-byte check to exclude binary files from model context."""
    if path.suffix.lower() in DEFAULT_BINARY_EXTENSIONS:
        return True
    try:
        with path.open("rb") as fh:
            sample = fh.read(4096)
        return b"\x00" in sample
    except OSError:
        return True


def _compile_gitignore_pattern(raw_pattern: str) -> tuple[bool, bool, re.Pattern[str]] | None:
    """Compile a single `.gitignore` line into `(negated, dir_only, compiled_regex)`."""
    line = raw_pattern.rstrip("\r\n")
    if not line or line.lstrip().startswith("#"):
        return None
    if not line.strip():
        return None

    negated = False
    if line.startswith("!"):
        negated = True
        line = line[1:]
    if not line:
        return None

    dir_only = line.endswith("/")
    if dir_only:
        line = line.rstrip("/")
    if not line:
        return None

    anchored = line.startswith("/")
    if anchored:
        line = line.lstrip("/")
    else:
        # If there is a slash inside (not trailing), gitignore anchors it relative to root
        if "/" in line and not line.startswith("**/"):
            anchored = True

    # Translate gitignore glob into regex over POSIX paths
    regex_parts: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        if line[i : i + 3] == "**/":
            regex_parts.append("(?:.+/)?")
            i += 3
        elif line[i : i + 3] == "/**":
            regex_parts.append("(?:/.*)?")
            i += 3
        elif line[i : i + 2] == "**":
            regex_parts.append(".*")
            i += 2
        elif line[i] == "*":
            regex_parts.append("[^/]*")
            i += 1
        elif line[i] == "?":
            regex_parts.append("[^/]")
            i += 1
        else:
            regex_parts.append(re.escape(line[i]))
            i += 1

    body = "".join(regex_parts)
    if anchored:
        full_expr = rf"^{body}(?:/.*)?$"
    else:
        full_expr = rf"^(?:.+/)?{body}(?:/.*)?$"

    return negated, dir_only, re.compile(full_expr)


class GitIgnoreMatcher:
    """Evaluates `.gitignore` rules plus configurable exclusions and explicit inclusions."""

    def __init__(
        self,
        gitignore_lines: Iterable[str] = (),
        *,
        extra_exclusions: Iterable[str] = (),
        explicit_inclusions: Iterable[str] = (),
    ) -> None:
        self._rules: list[tuple[bool, bool, re.Pattern[str]]] = []
        for raw in gitignore_lines:
            compiled = _compile_gitignore_pattern(raw)
            if compiled is not None:
                self._rules.append(compiled)
        for excl in extra_exclusions:
            compiled = _compile_gitignore_pattern(excl)
            if compiled is not None:
                self._rules.append(compiled)
        for incl in explicit_inclusions:
            clean = incl.lstrip("!")
            compiled = _compile_gitignore_pattern(f"!{clean}")
            if compiled is not None:
                self._rules.append(compiled)

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        *,
        extra_exclusions: Iterable[str] = (),
        explicit_inclusions: Iterable[str] = (),
    ) -> "GitIgnoreMatcher":
        lines: list[str] = []
        root_gitignore = project_root / ".gitignore"
        if root_gitignore.is_file():
            try:
                lines.extend(root_gitignore.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                pass
        return cls(
            lines,
            extra_exclusions=extra_exclusions,
            explicit_inclusions=explicit_inclusions,
        )

    def is_ignored(self, rel_posix_path: str, *, is_dir: bool = False) -> bool:
        norm = rel_posix_path.strip("/").replace("\\", "/")
        if not norm or norm == ".":
            return False

        parts = norm.split("/")
        for idx, segment in enumerate(parts):
            is_segment_dir = is_dir or (idx < len(parts) - 1)
            if is_segment_dir and (
                segment in DEFAULT_EXCLUDED_DIRS or segment.endswith(".egg-info")
            ):
                return True

        ignored = False
        for negated, dir_only, regex in self._rules:
            if dir_only and not is_dir and "/" not in norm:
                # A directory-only pattern does not match a top-level plain file of the same name
                continue
            if dir_only and not is_dir:
                # Match only if a parent directory segment matches the pattern
                parent_norm = "/".join(parts[:-1])
                if not parent_norm or not regex.match(parent_norm):
                    continue
                ignored = not negated
            elif regex.match(norm):
                ignored = not negated
        return ignored


class PathGuard:
    """Enforces project-root confinement, symlink escape checks, and secret-file rules."""

    def __init__(self, project_root: Path, permission_manager: PermissionManager) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.permissions = permission_manager

    def resolve_path(self, raw_path: str | Path, *, operation: str = "read") -> tuple[Path, str]:
        """Resolve `raw_path` and verify confinement within `project_root`.

        Returns `(resolved_absolute_path, relative_posix_path)`.
        """
        raw_str = str(raw_path).strip()
        if not raw_str:
            raw_str = "."
        if "\x00" in raw_str:
            raise PathViolationError("Invalid path containing null byte.")

        candidate_unres = Path(raw_str).expanduser()
        if not candidate_unres.is_absolute():
            candidate_unres = self.project_root / candidate_unres

        resolved = candidate_unres.resolve()

        if resolved == self.project_root:
            return resolved, "."

        if resolved.is_relative_to(self.project_root):
            rel_posix = resolved.relative_to(self.project_root).as_posix()
            first_seg = rel_posix.split("/", 1)[0]
            if first_seg == ".git":
                raise PathViolationError(
                    f"Access to internal `.git` directory ('{rel_posix}') is blocked."
                )
            if is_secret_like_path(rel_posix):
                self.permissions.check_secret_file_access(rel_posix, operation=operation)
            return resolved, rel_posix

        # Outside project root -> delegates to PermissionManager (rejected by default)
        self.permissions.check_outside_root(resolved, self.project_root, operation=operation)
        if is_secret_like_path(resolved.name):
            self.permissions.check_secret_file_access(str(resolved), operation=operation)
        return resolved, str(resolved)


@dataclass
class RepoOverview:
    """Structured lightweight summary distinguishing verified facts from inferences."""

    project_root: str
    is_git_repo: bool
    languages: dict[str, int]
    manifests: list[str]
    top_level_entries: list[str]
    indexed_file_count: int
    excluded_secret_count: int
    inferred_commands: list[dict[str, str]] = field(default_factory=list)

    def format_text(self) -> str:
        git_desc = "yes" if self.is_git_repo else "no (plain local directory)"
        lines = [
            "Local Codebase Overview:",
            "  Verified Facts:",
            f"    Local project root    : {self.project_root}",
            f"    Git initialized       : {git_desc}",
            f"    Indexed source files  : {self.indexed_file_count}",
            f"    Excluded secret files : {self.excluded_secret_count}",
        ]
        if self.languages:
            lang_str = ", ".join(f"{lang} ({cnt})" for lang, cnt in self.languages.items())
            lines.append(f"    Detected languages    : {lang_str}")
        else:
            lines.append("    Detected languages    : none detected (empty or docs-only repository)")

        if self.manifests:
            lines.append(f"    Build/package files   : {', '.join(self.manifests)}")
        else:
            lines.append("    Build/package files   : none found at root")

        if self.top_level_entries:
            lines.append(f"    Top-level layout      : {', '.join(self.top_level_entries[:25])}")

        lines.append("  Inferred Likely Commands (unverified until run):")
        if self.inferred_commands:
            for item in self.inferred_commands:
                lines.append(f"    - {item['command']}  (inferred from {item['source']})")
        else:
            lines.append("    - None inferred from root manifests")
        return "\n".join(lines)


class RepositoryIndexer:
    """Discovers allowed repository files while respecting `.gitignore` and secret filters."""

    def __init__(
        self,
        project_root: Path,
        *,
        extra_exclusions: Iterable[str] = (),
        explicit_inclusions: Iterable[str] = (),
        max_indexed_files: int = 5000,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.matcher = GitIgnoreMatcher.from_project_root(
            self.project_root,
            extra_exclusions=extra_exclusions,
            explicit_inclusions=explicit_inclusions,
        )
        self.max_indexed_files = max_indexed_files

    def iter_allowed_files(
        self,
        *,
        subpath: Path | None = None,
    ) -> tuple[list[tuple[Path, str]], int]:
        """Walk repository returning `(allowed_files, skipped_secret_count)`.

        Each item in `allowed_files` is `(resolved_path, rel_posix_path)`.
        Symlinks escaping `project_root`, ignored paths, binaries, and secret-like files are skipped.
        """
        start_dir = (subpath.resolve() if subpath is not None else self.project_root)
        allowed: list[tuple[Path, str]] = []
        secret_skipped = 0

        for root_str, dirs, files in os.walk(start_dir, topdown=True, followlinks=False):
            root_path = Path(root_str)

            # Prune directories in place for fast traversal
            kept_dirs: list[str] = []
            for d in sorted(dirs):
                d_path = root_path / d
                if d_path.is_symlink():
                    continue
                try:
                    rel_d = d_path.resolve().relative_to(self.project_root).as_posix()
                except ValueError:
                    continue
                if self.matcher.is_ignored(rel_d, is_dir=True):
                    continue
                if is_secret_like_path(rel_d):
                    secret_skipped += 1
                    continue
                kept_dirs.append(d)
            dirs[:] = kept_dirs

            for fname in sorted(files):
                fpath = root_path / fname
                if fpath.is_symlink():
                    try:
                        resolved_f = fpath.resolve()
                    except OSError:
                        continue
                    if not resolved_f.is_relative_to(self.project_root):
                        continue
                else:
                    resolved_f = fpath.resolve()

                try:
                    rel_f = resolved_f.relative_to(self.project_root).as_posix()
                except ValueError:
                    continue

                if is_secret_like_path(rel_f):
                    secret_skipped += 1
                    continue

                if self.matcher.is_ignored(rel_f, is_dir=False):
                    continue

                if resolved_f.suffix.lower() in DEFAULT_BINARY_EXTENSIONS:
                    continue

                allowed.append((resolved_f, rel_f))
                if len(allowed) >= self.max_indexed_files:
                    return allowed, secret_skipped

        return allowed, secret_skipped

    def inspect_overview(self) -> RepoOverview:
        """Produce a lightweight repository summary without dumping source contents."""
        allowed_files, secret_count = self.iter_allowed_files()
        is_git = (self.project_root / ".git").exists()

        lang_counts: dict[str, int] = {}
        for _, rel_f in allowed_files:
            ext = os.path.splitext(rel_f)[1].lower()
            lang = _LANGUAGE_BY_EXT.get(ext)
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        sorted_langs = dict(sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0])))

        manifests = [m for m in _KNOWN_MANIFESTS if (self.project_root / m).is_file()]

        top_entries: list[str] = []
        try:
            for item in sorted(self.project_root.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
                if item.name == ".git":
                    continue
                rel_name = item.name
                if is_secret_like_path(rel_name):
                    continue
                if self.matcher.is_ignored(rel_name, is_dir=item.is_dir()):
                    continue
                top_entries.append(f"{rel_name}/" if item.is_dir() else rel_name)
        except OSError:
            pass

        inferred = self._infer_commands(manifests)
        return RepoOverview(
            project_root=str(self.project_root),
            is_git_repo=is_git,
            languages=sorted_langs,
            manifests=manifests,
            top_level_entries=top_entries,
            indexed_file_count=len(allowed_files),
            excluded_secret_count=secret_count,
            inferred_commands=inferred,
        )

    def _infer_commands(self, manifests: list[str]) -> list[dict[str, str]]:
        inferred: list[dict[str, str]] = []
        pyproject = self.project_root / "pyproject.toml"
        if "pyproject.toml" in manifests and pyproject.is_file():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                tool_tbl = data.get("tool", {})
                if isinstance(tool_tbl, dict) and "pytest" in tool_tbl:
                    inferred.append({"command": "pytest", "source": "pyproject.toml [tool.pytest]"})
                else:
                    inferred.append({"command": "pytest", "source": "pyproject.toml (Python project)"})
                if isinstance(tool_tbl, dict) and "ruff" in tool_tbl:
                    inferred.append({"command": "ruff check .", "source": "pyproject.toml [tool.ruff]"})
                if isinstance(tool_tbl, dict) and "mypy" in tool_tbl:
                    inferred.append({"command": "mypy .", "source": "pyproject.toml [tool.mypy]"})
            except Exception:
                inferred.append({"command": "pytest", "source": "pyproject.toml"})
        elif "requirements.txt" in manifests or "setup.py" in manifests:
            inferred.append({"command": "pytest", "source": "Python manifest"})

        pkg_json = self.project_root / "package.json"
        if "package.json" in manifests and pkg_json.is_file():
            try:
                pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                scripts = pkg.get("scripts", {})
                if isinstance(scripts, dict):
                    for key in ("test", "lint", "typecheck", "check"):
                        if key in scripts:
                            cmd = "npm test" if key == "test" else f"npm run {key}"
                            inferred.append({"command": cmd, "source": f"package.json scripts.{key}"})
            except Exception:
                pass

        if "Cargo.toml" in manifests:
            inferred.append({"command": "cargo test", "source": "Cargo.toml"})
        if "go.mod" in manifests:
            inferred.append({"command": "go test ./...", "source": "go.mod"})

        return inferred
