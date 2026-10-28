"""Secret-redacting logging utilities and structured error hierarchy."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any
from urllib.parse import quote

REDACTED = "[REDACTED]"

# Heuristic patterns for common API tokens, headers, and credential fields.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization headers (Bearer / Token / Basic)
    (
        re.compile(r"(?i)(\bAuthorization\s*[:=]\s*(?:Bearer|Token|Basic)\s+)([^\s\"',;]+)"),
        rf"\1{REDACTED}",
    ),
    (
        re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9._~+/=-]{8,})"),
        rf"\1{REDACTED}",
    ),
    # OpenRouter / OpenAI / NVIDIA style prefixes
    (
        re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{8,}\b"),
        REDACTED,
    ),
    (
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
        REDACTED,
    ),
    (
        re.compile(r"\bnvapi-[A-Za-z0-9_-]{8,}\b"),
        REDACTED,
    ),
    # JSON fields like "api_key": "..." or "apiKey": "..."
    (
        re.compile(
            r'(?i)("(?:api_key|apikey|access_token|secret_key|authorization)"\s*:\s*")([^"]+)(")'
        ),
        rf"\1{REDACTED}\3",
    ),
    # Env / query assignments like OPENROUTER_API_KEY=... or api_key=...
    (
        re.compile(
            r"(?i)(\b(?:NEMOTRON_AGENT_API_KEY|OPENROUTER_API_KEY|OPENAI_API_KEY|NVIDIA_API_KEY|api_key|apikey)\s*=\s*)([^\s&\"',;]+)"
        ),
        rf"\1{REDACTED}",
    ),
)

_SENSITIVE_DICT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "secret",
        "secret_key",
        "password",
        "token",
        "nemotron_agent_api_key",
        "openrouter_api_key",
        "openai_api_key",
    }
)


class SecretRedactor:
    """Registry and scrubber for known secrets and common token patterns."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._secrets: set[str] = set()

    def register(self, secret: str | None) -> None:
        """Register an exact secret value to be scrubbed from all output."""
        if not secret:
            return
        cleaned = secret.strip()
        if len(cleaned) < 4:
            return
        with self._lock:
            self._secrets.add(cleaned)
            encoded = quote(cleaned, safe="")
            if len(encoded) >= 4:
                self._secrets.add(encoded)

    def clear(self) -> None:
        """Clear registered runtime secrets (primarily for test isolation)."""
        with self._lock:
            self._secrets.clear()

    def redact_text(self, text: str, extra_secrets: tuple[str, ...] = ()) -> str:
        """Scrub known secret values and heuristic credential patterns from text."""
        if not text:
            return text
        result = text
        with self._lock:
            candidates = sorted(
                self._secrets.union(s for s in extra_secrets if s and len(s) >= 4),
                key=len,
                reverse=True,
            )
        for secret in candidates:
            if secret in result:
                result = result.replace(secret, REDACTED)
        for pattern, replacement in _SECRET_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def redact_value(self, value: Any, extra_secrets: tuple[str, ...] = ()) -> Any:
        """Recursively scrub strings, lists, tuples, and mappings."""
        if isinstance(value, str):
            return self.redact_text(value, extra_secrets=extra_secrets)
        if isinstance(value, dict):
            scrubbed: dict[Any, Any] = {}
            for k, v in value.items():
                key_norm = str(k).strip().lower().replace("-", "_")
                if key_norm in _SENSITIVE_DICT_KEYS and v is not None:
                    scrubbed[k] = REDACTED
                else:
                    scrubbed[k] = self.redact_value(v, extra_secrets=extra_secrets)
            return scrubbed
        if isinstance(value, list):
            return [self.redact_value(item, extra_secrets=extra_secrets) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_value(item, extra_secrets=extra_secrets) for item in value)
        return value


GLOBAL_REDACTOR = SecretRedactor()


def register_secret(secret: str | None) -> None:
    """Register a secret string with the process-wide redactor."""
    GLOBAL_REDACTOR.register(secret)


def redact_text(text: str, extra_secrets: tuple[str, ...] = ()) -> str:
    """Redact registered secrets and heuristic token patterns from a string."""
    return GLOBAL_REDACTOR.redact_text(text, extra_secrets=extra_secrets)


def redact_value(value: Any, extra_secrets: tuple[str, ...] = ()) -> Any:
    """Redact registered secrets and sensitive keys from arbitrary Python structures."""
    return GLOBAL_REDACTOR.redact_value(value, extra_secrets=extra_secrets)


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from log messages and arguments."""

    def __init__(self, redactor: SecretRedactor = GLOBAL_REDACTOR) -> None:
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        record.msg = self._redactor.redact_text(rendered)
        record.args = ()
        return True


class RedactingFormatter(logging.Formatter):
    """Logging formatter that also scrubs formatted exception tracebacks."""

    def __init__(
        self,
        fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt: str | None = "%Y-%m-%d %H:%M:%S",
        redactor: SecretRedactor = GLOBAL_REDACTOR,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return self._redactor.redact_text(formatted)


def configure_logging(level: int = logging.WARNING, stream: Any = None) -> logging.Logger:
    """Configure the package root logger with secret redaction enabled."""
    logger = logging.getLogger("nemotron_agent")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.addFilter(RedactingFilter())
    handler.setFormatter(RedactingFormatter())
    logger.addHandler(handler)

    # Silence verbose third-party HTTP request logging that could emit URLs or headers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logger


# ---------------------------------------------------------------------------
# Structured exception hierarchy (all automatically redact secrets)
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base error for nemotron-agent; automatically redacts secrets from message."""

    def __init__(self, message: str) -> None:
        clean = redact_text(message)
        super().__init__(clean)
        self.safe_message = clean

    def __str__(self) -> str:
        return redact_text(self.safe_message)


class ConfigError(AgentError):
    """Configuration validation, parsing, or storage error."""


class CredentialError(AgentError):
    """Credential retrieval, storage, or consent error."""


class FallbackConsentRequiredError(CredentialError):
    """Raised when OS keyring is unavailable and user has not approved file fallback."""


class ProviderError(AgentError):
    """Base error for model provider communication."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(ProviderError):
    """Invalid, missing, or unauthorized API credentials (HTTP 401/403)."""


class RateLimitError(ProviderError):
    """Provider rate limit exceeded (HTTP 429)."""


class EndpointUnavailableError(ProviderError):
    """Model endpoint unavailable, timed out, or unreachable (HTTP 404/5xx/network)."""


class ModelCapabilityError(ProviderError):
    """Selected model/provider rejected an unsupported API feature (e.g., tools)."""


class MalformedResponseError(ProviderError):
    """Provider or model returned malformed JSON or an invalid response envelope."""


class ToolValidationError(AgentError):
    """Model requested an unknown tool or provided schema-invalid arguments."""


class PermissionDeniedError(AgentError):
    """Action blocked by read-only policy, command allowlist, or user denial."""


class PathViolationError(PermissionDeniedError):
    """Attempted access outside project root or to an unapproved secret file."""


class ConflictError(AgentError):
    """File changed externally since read, or target patch snippet was not unique."""


class CommandExecutionError(AgentError):
    """Allowlisted command failed to start, timed out, or was cancelled."""
