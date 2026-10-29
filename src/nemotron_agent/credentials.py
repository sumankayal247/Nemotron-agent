"""Secure credential resolution, OS keyring integration, and permissioned 0600 fallback."""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import keyring
from keyring.errors import KeyringError

from nemotron_agent.config import (
    ENV_API_KEY,
    ENV_OPENROUTER_API_KEY,
     _ensure_not_inside_repo,
    get_user_config_dir,
)
from nemotron_agent.logging_utils import (
    CredentialError,
    FallbackConsentRequiredError,
    register_secret,
)

KEYRING_SERVICE_NAME = "nemotron-agent"
CREDENTIALS_FILE_NAME = "credentials.json"


class KeyringAdapter(Protocol):
    """Injectable protocol over python-keyring for clean unit testing."""

    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...
    def backend_Description(self) -> tuple[bool, str, str | None]: ...


class SystemKeyringAdapter:
    """Adapter around the system `keyring` library with safe availability probing."""

    def backend_Description(self) -> tuple[bool, str, str | None]:
        try:
            kr = keyring.get_keyring()
        except Exception as exc:
            return False, "unavailable", f"Failed to initialize keyring: {exc}"

        backend_cls = f"{kr.__class__.__module__}.{kr.__class__.__qualname__}"
        lower_cls = backend_cls.lower()
        if any(bad in lower_cls for bad in ("fail.keyring", "null.keyring")):
            return False, backend_cls, "No active OS credential store backend was detected."

        try:
            # Lightweight read probe to verify DBus / Keychain / Credential Locker is reachable.
            kr.get_password(KEYRING_SERVICE_NAME, "__availability_probe__")
        except Exception as exc:
            return False, backend_cls, f"OS keyring backend ({backend_cls}) is not reachable: {exc}"

        return True, backend_cls, None

    def get_password(self, service: str, username: str) -> str | None:
        return keyring.get_password(service, username)

    def set_password(self, service: str, username: str, password: str) -> None:
        keyring.set_password(service, username, password)

    def delete_password(self, service: str, username: str) -> None:
        keyring.delete_password(service, username)


@dataclass(frozen=True)
class KeyringStatus:
    available: bool
    backend_name: str
    reason_unavailable: str | None = None


@dataclass(frozen=True)
class ResolvedCredential:
    provider: str
    api_key: str
    source: str  # e.g., "env:NEMOTRON_AGENT_API_KEY", "env:OPENROUTER_API_KEY", "keyring", "file"
    warning: str | None = None


@dataclass(frozen=True)
class SavedCredentialResult:
    provider: str
    storage: str  # "keyring" or "file"
    detail: str


class CredentialManager:
    """Manages provider API keys across env vars, OS keyring, and consented 0600 fallback."""

    def __init__(
        self,
        config_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
        keyring_adapter: KeyringAdapter | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._env = env if env is not None else os.environ
        self.config_dir = (
            config_dir.expanduser().resolve() if config_dir else get_user_config_dir(self._env)
        )
        self.secrets_path = self.config_dir / CREDENTIALS_FILE_NAME
        self._keyring = keyring_adapter if keyring_adapter is not None else SystemKeyringAdapter()
        self._project_root = project_root.resolve() if project_root else None
        self._session_keys: dict[str, str] = {}

    def set_project_root(self, project_root: Path) -> None:
        """Update the active repository used to guard credential storage."""
        root = project_root.expanduser().resolve()
        _ensure_not_inside_repo(self.config_dir, root)
        self._project_root = root

    def keyring_status(self) -> KeyringStatus:
        ok, name, reason = self._keyring.backend_Description()
        return KeyringStatus(available=ok, backend_name=name, reason_unavailable=reason)

    def fallback_explanation(self, reason: str | None = None) -> str:
        why = f" ({reason})" if reason else ""
        return (
            f"Operating system credential store is not available{why}.\n"
            f"As a fallback, nemotron-agent can save your API key in a user-local file:\n"
            f"  {self.secrets_path}\n"
            f"with owner-only permissions (chmod 0600 inside a 0700 directory). "
            f"Alternatively, you can decline persistent storage and use environment variables "
            f"({ENV_OPENROUTER_API_KEY} or {ENV_API_KEY}) or keep the key for this session only."
        )

    def set_session_key(self, provider: str, api_key: str) -> None:
        """Hold an API key in process memory only (not persisted to disk)."""
        cleaned = self._validate_key(api_key)
        norm_provider = provider.strip().lower()
        register_secret(cleaned)
        self._session_keys[norm_provider] = cleaned

    def resolve_api_key(self, provider: str) -> ResolvedCredential | None:
        """Resolve an API key with precedence: Session > Env > OS Keyring > Local 0600 File."""
        norm_provider = provider.strip().lower()
        if norm_provider == "mock":
            return None

        # 1. Explicit in-memory session key (e.g., when user declined disk persistence)
        if norm_provider in self._session_keys:
            key = self._session_keys[norm_provider]
            register_secret(key)
            return ResolvedCredential(provider=norm_provider, api_key=key, source="session-memory")

        # 2. Generic CLI env var
        generic_env = self._env.get(ENV_API_KEY, "").strip()
        if generic_env:
            register_secret(generic_env)
            return ResolvedCredential(
                provider=norm_provider,
                api_key=generic_env,
                source=f"env:{ENV_API_KEY}",
            )

        # 3. Provider-specific env var
        if norm_provider == "openrouter":
            or_env = self._env.get(ENV_OPENROUTER_API_KEY, "").strip()
            if or_env:
                register_secret(or_env)
                return ResolvedCredential(
                    provider=norm_provider,
                    api_key=or_env,
                    source=f"env:{ENV_OPENROUTER_API_KEY}",
                )

        # 4. OS Keyring
        status = self.keyring_status()
        if status.available:
            try:
                stored = self._keyring.get_password(KEYRING_SERVICE_NAME, norm_provider)
                if stored and stored.strip():
                    cleaned = stored.strip()
                    register_secret(cleaned)
                    return ResolvedCredential(
                        provider=norm_provider,
                        api_key=cleaned,
                        source=f"keyring ({status.backend_name})",
                    )
            except Exception:
                # Fall through to local secrets file if keyring read fails unexpectedly
                pass

        # 5. Local 0600 fallback file
        file_key, perm_warning = self._read_local_file_key(norm_provider)
        if file_key:
            register_secret(file_key)
            return ResolvedCredential(
                provider=norm_provider,
                api_key=file_key,
                source=f"file:{self.secrets_path}",
                warning=perm_warning,
            )

        return None

    def save_api_key(
        self,
        provider: str,
        api_key: str,
        *,
        allow_file_fallback: bool = False,
        confirm_fallback: Callable[[str], bool] | None = None,
    ) -> SavedCredentialResult:
        """Store API key in OS keyring, or in a 0600 local file only with explicit permission."""
        _ensure_not_inside_repo(self.config_dir, self._project_root)
        norm_provider = provider.strip().lower()
        cleaned = self._validate_key(api_key)
        register_secret(cleaned)

        status = self.keyring_status()
        keyring_failure_reason = status.reason_unavailable
        if status.available:
            try:
                self._keyring.set_password(KEYRING_SERVICE_NAME, norm_provider, cleaned)
                return SavedCredentialResult(
                    provider=norm_provider,
                    storage="keyring",
                    detail=f"OS credential store ({status.backend_name})",
                )
            except Exception as exc:
                keyring_failure_reason = f"Keyring write failed: {exc}"

        explanation = self.fallback_explanation(keyring_failure_reason)
        approved = allow_file_fallback
        if not approved and confirm_fallback is not None:
            approved = bool(confirm_fallback(explanation))

        if not approved:
            raise FallbackConsentRequiredError(
                f"{explanation}\nLocal file storage was not approved."
            )

        self._write_local_file_key(norm_provider, cleaned)
        return SavedCredentialResult(
            provider=norm_provider,
            storage="file",
            detail=f"{self.secrets_path} (permissions 0600)",
        )

    def clear_credentials(self, provider: str | None = None) -> list[str]:
        """Remove saved credentials from session memory, OS keyring, and local secrets file."""
        cleared: list[str] = []
        norm_provider = provider.strip().lower() if provider else None

        if norm_provider:
            if self._session_keys.pop(norm_provider, None) is not None:
                cleared.append(f"session-memory ({norm_provider})")
        elif self._session_keys:
            self._session_keys.clear()
            cleared.append("session-memory (all)")

        # Clear from OS keyring
        status = self.keyring_status()
        if status.available:
            targets = [norm_provider] if norm_provider else ["openrouter", "openai_compatible"]
            for target in targets:
                if not target:
                    continue
                try:
                    existing = self._keyring.get_password(KEYRING_SERVICE_NAME, target)
                    if existing:
                        self._keyring.delete_password(KEYRING_SERVICE_NAME, target)
                        cleared.append(f"keyring:{target}")
                except KeyringError:
                    pass
                except Exception:
                    pass

        # Clear from local fallback file
        if self.secrets_path.exists():
            try:
                if norm_provider is None:
                    self.secrets_path.unlink()
                    cleared.append(f"file:{self.secrets_path}")
                else:
                    data = self._load_secrets_dict()
                    if norm_provider in data:
                        del data[norm_provider]
                        if data:
                            self._atomic_write_secrets_dict(data)
                        else:
                            self.secrets_path.unlink()
                        cleared.append(f"file:{self.secrets_path} ({norm_provider})")
            except OSError as exc:
                raise CredentialError(
                    f"Failed to clear local credentials at {self.secrets_path}: {exc}"
                ) from exc

        return cleared

    @staticmethod
    def _validate_key(api_key: str) -> str:
        cleaned = api_key.strip()
        if not cleaned:
            raise CredentialError("API key cannot be empty.")
        if any(ch.isspace() for ch in cleaned):
            raise CredentialError("API key must not contain whitespace characters.")
        return cleaned

    def _load_secrets_dict(self) -> dict[str, str]:
        if not self.secrets_path.exists():
            return {}
        try:
            raw = json.loads(self.secrets_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CredentialError(
                f"Corrupted or unreadable local credentials file at {self.secrets_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise CredentialError(f"Invalid format in {self.secrets_path}; expected JSON object.")
        keys = raw.get("api_keys", raw)
        if not isinstance(keys, dict):
            return {}
        return {str(k): str(v) for k, v in keys.items() if isinstance(v, str) and v.strip()}

    def _read_local_file_key(self, provider: str) -> tuple[str | None, str | None]:
        if not self.secrets_path.exists():
            return None, None
        warning: str | None = None
        if sys.platform != "win32":
            try:
                mode = stat.S_IMODE(self.secrets_path.stat().st_mode)
                if mode & 0o077:
                    warning = (
                        f"Local credentials file {self.secrets_path} has overly broad permissions "
                        f"({oct(mode)}); tightening to 0o600."
                    )
                    os.chmod(self.secrets_path, 0o600)
            except OSError:
                pass
        data = self._load_secrets_dict()
        return data.get(provider), warning

    def _write_local_file_key(self, provider: str, api_key: str) -> None:
        data = self._load_secrets_dict() if self.secrets_path.exists() else {}
        data[provider] = api_key
        self._atomic_write_secrets_dict(data)

    def _atomic_write_secrets_dict(self, data: dict[str, str]) -> None:
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            if sys.platform != "win32":
                os.chmod(self.config_dir, 0o700)
            tmp_path = self.config_dir / f".{CREDENTIALS_FILE_NAME}.tmp"
            payload = json.dumps({"api_keys": data}, indent=2) + "\n"
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd = os.open(tmp_path, flags, 0o600)
            try:
                os.write(fd, payload.encode("utf-8"))
            finally:
                os.close(fd)
            if sys.platform != "win32":
                os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self.secrets_path)
            if sys.platform != "win32":
                os.chmod(self.secrets_path, 0o600)
        except OSError as exc:
            raise CredentialError(
                f"Failed to write local credentials file at {self.secrets_path}: {exc}"
            ) from exc
