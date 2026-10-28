"""Unit tests for secret redaction, OS keyring storage, and 0600 local fallback."""

from __future__ import annotations

import io
import logging
import os
import stat
import sys
from pathlib import Path

import pytest

from nemotron_agent.config import ENV_API_KEY, ENV_OPENROUTER_API_KEY
from nemotron_agent.credentials import CredentialManager
from nemotron_agent.logging_utils import (
    GLOBAL_REDACTOR,
    REDACTED,
    AuthenticationError,
    FallbackConsentRequiredError,
    configure_logging,
    redact_text,
    redact_value,
    register_secret,
)


class FakeMemoryKeyring:
    """In-memory fake OS keyring backend."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.store: dict[tuple[str, str], str] = {}

    def backend_Description(self) -> tuple[bool, str, str | None]:
        if self.available:
            return True, "fake.MemoryKeyring", None
        return False, "fake.UnavailableKeyring", "No DBus session available"

    def get_password(self, service: str, username: str) -> str | None:
        if not self.available:
            raise RuntimeError("Keyring unavailable")
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        if not self.available:
            raise RuntimeError("Keyring unavailable")
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.store.pop((service, username), None)


@pytest.fixture(autouse=True)
def _clean_redactor() -> None:
    GLOBAL_REDACTOR.clear()
    yield
    GLOBAL_REDACTOR.clear()


def test_secret_redaction_covers_registered_and_heuristic_tokens() -> None:
    custom_secret = "my-custom-provider-token-998877"
    register_secret(custom_secret)

    raw = (
        f"Header Authorization: Bearer sk-or-v1-abcdef1234567890 "
        f"custom={custom_secret} "
        f'payload={{"api_key": "nvapi-xyz987654321"}} '
        f"OPENROUTER_API_KEY=plain_key_value_123"
    )
    scrubbed = redact_text(raw)
    assert "sk-or-v1-abcdef1234567890" not in scrubbed
    assert custom_secret not in scrubbed
    assert "nvapi-xyz987654321" not in scrubbed
    assert "plain_key_value_123" not in scrubbed
    assert REDACTED in scrubbed

    nested = {
        "Authorization": "Bearer secret-token-value-111",
        "nested": [{"api_key": custom_secret, "safe": "ok"}],
    }
    scrubbed_map = redact_value(nested)
    assert scrubbed_map["Authorization"] == REDACTED
    assert scrubbed_map["nested"][0]["api_key"] == REDACTED
    assert scrubbed_map["nested"][0]["safe"] == "ok"


def test_logger_and_exceptions_never_emit_secrets() -> None:
    secret = "sk-or-v1-supersecret99999999"
    register_secret(secret)

    buf = io.StringIO()
    logger = configure_logging(level=logging.INFO, stream=buf)
    logger.error("Failed calling endpoint with key %s", secret)

    err = AuthenticationError(f"Invalid token {secret} rejected by upstream")
    assert secret not in str(err)
    assert REDACTED in str(err)
    assert secret not in buf.getvalue()
    assert REDACTED in buf.getvalue()


def test_credential_manager_uses_os_keyring_when_available(tmp_path: Path) -> None:
    kr = FakeMemoryKeyring(available=True)
    mgr = CredentialManager(config_dir=tmp_path / "cfg", env={}, keyring_adapter=kr)

    saved = mgr.save_api_key("openrouter", "sk-or-v1-keyringsecret12345")
    assert saved.storage == "keyring"
    assert not (tmp_path / "cfg" / "credentials.json").exists()

    resolved = mgr.resolve_api_key("openrouter")
    assert resolved is not None
    assert resolved.api_key == "sk-or-v1-keyringsecret12345"
    assert "keyring" in resolved.source

    cleared = mgr.clear_credentials("openrouter")
    assert "keyring:openrouter" in cleared
    assert mgr.resolve_api_key("openrouter") is None


def test_credential_fallback_requires_explicit_consent_and_sets_0600(tmp_path: Path) -> None:
    kr = FakeMemoryKeyring(available=False)
    cfg_dir = tmp_path / "cfg"
    mgr = CredentialManager(config_dir=cfg_dir, env={}, keyring_adapter=kr)

    # Without consent, refuses to write local file
    with pytest.raises(FallbackConsentRequiredError):
        mgr.save_api_key(
            "openrouter",
            "sk-or-v1-localfilesecret123",
            confirm_fallback=lambda _msg: False,
        )
    assert not mgr.secrets_path.exists()

    # With explicit approval, writes 0600 file inside 0700 directory
    explanations: list[str] = []

    def approve(msg: str) -> bool:
        explanations.append(msg)
        return True

    saved = mgr.save_api_key(
        "openrouter",
        "sk-or-v1-localfilesecret123",
        confirm_fallback=approve,
    )
    assert len(explanations) == 1
    assert "not available" in explanations[0]
    assert saved.storage == "file"
    assert mgr.secrets_path.exists()

    if sys.platform != "win32":
        file_mode = stat.S_IMODE(mgr.secrets_path.stat().st_mode)
        dir_mode = stat.S_IMODE(cfg_dir.stat().st_mode)
        assert file_mode == 0o600
        assert dir_mode == 0o700

    resolved = mgr.resolve_api_key("openrouter")
    assert resolved is not None
    assert resolved.api_key == "sk-or-v1-localfilesecret123"

    # Simulate loosened permissions and ensure reader warns & repairs to 0600
    if sys.platform != "win32":
        os.chmod(mgr.secrets_path, 0o644)
        repaired = mgr.resolve_api_key("openrouter")
        assert repaired is not None
        assert repaired.warning is not None
        assert stat.S_IMODE(mgr.secrets_path.stat().st_mode) == 0o600


def test_credential_precedence_env_over_keyring(tmp_path: Path) -> None:
    kr = FakeMemoryKeyring(available=True)
    env = {
        ENV_OPENROUTER_API_KEY: "sk-or-v1-from-openrouter-env",
    }
    mgr = CredentialManager(config_dir=tmp_path / "cfg", env=env, keyring_adapter=kr)
    mgr.save_api_key("openrouter", "sk-or-v1-from-keyring-store")

    res1 = mgr.resolve_api_key("openrouter")
    assert res1 is not None
    assert res1.api_key == "sk-or-v1-from-openrouter-env"
    assert res1.source == f"env:{ENV_OPENROUTER_API_KEY}"

    env2 = {
        ENV_API_KEY: "sk-or-v1-from-generic-env",
        ENV_OPENROUTER_API_KEY: "sk-or-v1-from-openrouter-env",
    }
    mgr2 = CredentialManager(config_dir=tmp_path / "cfg", env=env2, keyring_adapter=kr)
    res2 = mgr2.resolve_api_key("openrouter")
    assert res2 is not None
    assert res2.api_key == "sk-or-v1-from-generic-env"
    assert res2.source == f"env:{ENV_API_KEY}"
