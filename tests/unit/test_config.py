"""Unit tests for configuration, presets, capabilities, and env precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemotron_agent.config import (
    DEFAULT_MODEL_PRESETS,
    ENV_BASE_URL,
    ENV_CONFIG_DIR,
    ENV_MAX_TURNS,
    ENV_MOCK,
    ENV_MODEL,
    ENV_PROVIDER,
    ConfigManager,
    ModelCapabilities,
    UserConfig,
    get_user_config_dir,
)
from nemotron_agent.logging_utils import ConfigError


def test_four_required_nemotron_presets_present_and_accurate() -> None:
    expected = [
        ("Nemotron 3 Super", "nvidia/nemotron-3-super-120b-a12b:free"),
        ("Nemotron 3.5 Lightning", "nvidia/nemotron-3.5-lightning:free"),
        ("Nemotron 3.5 Super VL", "nvidia/nemotron-3.5-super-vl:free"),
        ("Nemotron 3 Ultra", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    ]
    assert [(p.display_name, p.model_id) for p in DEFAULT_MODEL_PRESETS] == expected
    for preset in DEFAULT_MODEL_PRESETS:
        assert preset.provider == "openrouter"
        assert preset.is_free_endpoint is True
        assert len(preset.description) > 20


def test_presets_lookup_by_index_slug_name_and_id() -> None:
    cfg = UserConfig()
    assert cfg.find_preset("1") is not None
    assert cfg.find_preset("1").model_id == "nvidia/nemotron-3-super-120b-a12b:free"
    assert cfg.find_preset("nemotron-3.5-lightning").index == 2
    assert cfg.find_preset("Nemotron 3.5 Super VL").capabilities.supports_vision is True
    assert cfg.find_preset("Nemotron 3.5 Super VL").capabilities.supports_tool_calling is False
    assert cfg.find_preset("nvidia/nemotron-3-ultra-550b-a55b:free").index == 4
    assert cfg.find_preset("nonexistent-model") is None


def test_presets_can_be_updated_in_config_without_code_changes(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "user_cfg"
    mgr = ConfigManager(config_dir=cfg_dir)
    cfg = UserConfig()
    cfg.presets[1].model_id = "nvidia/nemotron-3.5-lightning-v2:free"
    cfg.capability_overrides["nvidia/nemotron-3.5-lightning-v2:free"] = ModelCapabilities(
        supports_tool_calling=False,
        max_context_chars=24000,
    )
    mgr.save_user_config(cfg)

    loaded = mgr.load_user_config()
    assert loaded is not None
    assert loaded.find_preset("2").model_id == "nvidia/nemotron-3.5-lightning-v2:free"
    caps = loaded.capabilities_for("nvidia/nemotron-3.5-lightning-v2:free")
    assert caps.supports_tool_calling is False
    assert caps.max_context_chars == 24000


def test_user_config_rejects_secret_fields(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "user_cfg"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(
        '{"provider": "openrouter", "api_key": "sk-or-v1-secretvalue123456"}',
        encoding="utf-8",
    )
    mgr = ConfigManager(config_dir=cfg_dir)
    with pytest.raises(ConfigError, match="Refusing to load or store secret fields"):
        mgr.load_user_config()


def test_refuses_saving_config_inside_project_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "my_repo"
    repo_root.mkdir()
    inside_dir = repo_root / ".nemotron-config"
    mgr = ConfigManager(config_dir=inside_dir, project_root=repo_root)
    with pytest.raises(ConfigError, match="Refusing to store user configuration"):
        mgr.save_user_config(UserConfig())


def test_configuration_precedence_cli_over_env_over_saved_over_default(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    saved = UserConfig(
        provider="openrouter",
        model_id="nvidia/nemotron-3-super-120b-a12b:free",
        base_url="https://openrouter.ai/api/v1",
        max_turns=6,
    )
    ConfigManager(config_dir=cfg_dir).save_user_config(saved)

    env = {
        ENV_CONFIG_DIR: str(cfg_dir),
        ENV_PROVIDER: "openrouter",
        ENV_MODEL: "Nemotron 3 Ultra",
        ENV_BASE_URL: "https://proxy.example.org/v1/",
        ENV_MAX_TURNS: "10",
    }
    mgr = ConfigManager(env=env)
    resolved_env = mgr.resolve()
    assert get_user_config_dir(env) == cfg_dir.resolve()
    assert resolved_env.model_id == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert resolved_env.base_url == "https://proxy.example.org/v1"
    assert resolved_env.max_turns == 10
    assert resolved_env.sources["model_id"] == f"env:{ENV_MODEL}"

    # CLI flags take highest precedence
    resolved_cli = mgr.resolve(
        cli_model="2",
        cli_base_url="https://cli.example.org/v1/",
        cli_max_turns=4,
    )
    assert resolved_cli.model_id == "nvidia/nemotron-3.5-lightning:free"
    assert resolved_cli.base_url == "https://cli.example.org/v1"
    assert resolved_cli.max_turns == 4
    assert resolved_cli.sources["model_id"] == "cli"


def test_mock_env_sets_mock_provider(tmp_path: Path) -> None:
    mgr = ConfigManager(config_dir=tmp_path / "cfg", env={ENV_MOCK: "true"})
    resolved = mgr.resolve()
    assert resolved.mock_mode is True
    assert resolved.provider == "mock"
