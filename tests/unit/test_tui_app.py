"""Headless smoke tests for NemotronTuiApp construction and modal flows.

These never call app.run() (no terminal required): they build the application
shell, render header/status panes, and drive the palette / file-picker /
confirmation modals synchronously.
"""

from __future__ import annotations

from pathlib import Path

from nemotron_agent.agent import AgentSession
from nemotron_agent.config import ConfigManager
from nemotron_agent.credentials import CredentialManager
from nemotron_agent.providers.mock import MockProviderClient


class MemoryKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def backend_Description(self):
        return True, "test-keyring", None

    def get_password(self, service, username):
        return self.values.get((service, username))

    def set_password(self, service, username, password):
        self.values[(service, username)] = password

    def delete_password(self, service, username):
        self.values.pop((service, username), None)
from nemotron_agent.repl.session import ReplEnvironment
from nemotron_agent.tui.app import NemotronTuiApp
from nemotron_agent.tui.widgets import ConfirmModal, FilePickerModal, PaletteModal


def _make_env(tmp_path: Path) -> ReplEnvironment:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Demo\nproject introduction\n", encoding="utf-8")
    src = repo / "src"
    src.mkdir()
    (src / "foo.py").write_text("VALUE = 42\n", encoding="utf-8")
    (src / "bar.py").write_text("VALUE = 7\n", encoding="utf-8")

    cfg_dir = tmp_path / "cfg"
    manager = ConfigManager(config_dir=cfg_dir, env={}, project_root=repo)
    cfg = manager.resolve(cli_mock=True)
    keyring = MemoryKeyring()
    creds = CredentialManager(
        config_dir=cfg_dir, env={}, project_root=repo, keyring_adapter=keyring
    )
    outputs: list[str] = []
    session = AgentSession(
        project_root=repo,
        config=cfg,
        provider_client=MockProviderClient(),
        interactive=True,
        read_only=False,
        input_fn=lambda _prompt: "n",
        output_fn=outputs.append,
    )
    env = ReplEnvironment(
        session,
        config_manager=manager,
        credential_manager=creds,
        input_fn=lambda _prompt: "",
        secret_input_fn=lambda _prompt: "",
        output_fn=outputs.append,
        client_factory=lambda *args, **kwargs: MockProviderClient(),
    )
    env._outputs = outputs
    return env


def test_tui_constructs_headless_and_renders_panels(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    app = NemotronTuiApp(env)
    assert app.theme is env.theme
    assert app.busy is False
    assert app._modal is None

    header = app._render_header()
    status = app._render_status()
    joined_header = "".join(text for block in header for _, text in block)
    joined_status = "".join(text for line in status for _, text in line)
    assert env.session.config.model_id in joined_header
    assert str(env.session.project_root) in joined_header
    assert env.session.config.model_id in joined_status
    assert env.effort in joined_status


def test_tui_palette_filters_and_enters_command(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    app = NemotronTuiApp(env)
    app._open_palette()
    assert isinstance(app._modal, PaletteModal)
    modal = app._modal

    class Buffer:
        def __init__(self) -> None:
            self.text = "/"

    buf = Buffer()
    buf.text = "/models"
    app._palette_filter_changed(modal, buf)
    assert modal.matches  # "models" matches the /models command
    modal.on_enter()
    assert modal.resolved is True
    assert modal.chosen == "models"

    app._check_modal_resolved()
    assert app._modal is None
    assert app._input_buffer.text == "/models "


def test_tui_palette_escape_cancels(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    app = NemotronTuiApp(env)
    app._open_palette()
    modal = app._modal
    modal.on_cancel()
    app._check_modal_resolved()
    assert app._modal is None
    assert modal.resolved is True
    assert modal.chosen is None


def test_tui_file_picker_selects_existing_file(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    app = NemotronTuiApp(env)
    app._open_file_picker()
    modal: FilePickerModal = app._modal
    assert isinstance(modal, FilePickerModal)
    assert modal.root == str(env.session.project_root)

    # Typing a path that names an existing file selects and resolves it.
    modal.filter_text = "@src/foo.py"
    modal.on_enter()
    assert modal.resolved is True
    assert modal.chosen == "src/foo.py"

    app._check_modal_resolved()
    assert app._modal is None
    assert app._input_buffer.text == "@src/foo.py "


def test_tui_file_picker_enters_directory(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    app = NemotronTuiApp(env)
    app._open_file_picker()
    modal: FilePickerModal = app._modal
    modal.filter_text = "@src"
    modal.on_enter()
    assert modal.resolved is True
    assert modal.chosen == "bar.py"
    assert "src" in str(modal.cwd)


def test_tui_confirm_modal_yes_no(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    app = NemotronTuiApp(env)
    modal = ConfirmModal(title="Quit Nemotron?", text="Quit the application?")
    app._open_modal(modal)
    assert app._modal is modal
    modal.on_down()  # move selection to "No"
    modal.on_enter()
    assert modal.chosen is False
    assert modal.resolved is True

    modal2 = ConfirmModal(title="Quit Nemotron?", text="Quit the application?")
    modal2.on_enter()  # default is "Yes"
    assert modal2.chosen is True


def test_tui_color_command_changes_theme(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    from nemotron_agent.tui.theme import theme_for

    env.input_fn = lambda _prompt: "neon_green"
    assert env.theme.user_prompt == theme_for("orange").user_prompt
    env.registry.dispatch(env, "/color")
    assert env.theme == theme_for("neon_green")

    env.registry.dispatch(env, "/color blue")
    assert env.theme == theme_for("blue")


def test_tui_update_slash_command_runs(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    rc = env.registry.dispatch(env, "/update --check")
    assert rc in (True, False, None)
    assert env._outputs
    joined = "\n".join(env._outputs)
    assert "nemotron-agent" in joined and "Install method:" in joined


def test_tui_exit_dispatch_returns_exit_flag(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    assert env.registry.dispatch(env, "/exit") is True
