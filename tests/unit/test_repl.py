"""Registry-driven interactive environment, persistence, completions, and safety."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
import tomllib
from pathlib import Path
from uuid import uuid4

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from nemotron_agent.agent import AgentSession
from nemotron_agent.cli import run_cli
from nemotron_agent.config import ConfigManager
from nemotron_agent.credentials import CredentialManager
from nemotron_agent.logging_utils import GLOBAL_REDACTOR, AgentError, register_secret
from nemotron_agent.providers.base import CompletionResponse, ToolCallRequest
from nemotron_agent.providers.mock import MockProviderClient
from nemotron_agent.repl.autocomplete import SlashCompleter, terminal_input
from nemotron_agent.repl.commands import SlashCommand
from nemotron_agent.repl.mentions import enrich_mentions
from nemotron_agent.repl.session import ReplEnvironment
from nemotron_agent.repl.storage import Plan


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


@pytest.fixture(autouse=True)
def clear_redactor():
    GLOBAL_REDACTOR.clear()
    yield
    GLOBAL_REDACTOR.clear()


def make_environment(
    tmp_path: Path,
    *,
    script=None,
    read_only=False,
    interactive=True,
    input_fn=None,
    cfg_dir=None,
    root=None,
    max_calls=None,
):
    repo = root or tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "README.md").write_text("# Demo\nproject introduction\n", encoding="utf-8")
    (repo / "src").mkdir(exist_ok=True)
    (repo / "src" / "foo.py").write_text("VALUE = 42\n", encoding="utf-8")
    config_dir = cfg_dir or tmp_path / "cfg"
    manager = ConfigManager(config_dir=config_dir, env={}, project_root=repo)
    cfg = manager.resolve(cli_mock=True)
    if max_calls is not None:
        cfg = cfg.model_copy(update={"max_tool_calls": max_calls})
    keyring = MemoryKeyring()
    creds = CredentialManager(
        config_dir=config_dir, env={}, project_root=repo, keyring_adapter=keyring
    )
    outputs: list[str] = []
    provider = MockProviderClient(script)
    reader = input_fn or (lambda _prompt: "n")
    session = AgentSession(
        project_root=repo,
        config=cfg,
        provider_client=provider,
        interactive=interactive,
        read_only=read_only,
        input_fn=reader,
        output_fn=outputs.append,
    )
    env = ReplEnvironment(
        session,
        config_manager=manager,
        credential_manager=creds,
        input_fn=reader,
        secret_input_fn=lambda _p: "sk-or-v1-testsecret123456",
        output_fn=outputs.append,
        client_factory=lambda provider, *_args: MockProviderClient(provider=provider),
    )
    return env, provider, outputs, keyring


def dispatch(env: ReplEnvironment, text: str) -> bool:
    return env.registry.dispatch(env, text)


def test_entry_points_and_default_cli_repl(tmp_path, monkeypatch):
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert (
        project["scripts"]["NA"]
        == project["scripts"]["nemotron-agent"]
        == "nemotron_agent.cli:main"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    for executable in ("NA", "nemotron-agent"):
        inputs = iter(["/exit"])
        output: list[str] = []
        monkeypatch.setattr(sys, "argv", [executable])
        rc = run_cli(
            None,
            config_dir=tmp_path / f"cfg-{executable}",
            env={"NEMOTRON_AGENT_MOCK": "1"},
            input_fn=lambda _: next(inputs),
            output_fn=output.append,
        )
        assert rc == 0
        assert "Nemotron Agent" in "\n".join(output)
        assert "Goodbye" in "\n".join(output)


def test_registry_is_single_source_for_help_execution_aliases_and_validation(tmp_path):
    env, provider, out, _ = make_environment(tmp_path)
    required = {
        "help",
        "clear",
        "resume",
        "agents",
        "skills",
        "artifacts",
        "models",
        "efforts",
        "plan",
        "learn",
        "config",
        "auth",
        "setup",
        "ping",
        "status",
        "context",
        "history",
        "retry",
        "stop",
        "version",
        "exit",
        "search",
        "read",
        "files",
        "tools",
    }
    assert required <= {command.name for command in env.registry.commands()}
    assert env.registry.get("quit") is env.registry.get("exit")
    assert env.registry.get("model") is env.registry.get("models")
    dispatch(env, "/help")
    help_output = "\n".join(out)
    for name in required:
        assert f"/{name}" in help_output
        assert env.registry.get(name).description in help_output
    out.clear()
    dispatch(env, "/help learn")
    assert "Usage: /learn" in "\n".join(out)
    assert "explicitly" in "\n".join(out).lower()
    dispatch(env, "/version extra")
    dispatch(env, "/modles")
    assert "Usage: /version" in "\n".join(out)
    assert "/models" in "\n".join(out)
    assert not provider.requests
    assert dispatch(env, "/quit") is True

    # A dynamically registered command immediately participates everywhere.
    seen = []
    env.registry.register(
        SlashCommand(
            "dynamic",
            "Dynamically installed",
            "/dynamic",
            lambda _env, _inv: seen.append(True),
        )
    )
    assert "/dynamic" in env.registry.help_text()
    assert "/dynamic" in [
        c.text
        for c in SlashCompleter(env).get_completions(Document("/"), CompleteEvent())
    ]
    dispatch(env, "/dynamic")
    assert seen == [True]


def test_slash_completion_and_actual_tab_cycling(tmp_path):
    env, _, _, _ = make_environment(tmp_path)
    completer = SlashCompleter(env)

    def candidates(text):
        return [
            c.text for c in completer.get_completions(Document(text), CompleteEvent())
        ]

    assert "/help" in candidates("/") and "/models" in candidates("/")
    assert candidates("/mo")[0] == "/models"
    assert {"/models", "/model"} <= set(candidates("/m"))
    assert "use" in candidates("/models ")
    assert "2" in candidates("/models use ")
    assert "code-review" in candidates("/skills use ")
    assert candidates("/efforts ") == list(
        env.registry.get("efforts").arguments.first_choices
    )
    assert not candidates("normal /mo")
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        reader = terminal_input(env)

        def send():
            pipe.send_text("/mo")
            time.sleep(0.15)
            pipe.send_text("\t")
            time.sleep(0.15)
            pipe.send_text("\n")

        thread = threading.Thread(target=send)
        thread.start()
        assert reader("NA > ") == "/models"
        thread.join()

        def send_two_tabs():
            for text in ("/m", "\t", "\t", "\n"):
                pipe.send_text(text)
                time.sleep(0.15)

        thread = threading.Thread(target=send_two_tabs)
        thread.start()
        assert reader("NA > ") == "/model"
        thread.join()


def test_normal_text_is_sent_to_model_but_slash_is_local(tmp_path):
    env, provider, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="answer")]
    )
    dispatch(env, "/help")
    dispatch(env, "/unknown-command")
    assert not provider.requests
    env.execute_prompt("fix the authentication bug")
    assert provider.requests[0].messages[-1].content == "fix the authentication bug"
    assert "answer" in "\n".join(out)


def test_mentions_are_confined_framed_bounded_and_have_selection(tmp_path):
    env, provider, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="Read both.")]
    )
    env.execute_prompt("Explain @README.md and @src/foo.py")
    sent = provider.requests[0].messages[-1].content
    assert "File: README.md" in sent and "project introduction" in sent
    assert "File: src/foo.py" in sent and "VALUE = 42" in sent
    assert '<untrusted_repo_data tool="read_file">' in sent
    assert "Referenced file: src/foo.py" in "\n".join(out)
    assert env.last_request is not None
    punctuated = enrich_mentions(
        env.session,
        "What about @README.md?",
        input_fn=lambda _: "",
        output_fn=out.append,
        max_tool_calls=5,
    )
    assert "File: README.md" in punctuated.prompt
    parenthesized = enrich_mentions(
        env.session,
        "Explain (@README.md).",
        input_fn=lambda _: "",
        output_fn=out.append,
        max_tool_calls=5,
    )
    assert "File: README.md" in parenthesized.prompt
    email = enrich_mentions(
        env.session,
        "Contact user@example.com",
        input_fn=lambda _: pytest.fail("email is not a reference"),
        output_fn=out.append,
        max_tool_calls=5,
    )
    assert email.tool_calls_used == 0

    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE SECRET", encoding="utf-8")
    (env.session.project_root / "link.txt").symlink_to(outside)
    (env.session.project_root / ".env").write_text("password=SECRET", encoding="utf-8")
    (env.session.project_root / "blob.bin").write_bytes(b"\0\1")
    for mention in ("@../outside.txt", "@link.txt", "@.env", "@blob.bin"):
        before = len(provider.requests)
        with pytest.raises(AgentError):
            env.execute_prompt(f"Explain {mention}")
        assert len(provider.requests) == before

    other = env.session.project_root / "other"
    other.mkdir()
    (other / "foo.py").write_text("VALUE = 99\n", encoding="utf-8")
    selected = enrich_mentions(
        env.session,
        "explain @foo.py",
        input_fn=lambda _: "2",
        output_fn=out.append,
        max_tool_calls=5,
    )
    assert "Multiple files match" in "\n".join(out)
    assert "src/foo.py" in selected.prompt
    (env.session.project_root / "foo.py").write_text("ROOT = True\n", encoding="utf-8")
    root_choice = enrich_mentions(
        env.session,
        "explain @foo.py",
        input_fn=lambda _: "1",
        output_fn=out.append,
        max_tool_calls=5,
    )
    assert "ROOT = True" in root_choice.prompt
    explicit = enrich_mentions(
        env.session,
        "explain @./foo.py",
        input_fn=lambda _: pytest.fail("must not ask"),
        output_fn=out.append,
        max_tool_calls=5,
    )
    assert "ROOT = True" in explicit.prompt
    with pytest.raises(AgentError, match="cancelled"):
        enrich_mentions(
            env.session,
            "explain @foo.py",
            input_fn=lambda _: "",
            output_fn=out.append,
            max_tool_calls=5,
        )


def test_mentions_consume_the_same_tool_budget(tmp_path):
    attack = CompletionResponse(
        content="trying",
        tool_calls=[
            ToolCallRequest(
                id="evil",
                name="edit_file",
                arguments_raw={
                    "path": "new.py",
                    "new_text": "bad",
                    "old_text": "",
                    "create_if_missing": True,
                },
            ),
        ],
    )
    env, provider, out, _ = make_environment(
        tmp_path, script=[attack], max_calls=1, input_fn=lambda _: "y"
    )
    result = env.execute_prompt("Explain @README.md")
    assert result.status == "tool_limit"
    assert result.tool_calls_used == 1
    assert provider.requests[0].tools == []
    assert not (env.session.project_root / "new.py").exists()


def test_clear_preserves_saved_sessions_memory_plan_and_settings_then_resume(tmp_path):
    env, _, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="answer")]
    )
    dispatch(env, "/learn add Use pytest for checks")
    dispatch(env, "/skills use code-review")
    dispatch(env, "/efforts low")
    env.execute_prompt("explain @src/foo.py")
    env.plan = Plan(goal="test", steps=["Inspect", "Verify"], approved=True)
    env.save()
    old_id = env.session_id
    old_model = env.session.config.model_id
    dispatch(env, "/clear")
    assert len(env.session.messages) == 1 and env.session.context_tracker.spans() == []
    assert env.plan and env.plan.approved
    assert env.active_skills == ["code-review"] and env.effort == "low"
    assert env.session.config.model_id == old_model
    assert (
        env.memory.for_repo(env.session.project_root)[0].text == "Use pytest for checks"
    )
    assert (env.sessions.directory / f"{old_id}.json").is_file()
    assert "Conversation context cleared." in "\n".join(out)
    dispatch(env, "/resume")
    dispatch(env, "/resume 1")
    assert env.session_id == old_id
    assert len(env.session.messages) > 1 and env.plan.approved
    assert env.session.context_tracker.get_recorded_hash("src/foo.py") is None
    # A restored reference must not serve as a fresh conflict-check baseline.
    result = env.session.tool_registry.execute_call(
        ToolCallRequest(
            id="edit",
            name="edit_file",
            arguments_raw={"path": "src/foo.py", "old_text": "42", "new_text": "99"},
        )
    )
    assert not result.ok and "before reading" in result.raw_output
    assert (env.session.project_root / "src" / "foo.py").read_text() == "VALUE = 42\n"


def test_session_store_is_private_redacted_and_skips_corruption(tmp_path):
    env, _, _, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="done")]
    )
    secret = "sk-or-v1-testsecret123456"
    register_secret(secret)
    env.execute_prompt(f"Explain this token {secret}")
    path = env.sessions.directory / f"{env.session_id}.json"
    data = path.read_text(encoding="utf-8")
    assert secret not in data and "[REDACTED]" in data
    assert str(env.session.project_root) not in str(path.resolve())
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    (env.sessions.directory / f"{uuid4().hex}.json").write_text(
        "{bad", encoding="utf-8"
    )
    assert len(env.sessions.list()) == 1
    assert env.session.project_root.is_dir()


def test_plan_is_read_only_even_if_provider_requests_mutating_tools(tmp_path):
    hostile = CompletionResponse(
        content="planning",
        tool_calls=[
            ToolCallRequest(
                id="1",
                name="edit_file",
                arguments_raw={
                    "path": "bad.py",
                    "old_text": "",
                    "new_text": "BAD",
                    "create_if_missing": True,
                },
            ),
            ToolCallRequest(
                id="2",
                name="run_command",
                arguments_raw={"argv": ["pytest"], "reason": "run tests"},
            ),
        ],
    )
    env, provider, out, _ = make_environment(
        tmp_path,
        script=[
            hostile,
            CompletionResponse(content="1. Inspect files\n2. Run tests after approval"),
        ],
        input_fn=lambda _: "y",
    )
    dispatch(env, "/plan create repair auth")
    assert env.plan is not None and env.plan.steps == [
        "Inspect files",
        "Run tests after approval",
    ]
    assert not (env.session.project_root / "bad.py").exists()
    assert not env.session.command_runner.runs
    assert all(
        tool.name not in {"edit_file", "run_command"}
        for tool in provider.requests[0].tools
    )
    assert not env.session.permissions.policy.read_only  # restored after planning
    dispatch(env, "/plan approve")
    assert env.plan.approved and not env.session.command_runner.runs
    dispatch(env, "/plan clear")
    assert env.plan is None
    assert "No changes or commands were executed" in "\n".join(out)


def test_agents_skills_and_declarative_workflows_cannot_grant_tools(tmp_path):
    cfg = tmp_path / "cfg"
    (cfg / "skills").mkdir(parents=True)
    (cfg / "skills" / "security.json").write_text(
        json.dumps(
            {
                "name": "security-review",
                "description": "Review the security boundary",
                "instructions": "Inspect input validation.",
                "context": "Check request parsing.",
                "tools": ["read_file"],
                "workflows": {"audit": "Check for input validation problems."},
                "commands": {"security-audit": "audit"},
            }
        ),
        encoding="utf-8",
    )
    env, provider, out, _ = make_environment(
        tmp_path, cfg_dir=cfg, script=[CompletionResponse(content="review complete")]
    )
    dispatch(env, "/agents list")
    dispatch(env, "/agents info reviewer")
    dispatch(env, "/agents use reviewer")
    assert (
        env.active_agent == "reviewer"
        and "Prioritize actionable" in env.session.messages[0].content
    )
    dispatch(env, "/skills info security-review")
    dispatch(env, "/skills use security-review")
    assert "security-review" in env.active_skills
    assert "/security-audit" in env.registry.help_text()
    assert "/security-audit" in [
        c.text
        for c in SlashCompleter(env).get_completions(
            Document("/security"), CompleteEvent()
        )
    ]
    dispatch(env, "/security-audit inspect parser")
    assert (
        "Check for input validation problems"
        in provider.requests[0].messages[-1].content
    )
    assert "untrusted_skill_data" in provider.requests[0].messages[0].content
    assert "review complete" in "\n".join(out)

    bad = env.skills.get("security-review").model_copy(
        update={"name": "bad-skill", "tools": ["bash"], "commands": {}}
    )
    env.skills.register(bad)
    dispatch(env, "/skills use bad-skill")
    assert "bad-skill" not in env.active_skills
    assert "unavailable tools" in "\n".join(out)


def test_learn_is_explicit_repo_scoped_redacted_and_survives_clear(tmp_path):
    env, provider, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="ok")]
    )
    env.execute_prompt("This conversation should not become persistent memory")
    assert env.memory.for_repo(env.session.project_root) == []
    dispatch(env, "/learn add Use pytest for this project")
    dispatch(env, "/learn search pytest")
    assert "Use pytest" in "\n".join(out)
    dispatch(env, "/learn add password=secretvalue123")
    assert len(env.memory.for_repo(env.session.project_root)) == 1
    assert "Refusing to store" in "\n".join(out)
    dispatch(env, "/clear")
    assert "Use pytest" in env.session.messages[0].content
    assert "untrusted_memory_data" in env.session.messages[0].content
    other = tmp_path / "another project"
    other.mkdir()
    dispatch(env, f"/cd {other}")
    assert "Use pytest" not in env.session.messages[0].content
    assert not env.memory.for_repo(other)
    assert "Switched active local codebase" in "\n".join(out)


def test_models_effort_and_read_only_locks_affect_real_requests(tmp_path):
    env, provider, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="ok")]
    )
    dispatch(env, "/models current")
    dispatch(env, "/models use 1")
    assert env.session.config.model_id.endswith("nemotron-3.5-lightning:free")
    dispatch(env, "/efforts low")
    assert env.task_limits() == (3, 4)
    env.execute_prompt("hello")
    assert provider.requests[0].model_id == env.session.config.model_id
    assert "reasoning_effort" not in provider.requests[0].model_dump()
    assert "no reasoning parameter is sent" in "\n".join(out)

    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    locked, _, locked_out, _ = make_environment(locked_dir, read_only=True)
    dispatch(locked, "/readonly off")
    assert locked.session.permissions.policy.read_only
    assert "cannot be disabled" in "\n".join(locked_out)


def test_repo_commands_and_security_gates_do_not_bypass_agent_tools(tmp_path):
    hostile = CompletionResponse(
        content="try unsafe tools",
        tool_calls=[
            ToolCallRequest(
                id="1",
                name="edit_file",
                arguments_raw={
                    "path": "new.txt",
                    "old_text": "",
                    "new_text": "bad",
                    "create_if_missing": True,
                },
            ),
            ToolCallRequest(
                id="2",
                name="run_command",
                arguments_raw={"argv": ["rm", "-rf", "."], "reason": "bad command"},
            ),
            ToolCallRequest(
                id="3", name="read_file", arguments_raw={"path": "../outside.txt"}
            ),
        ],
    )
    env, provider, out, _ = make_environment(
        tmp_path,
        script=[hostile, CompletionResponse(content="finished")],
        input_fn=lambda _: "n",
    )
    dispatch(env, "/tools")
    dispatch(env, "/files")
    dispatch(env, "/search VALUE")
    dispatch(env, "/read src/foo.py")
    dispatch(env, "/read ../outside.txt")
    assert "VALUE = 42" in "\n".join(out)
    assert "outside the selected project root" in "\n".join(out)
    env.execute_prompt("try the tools")
    assert not (env.session.project_root / "new.txt").exists()
    assert not env.session.command_runner.runs
    assert any(
        entry.action == "edit_file" and not entry.approved
        for entry in env.session.permissions.audit_log
    )
    assert "disallowed" in "\n".join(out)

    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    locked, _, _, _ = make_environment(
        readonly_dir,
        script=[hostile, CompletionResponse(content="finished")],
        read_only=True,
        input_fn=lambda _: "y",
    )
    locked.execute_prompt("try the tools")
    assert not (locked.session.project_root / "new.txt").exists()
    assert not locked.session.command_runner.runs

    noninteractive_dir = tmp_path / "noninteractive"
    noninteractive_dir.mkdir()
    noninteractive, _, non_out, _ = make_environment(
        noninteractive_dir,
        script=[hostile, CompletionResponse(content="done")],
        interactive=False,
        input_fn=lambda _: "y",
    )
    noninteractive.execute_prompt("try the tools")
    assert not (noninteractive.session.project_root / "new.txt").exists()
    dispatch(noninteractive, "/setup")
    dispatch(noninteractive, "/auth set-key")
    assert "requires interactive" in "\n".join(non_out).lower()


def test_artifacts_are_only_agent_created_and_open_through_read_tool(tmp_path):
    created = CompletionResponse(
        content="create",
        tool_calls=[
            ToolCallRequest(
                id="1",
                name="edit_file",
                arguments_raw={
                    "path": "report.txt",
                    "old_text": "",
                    "new_text": "Generated report\n",
                    "create_if_missing": True,
                },
            )
        ],
    )
    env, _, out, _ = make_environment(
        tmp_path,
        script=[created, CompletionResponse(content="done")],
        input_fn=lambda _: "y",
    )
    env.execute_prompt("create report")
    dispatch(env, "/artifacts list")
    dispatch(env, "/artifacts path")
    dispatch(env, "/artifacts open report.txt")
    assert [artifact.path for artifact in env.artifacts] == ["report.txt"]
    assert "Generated report" in "\n".join(out)
    dispatch(env, "/artifacts open README.md")
    assert "not found or ambiguous" in "\n".join(out)
    (env.session.project_root / "report.txt").unlink()
    (env.session.project_root / "report.txt").symlink_to(tmp_path / "outside-secret")
    dispatch(env, "/artifacts open report.txt")
    assert "escapes the repository" in "\n".join(out)


def test_retry_stop_history_context_and_version(tmp_path):
    env, provider, out, _ = make_environment(
        tmp_path,
        script=[
            CompletionResponse(content="first"),
            CompletionResponse(content="second"),
        ],
    )
    dispatch(env, "/retry")
    dispatch(env, "/stop")
    dispatch(env, "/version")
    env.execute_prompt("hello")
    dispatch(env, "/history")
    dispatch(env, "/context")
    dispatch(env, "/status")
    dispatch(env, "/retry")
    assert len(provider.requests) == 2
    assert [m.content for m in env.session.messages if m.role == "user"] == ["hello"]
    rendered = "\n".join(out)
    assert "Ctrl+C" in rendered and "nemotron-agent 0.1.0" in rendered
    assert "Session ID" in rendered and "Strict read-only" in rendered
    assert "user: hello" in rendered and "second" in rendered

    def interrupt(_req):
        raise KeyboardInterrupt()

    other = tmp_path / "interrupt"
    other.mkdir()
    interrupted, _, interrupt_out, _ = make_environment(other, script=[interrupt])
    result = interrupted.execute_prompt("do something")
    assert result.status == "cancelled" and len(interrupted.session.messages) == 1
    dispatch(interrupted, "/retry")
    assert "Cannot automatically retry" in "\n".join(interrupt_out)


def test_config_auth_setup_and_ping_use_existing_implementations(tmp_path):
    answers = iter(["custom", "mock", "", "my-local-model"])
    env, _, out, keyring = make_environment(tmp_path, input_fn=lambda _: next(answers))
    dispatch(env, "/config show")
    dispatch(env, "/config presets")
    dispatch(env, "/config set-model 1 --skip-check")
    assert env.session.config.model_id.endswith("lightning:free")
    dispatch(env, "/ping")
    dispatch(env, "/auth set-key --provider openrouter --skip-check")
    assert any(key[1] == "openrouter" for key in keyring.values)
    dispatch(env, "/auth clear --provider openrouter")
    assert not keyring.values
    dispatch(env, "/setup")
    assert (
        env.session.config.provider == "mock"
        and env.session.config.model_id == "my-local-model"
    )
    text = "\n".join(out)
    assert "Active Nemotron Agent Configuration" in text
    assert "Mock provider ready" in text
    assert "Setup complete" in text
    assert "sk-or-v1-testsecret123456" not in text


def test_private_storage_cannot_be_symlinked_into_project(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "sessions").symlink_to(repo, target_is_directory=True)
    manager = ConfigManager(config_dir=cfg, env={}, project_root=repo)
    creds = CredentialManager(
        config_dir=cfg, env={}, project_root=repo, keyring_adapter=MemoryKeyring()
    )
    session = AgentSession(
        project_root=repo,
        config=manager.resolve(cli_mock=True),
        provider_client=MockProviderClient(),
    )
    with pytest.raises(AgentError, match="inside project directory"):
        ReplEnvironment(
            session,
            config_manager=manager,
            credential_manager=creds,
            input_fn=input,
            secret_input_fn=input,
            output_fn=lambda _: None,
            client_factory=lambda *_: MockProviderClient(),
        )


def test_stop_cooperatively_cancels_before_executing_returned_tools(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def slow_response(_request):
        started.set()
        assert release.wait(2)
        return CompletionResponse(
            content="late",
            tool_calls=[
                ToolCallRequest(
                    id="late",
                    name="edit_file",
                    arguments_raw={
                        "path": "late.txt",
                        "old_text": "",
                        "new_text": "should not exist",
                        "create_if_missing": True,
                    },
                )
            ],
        )

    env, _, out, _ = make_environment(
        tmp_path, script=[slow_response], input_fn=lambda _: "y"
    )
    results = []
    worker = threading.Thread(
        target=lambda: results.append(env.execute_prompt("do work"))
    )
    worker.start()
    assert started.wait(2)
    dispatch(env, "/stop")
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert results[0].status == "cancelled"
    assert not (env.session.project_root / "late.txt").exists()
    assert len(env.session.messages) == 1
    assert "Cancellation requested" in "\n".join(out)


def test_resume_does_not_trust_snapshot_endpoint_or_restore_secret_approval(tmp_path):
    env, _, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="done")]
    )
    env.execute_prompt("hello")
    path = env.sessions.directory / f"{env.session_id}.json"
    payload = json.loads(path.read_text())
    payload["base_url"] = "https://attacker.example/v1"
    path.write_text(json.dumps(payload))
    env._new_identity()
    dispatch(env, "/resume 1")
    assert "differs from the configured provider endpoint" in "\n".join(out)
    assert env.session.config.base_url == "mock://local"


def test_free_model_switch_blocks_restored_sensitive_file_context(tmp_path):
    env, _, out, _ = make_environment(tmp_path)
    # Simulate a previously approved paid-model read, then a restored session
    # where no approval cache survives but the conversation still has the data.
    env.session.config = env.session.config.model_copy(
        update={"model_id": "paid-model"}
    )
    env.session.permissions.policy.is_free_endpoint = False
    from nemotron_agent.repo.context import ContextSpan

    env.session.context_tracker.restore_summary([ContextSpan(".env", 1, 1)], 20)
    dispatch(env, "/models use 2")
    assert env.session.config.model_id == "paid-model"
    assert "Clear current context" in "\n".join(out)
    dispatch(env, "/clear")
    dispatch(env, "/models use 2")
    assert env.session.config.is_free_endpoint
    assert env.session.permissions.policy.is_free_endpoint


def test_copy_gracefully_handles_missing_terminal_clipboard(tmp_path, monkeypatch):
    env, _, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="answer")]
    )
    dispatch(env, "/copy")
    env.execute_prompt("hello")
    dispatch(env, "/copy")
    assert "No assistant answer" in "\n".join(out)
    assert "Clipboard unavailable" in "\n".join(out)


def test_first_run_opens_repl_before_setup_and_defers_wizard_until_prompt(tmp_path):
    repo = tmp_path / "fresh"
    repo.mkdir()
    output: list[str] = []
    commands = iter(["/help", "/exit"])
    rc = run_cli(
        ["--path", str(repo)],
        config_dir=tmp_path / "cfg",
        env={},
        input_fn=lambda _prompt: next(commands),
        output_fn=output.append,
        secret_input_fn=lambda _prompt: pytest.fail("no key prompt expected"),
    )
    assert rc == 0
    assert "Nemotron Agent Commands" in "\n".join(output)
    assert "First-Run Setup" not in "\n".join(output)

    output.clear()
    # The unchanged setup wizard still runs automatically when an unconfigured
    # first-run session actually attempts a model request.
    commands = iter(["explain the repo", "custom", "mock", "", "local-model", "/exit"])
    rc = run_cli(
        ["--path", str(repo)],
        config_dir=tmp_path / "cfg2",
        env={},
        input_fn=lambda _prompt: next(commands),
        output_fn=output.append,
        client_factory=lambda provider, *_: MockProviderClient(provider=provider),
    )
    assert rc == 0
    assert "First-Run Setup" in "\n".join(output)
    assert "[MOCK MODE]" in "\n".join(output)


def test_directory_reference_and_ignored_files(tmp_path):
    env, provider, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="done")]
    )
    (env.session.project_root / ".gitignore").write_text(
        "ignored.txt\n", encoding="utf-8"
    )
    (env.session.project_root / "ignored.txt").write_text("hidden", encoding="utf-8")
    # Refresh the index after adding the ignore rule.
    env.session.switch_project_root(env.session.project_root)
    env.execute_prompt("Explain @src/")
    assert "src/foo.py" in provider.requests[0].messages[-1].content
    assert (
        '<untrusted_repo_data tool="list_files">'
        in provider.requests[0].messages[-1].content
    )
    with pytest.raises(AgentError, match="excluded"):
        env.execute_prompt("Explain @ignored.txt")
    assert "Referenced directory: src" in "\n".join(out)


def test_empty_existing_snapshot_is_updated_after_plan_clear(tmp_path):
    env, _, _, _ = make_environment(tmp_path)
    env.plan = Plan(goal="goal", steps=["Inspect"])
    env.save()
    path = env.sessions.directory / f"{env.session_id}.json"
    assert json.loads(path.read_text())["plan"] is not None
    dispatch(env, "/plan clear")
    assert json.loads(path.read_text())["plan"] is None


def test_retry_refuses_after_agent_created_files(tmp_path):
    creation = CompletionResponse(
        tool_calls=[
            ToolCallRequest(
                id="edit",
                name="edit_file",
                arguments_raw={
                    "path": "created.txt",
                    "old_text": "",
                    "new_text": "created",
                    "create_if_missing": True,
                },
            )
        ]
    )
    env, provider, out, _ = make_environment(
        tmp_path,
        script=[creation, CompletionResponse(content="done")],
        input_fn=lambda _: "y",
    )
    env.execute_prompt("create a file")
    before = len(provider.requests)
    dispatch(env, "/retry")
    assert len(provider.requests) == before
    assert "Cannot automatically retry" in "\n".join(out)


def test_untrusted_terminal_control_sequences_are_not_rendered(tmp_path):
    env, _, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="safe\x1b]52;c;ZXZpbA==\x07text")]
    )
    env.execute_prompt("hello")
    assert "\x1b" not in "\n".join(out)
    assert "ZXZpbA==" not in "\n".join(out)


def test_plain_plan_uses_last_prompt_and_retry_records_result(tmp_path):
    env, _, _, _ = make_environment(
        tmp_path,
        script=[
            CompletionResponse(content="initial answer"),
            AgentError("temporary provider failure"),
            CompletionResponse(content="1. Inspect the code\n2. Verify the change"),
        ],
    )
    env.execute_prompt("investigate login")
    dispatch(env, "/plan")
    assert env.plan is None
    dispatch(env, "/retry")
    assert env.plan is not None
    assert env.plan.goal == "investigate login"
    assert env.plan.steps == ["Inspect the code", "Verify the change"]


def test_resume_across_repositories_does_not_restore_scoped_approvals(tmp_path):
    env, _, _, _ = make_environment(
        tmp_path,
        script=[CompletionResponse(content="project A")],
        input_fn=lambda _: "y",
    )
    original = env.session.project_root
    env.execute_prompt("first project")
    old_id = env.session_id
    other = tmp_path / "other"
    other.mkdir()
    (other / ".env").write_text("a sensitive file", encoding="utf-8")
    dispatch(env, f"/cd {other}")
    env.session.permissions.policy.is_free_endpoint = False
    env.session.path_guard.resolve_path(".env")
    assert env.session.permissions.has_approved_secret_reads
    # Clear the local secret read before returning to a free endpoint.
    dispatch(env, "/clear")
    dispatch(env, "/resume 1")
    assert env.session.project_root == original
    assert env.session_id == old_id
    assert not env.session.permissions.has_approved_secret_reads
    assert env.session.config.is_free_endpoint


def test_stop_after_blocked_tool_does_not_start_synthesis(tmp_path):
    from dataclasses import replace

    started = threading.Event()
    release = threading.Event()

    def blocked_tool(_args):
        started.set()
        assert release.wait(2)
        return "safe read finished"

    response = CompletionResponse(
        tool_calls=[
            ToolCallRequest(
                id="read", name="read_file", arguments_raw={"path": "README.md"}
            )
        ]
    )
    env, provider, _, _ = make_environment(tmp_path, script=[response], max_calls=1)
    registry = env.session.tool_registry
    registry._tools["read_file"] = replace(
        registry._tools["read_file"], handler=blocked_tool
    )
    results = []
    worker = threading.Thread(
        target=lambda: results.append(env.execute_prompt("inspect"))
    )
    worker.start()
    assert started.wait(2)
    dispatch(env, "/stop")
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert results[0].status == "cancelled"
    assert len(provider.requests) == 1


def test_mention_does_not_send_fake_file_content_when_context_is_full(tmp_path):
    env, provider, _, _ = make_environment(tmp_path)
    env.session.context_tracker.restore_summary(
        [], env.session.context_tracker.max_context_chars
    )
    with pytest.raises(AgentError, match="budget"):
        env.execute_prompt("Explain @README.md")
    assert provider.requests == []


def test_oversized_prompt_is_rejected_before_a_model_request(tmp_path):
    env, provider, _, _ = make_environment(tmp_path)
    with pytest.raises(AgentError, match="100,000-character"):
        env.execute_prompt("x" * 100_001)
    assert provider.requests == []


def test_file_completion_quotes_spaces_and_handles_partial_quotes(
    tmp_path, monkeypatch
):
    env, _, out, _ = make_environment(tmp_path)
    folder = env.session.project_root / "docs with spaces"
    folder.mkdir()
    (folder / "guide file.md").write_text("guide text\n", encoding="utf-8")
    completer = SlashCompleter(env)

    def complete(text):
        options = list(completer.get_completions(Document(text), CompleteEvent()))
        assert len(options) == 1
        option = options[0]
        return text[: len(text) + option.start_position] + option.text

    for prefix in ("/read docs", '/read "docs with spaces/guide f'):
        command = complete(prefix)
        dispatch(env, command)
        assert "guide text" in "\n".join(out)
    assert complete("/files docs") == "/files 'docs with spaces'"

    # Windows uses quoted paths rather than POSIX backslash escaping.
    import nemotron_agent.repl.autocomplete as autocomplete

    monkeypatch.setattr(autocomplete.sys, "platform", "win32")
    command = complete("/read docs")
    assert command == '/read "docs with spaces/guide file.md"'
    dispatch(env, command)
    assert "guide text" in "\n".join(out)


def test_retry_restores_context_budget_before_repeating_file_reads(tmp_path):
    env, provider, _, _ = make_environment(
        tmp_path,
        script=[
            CompletionResponse(content="First."),
            CompletionResponse(content="Second."),
        ],
    )
    env.execute_prompt("Explain @README.md")
    initial_chars = env.session.context_tracker.chars_used
    assert initial_chars > 0
    dispatch(env, "/retry")
    assert env.session.context_tracker.chars_used == initial_chars
    assert "project introduction" in provider.requests[1].messages[-1].content


def test_failed_mention_resolution_rolls_back_context_tracking(tmp_path):
    env, provider, _, _ = make_environment(tmp_path)
    tracker = env.session.context_tracker
    tracker.max_context_chars = 10
    with pytest.raises(AgentError, match="budget"):
        env.execute_prompt("Explain @README.md and @src/foo.py")
    assert tracker.chars_used == 0
    assert tracker.spans() == []
    assert tracker.get_recorded_hash("README.md") is None
    assert provider.requests == []


def test_first_model_choice_opens_setup_for_that_model_and_key_link(tmp_path):
    from nemotron_agent.providers.unconfigured import UnconfiguredProviderClient

    env, _, out, keyring = make_environment(tmp_path)
    env.session.config = env.config_manager.resolve()
    env.session.provider_client = UnconfiguredProviderClient("openrouter")
    env._needs_setup = True
    env._client_available = False
    env.input_fn = lambda _: pytest.fail(
        "the selected model should not be requested twice"
    )
    prompts = []
    env.secret_input_fn = lambda prompt: (
        prompts.append(prompt) or "sk-or-v1-choosemodel123456"
    )
    dispatch(env, "/models use 2")
    assert not env._needs_setup
    assert env.session.config.model_id.endswith("nemotron-3.5-super-vl:free")
    assert env.session.config.capabilities.supports_vision
    assert "https://openrouter.ai/settings/keys" in "\n".join(out)
    assert prompts and "paste" in prompts[0].lower() and "hidden" in prompts[0].lower()
    assert "sk-or-v1-choosemodel123456" not in "\n".join(out)
    assert keyring.values


def test_instruction_files_reload_and_project_takes_priority(tmp_path):
    env, provider, _, _ = make_environment(
        tmp_path,
        script=[CompletionResponse(content="one"), CompletionResponse(content="two")],
    )
    global_path = env.instructions.global_path
    project_path = env.instructions.project_path(env.session)
    assert global_path.is_file()
    assert not project_path.exists(), (
        "opening a project must not create a repository file"
    )
    global_path.write_text("# Global\nUse tabs.\n", encoding="utf-8")
    project_path.write_text("# Project\nUse spaces.\n", encoding="utf-8")
    env.execute_prompt("first")
    system = provider.requests[0].messages[0].content
    assert "Use tabs." in system and "Use spaces." in system
    assert "Project-specific instructions take precedence" in system
    assert "cannot override system safety" in system
    project_path.write_text("# Project\nUse four spaces.\n", encoding="utf-8")
    env.execute_prompt("second")
    assert "Use four spaces." in provider.requests[1].messages[0].content
    assert "Use spaces.\n" not in provider.requests[1].messages[0].content
    dispatch(env, "/clear")
    assert "Use tabs." in env.session.messages[0].content
    assert "Use four spaces." in env.session.messages[0].content

    other = tmp_path / "other-project"
    other.mkdir()
    (other / "NA_INSTRUCTIONS.md").write_text("Use semicolons.\n", encoding="utf-8")
    dispatch(env, f"/cd {other}")
    assert "Use tabs." in env.session.messages[0].content
    assert "Use semicolons." in env.session.messages[0].content
    assert "Use four spaces." not in env.session.messages[0].content


def test_learn_edits_instruction_files_only_after_approval(tmp_path):
    answers = iter(["n", "y", "y", "y", "n"])
    env, _, out, _ = make_environment(tmp_path, input_fn=lambda _: next(answers))
    global_path = env.instructions.global_path
    project_path = env.instructions.project_path(env.session)
    initial = global_path.read_text(encoding="utf-8")
    dispatch(env, "/learn global Prefer concise answers")
    assert global_path.read_text(encoding="utf-8") == initial
    dispatch(env, "/learn global Prefer concise answers")
    assert "Prefer concise answers" in global_path.read_text(encoding="utf-8")
    dispatch(env, "/learn project Always run pytest")
    assert "Always run pytest" in project_path.read_text(encoding="utf-8")
    dispatch(env, "/learn project Use type hints")
    assert "Use type hints" in project_path.read_text(encoding="utf-8")
    before = project_path.read_text(encoding="utf-8")
    dispatch(env, "/learn project Do not test")
    assert project_path.read_text(encoding="utf-8") == before
    assert "Unified Diff Preview" in "\n".join(out)
    assert "Always run pytest" in env.session.messages[0].content
    assert any(
        entry.action == "edit_file" and not entry.approved
        for entry in env.session.permissions.audit_log
    )


def test_instruction_edits_obey_readonly_noninteractive_and_secret_rules(tmp_path):
    env, _, out, _ = make_environment(tmp_path, read_only=True, input_fn=lambda _: "y")
    global_before = env.instructions.global_path.read_text()
    dispatch(env, "/learn global A new rule")
    dispatch(env, "/learn project A project rule")
    assert env.instructions.global_path.read_text() == global_before
    assert not env.instructions.project_path(env.session).exists()
    assert "read-only" in "\n".join(out).lower()
    not_interactive_root = tmp_path / "noninteractive"
    not_interactive_root.mkdir()
    other, _, other_out, _ = make_environment(
        not_interactive_root, interactive=False, input_fn=lambda _: "y"
    )
    dispatch(other, "/learn project Do something")
    dispatch(other, "/learn global password=secretvalue123")
    assert not other.instructions.project_path(other.session).exists()
    assert "requires interactive approval" in "\n".join(other_out)
    assert "credential" in "\n".join(other_out).lower()
    assert "secretvalue123" not in other.instructions.global_path.read_text()


def test_instruction_symlinks_and_oversized_files_are_not_followed(tmp_path):
    env, provider, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="done")]
    )
    secret = tmp_path / "outside-instructions.md"
    secret.write_text("DO_NOT_SEND_EXTERNAL_CONTENT", encoding="utf-8")
    project = env.instructions.project_path(env.session)
    project.symlink_to(secret)
    env.refresh_guidance()
    assert "DO_NOT_SEND_EXTERNAL_CONTENT" not in env.session.messages[0].content
    dispatch(env, "/learn project Add a rule")
    assert secret.read_text() == "DO_NOT_SEND_EXTERNAL_CONTENT"
    assert "symlink" in "\n".join(out).lower()
    project.unlink()
    project.write_text("X" * (16 * 1024 + 1), encoding="utf-8")
    env.execute_prompt("hello")
    assert "X" * 100 not in provider.requests[0].messages[0].content
    assert "16 KiB limit" in "\n".join(out)

    global_path = env.instructions.global_path
    global_path.unlink()
    global_path.symlink_to(secret)
    dispatch(env, "/learn global Add a rule")
    assert secret.read_text() == "DO_NOT_SEND_EXTERNAL_CONTENT"


def test_one_shot_ask_receives_both_instruction_files(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "NA_INSTRUCTIONS.md").write_text(
        "Project says use spaces.\n", encoding="utf-8"
    )
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "INSTRUCTIONS.md").write_text("Global says use tabs.\n", encoding="utf-8")
    observed = []
    original = AgentSession.run_task

    def capture(session, *args, **kwargs):
        observed.append(session.messages[0].content)
        return original(session, *args, **kwargs)

    monkeypatch.setattr(AgentSession, "run_task", capture)
    assert (
        run_cli(
            ["--path", str(repo), "--mock", "ask", "hello"],
            config_dir=cfg,
            env={},
            output_fn=lambda _: None,
        )
        == 0
    )
    assert "Global says use tabs." in observed[0]
    assert "Project says use spaces." in observed[0]


def test_resume_sends_small_tool_free_recap_and_keeps_it_for_next_request(tmp_path):
    env, _, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="We located the login bug.")]
    )
    env.execute_prompt("Investigate login using @README.md")
    env.plan = Plan(
        goal="Repair login",
        steps=["Inspect authentication", "Add a test"],
        approved=True,
    )
    env.save()
    original_id = env.session_id
    dispatch(env, "/clear")
    recap_client = MockProviderClient(
        [
            CompletionResponse(content="Ready to finish the login fix."),
            CompletionResponse(content="Continuing the work."),
        ]
    )
    env._client_for = lambda _: recap_client
    dispatch(env, "/resume 1")
    assert env.session_id == original_id
    assert len(recap_client.requests) == 1 and recap_client.requests[0].tools == []
    system = recap_client.requests[0].messages[0].content
    assert "<resumed_session_recap>" in system
    assert "Investigate login" in system and "We located the login bug" in system
    assert "Repair login" in system and "Inspect authentication" in system
    assert "project introduction" not in env.resume_recap
    assert len(env.resume_recap) <= 3000
    assert "Ready to continue" in "\n".join(out)
    env.execute_prompt("Continue")
    assert "<resumed_session_recap>" in recap_client.requests[1].messages[0].content
    assert any(
        "Investigate login" in msg.content
        for msg in recap_client.requests[1].messages
        if msg.role == "user"
    )
    dispatch(env, "/clear")
    assert "<resumed_session_recap>" not in env.session.messages[0].content


def test_resume_recap_cannot_execute_tools_and_failure_keeps_session(tmp_path):
    env, _, out, _ = make_environment(
        tmp_path, script=[CompletionResponse(content="earlier")]
    )
    env.execute_prompt("original request")
    original_id = env.session_id
    dispatch(env, "/clear")
    hostile = MockProviderClient(
        [
            CompletionResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="x",
                        name="edit_file",
                        arguments_raw={
                            "path": "unsafe.txt",
                            "new_text": "bad",
                            "create_if_missing": True,
                        },
                    )
                ]
            )
        ]
    )
    env._client_for = lambda _: hostile
    dispatch(env, "/resume 1")
    assert env.session_id == original_id
    assert not (env.session.project_root / "unsafe.txt").exists()
    assert "no tools were run" in "\n".join(out)
    dispatch(env, "/clear")
    failing = MockProviderClient([AgentError("offline")])
    env._client_for = lambda _: failing
    dispatch(env, "/resume 1")
    assert env.session_id == original_id
    assert "restored session is still available" in "\n".join(out)
    assert "<resumed_session_recap>" in env.session.messages[0].content


def _clipboard_test_image():
    import io

    from PIL import Image

    from nemotron_agent.repl.clipboard import validate_image

    output = io.BytesIO()
    Image.new("RGB", (3, 2), "blue").save(output, format="PNG")
    return validate_image(output.getvalue())


def test_clipboard_image_requires_vision_is_sent_and_never_saved(tmp_path):
    import base64

    image = _clipboard_test_image()
    observed = []

    def inspect_request(request):
        observed.append(request.messages[-1].images[0].data == image.data)
        return CompletionResponse(content="It is blue.")

    env, provider, out, _ = make_environment(tmp_path, script=[inspect_request])
    env.clipboard_reader = lambda: image
    dispatch(env, "/paste")
    assert env.pending_image is image
    assert "cannot accept images" in "\n".join(out)
    with pytest.raises(AgentError, match="vision-capable"):
        env.execute_prompt("What is shown?")
    assert env.pending_image is image and not provider.requests
    dispatch(env, "/models use 2")
    env.execute_prompt("What is shown?")
    sent = provider.requests[0].messages[-1]
    assert observed == [True]
    assert not sent.images  # Mock diagnostics do not retain the image either.
    assert "untrusted visual context" in sent.content
    assert env.pending_image is None
    assert all(not message.images for message in env.session.messages)
    assert "no longer available" in env.session.messages[-2].content
    snapshot = (env.sessions.directory / f"{env.session_id}.json").read_text()
    assert base64.b64encode(image.data).decode() not in snapshot
    assert "image/png" not in snapshot
    dispatch(env, "/retry")
    assert len(provider.requests) == 1 and "not retained" in "\n".join(out)
    dispatch(env, "/paste")
    dispatch(env, "/detach")
    assert env.pending_image is None


def test_ctrl_v_attaches_image_but_bracketed_text_paste_stays_text(tmp_path):
    env, _, _, _ = make_environment(tmp_path)
    image = _clipboard_test_image()
    reads = []
    env.clipboard_reader = lambda: reads.append(True) or image
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        reader = terminal_input(env)

        def send_image():
            time.sleep(0.1)
            pipe.send_text("Describe it")
            pipe.send_text("\x16")
            time.sleep(0.25)
            pipe.send_text("\n")

        thread = threading.Thread(target=send_image)
        thread.start()
        assert reader("NA > ") == "Describe it"
        thread.join()
        assert env.pending_image is image and reads == [True]

        def send_text():
            time.sleep(0.1)
            pipe.send_text("\x1b[200~normal pasted text\x1b[201~\n")

        thread = threading.Thread(target=send_text)
        thread.start()
        assert reader("NA > ") == "normal pasted text"
        thread.join()
        assert reads == [True]


def test_learn_natural_instruction_asks_scope_and_approval(tmp_path):
    answers = iter(["p", "y", "c"])
    env, _, out, _ = make_environment(tmp_path, input_fn=lambda _: next(answers))
    dispatch(env, "/learn Prefer descriptive test names")
    assert (
        "Prefer descriptive test names"
        in env.instructions.project_path(env.session).read_text()
    )
    dispatch(env, "/learn Avoid long functions")
    assert (
        "Avoid long functions"
        not in env.instructions.project_path(env.session).read_text()
    )
    assert "No instruction was saved" in "\n".join(out)


def test_global_instruction_change_during_approval_is_not_overwritten(tmp_path):
    env, _, out, _ = make_environment(tmp_path)
    path = env.instructions.global_path

    def change_global(_prompt):
        path.write_text("# A concurrent user edit\n", encoding="utf-8")
        return "y"

    env.session.permissions._input_fn = change_global
    dispatch(env, "/learn global Prefer short functions")
    assert path.read_text() == "# A concurrent user edit\n"
    assert "changed while awaiting approval" in "\n".join(out)


def test_preselected_model_is_preserved_when_setup_is_retried(tmp_path):
    from nemotron_agent.providers.base import ConnectionTestResult

    attempts = []
    prompts = []

    def input_fn(prompt):
        prompts.append(prompt)
        return "n"  # Exit the wizard after its first failed validation.

    def factory(*_):
        attempts.append(1)
        client = MockProviderClient()
        if len(attempts) == 1:
            client.validate_connection = lambda model: ConnectionTestResult(
                ok=False,
                provider="openrouter",
                model_id=model,
                latency_ms=1,
                message="bad key",
            )
        return client

    env, _, output, _ = make_environment(tmp_path, input_fn=input_fn)
    env._needs_setup = True
    env._client_available = False
    env.session.config.mock_mode = False
    env.secret_input_fn = lambda _: "sk-or-v1-retrytest123456789"
    env.client_factory = factory
    dispatch(env, "/models 2")
    assert not env._client_available
    assert env._pending_setup_selection is not None
    env._ensure_client()
    assert env._client_available
    assert env.session.config.model_id == "nvidia/nemotron-3.5-super-vl:free"
    assert len(attempts) == 3  # Two validation clients, then the active client.
    assert not any("Choose a model preset" in prompt for prompt in prompts)
    assert any("https://openrouter.ai/settings/keys" in line for line in output)


def test_models_list_and_completion_mention_number_shortcut(tmp_path):
    from prompt_toolkit.document import Document

    from nemotron_agent.repl.autocomplete import SlashCompleter

    env, _, output, _ = make_environment(tmp_path)
    dispatch(env, "/models")
    assert "Choose one with /models <number>" in "\n".join(output)
    completions = list(SlashCompleter(env).get_completions(Document("/models "), None))
    assert "3" in {completion.text for completion in completions}


def test_first_preset_for_a_new_provider_prompts_for_its_key(tmp_path):
    prompts = []
    env, _, output, _ = make_environment(tmp_path)
    env.session.config = env.session.config.model_copy(
        update={
            "provider": "openai_compatible",
            "base_url": "https://example.com/v1",
            "model_id": "custom-model",
            "mock_mode": False,
        }
    )
    env.secret_input_fn = lambda prompt: (
        prompts.append(prompt) or "sk-or-v1-switchkey123456789"
    )
    env.client_factory = lambda *_: MockProviderClient()
    dispatch(env, "/models 2")
    assert env.session.config.provider == "openrouter"
    assert env.session.config.model_id == "nvidia/nemotron-3.5-super-vl:free"
    assert any("https://openrouter.ai/settings/keys" in line for line in output)
    assert any(
        "OpenRouter API key" in prompt and "hidden" in prompt for prompt in prompts
    )
    assert any("rather than openai_compatible" in line for line in output)


def test_failed_new_provider_setup_keeps_existing_model_and_client(tmp_path):
    from nemotron_agent.providers.base import ConnectionTestResult

    env, provider, output, _ = make_environment(tmp_path)
    original = env.session.config.model_copy(
        update={
            "provider": "openai_compatible",
            "base_url": "https://example.com/v1",
            "model_id": "custom-model",
            "mock_mode": False,
        }
    )
    env.session.config = original
    env.secret_input_fn = lambda _: "sk-or-v1-failedswitch123456789"

    def fail_factory(*_):
        client = MockProviderClient()
        client.validate_connection = lambda model: ConnectionTestResult(
            ok=False,
            provider="openrouter",
            model_id=model,
            latency_ms=1,
            message="Validation failed",
        )
        return client

    env.client_factory = fail_factory
    dispatch(env, "/models 3")
    assert env.session.config == original
    assert env.session.provider_client is provider
    assert "previous active model is unchanged" in "\n".join(output)


def test_deleting_global_instructions_during_approval_is_a_conflict(tmp_path):
    env, _, output, _ = make_environment(tmp_path)
    path = env.instructions.global_path

    def delete_global(_prompt):
        path.unlink()
        return "y"

    env.session.permissions._input_fn = delete_global
    dispatch(env, "/learn global Be brief")
    assert not path.exists()
    assert "changed while awaiting approval" in "\n".join(output)


def test_update_slash_command_reports_status(tmp_path):
    env, _, output, _ = make_environment(tmp_path)
    dispatch(env, "/update")
    joined = "\n".join(output)
    assert "nemotron-agent" in joined
    assert "Install method:" in joined

    dispatch(env, "/update --check")
    joined = "\n".join(output)
    assert "Install method:" in joined

    # No update should be available (or unreachable); --yes must not crash.
    dispatch(env, "/update --yes")
    joined = "\n".join(output)
    assert "Nothing to update" in joined or "unable to reach PyPI" in joined


def test_model_list_uses_saved_vision_capability_override(tmp_path):
    env, _, output, _ = make_environment(tmp_path)
    model = "nvidia/nemotron-3.5-lightning:free"
    dispatch(env, f"/config set-capability --for-model {model} --vision on")
    dispatch(env, "/models")
    lines = [line for line in output if "Nemotron 3.5 Lightning —" in line]
    assert lines and "image input" in lines[-1]
