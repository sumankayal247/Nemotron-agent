# Nemotron Agent

`NA` (also available as `nemotron-agent`) is a bounded command-line coding assistant for **local codebases on your computer**, powered by NVIDIA Nemotron models through an OpenAI-compatible API provider (initially configured for **OpenRouter**).

It runs locally on your machine against **any local project directory** (plain folders or Git projects — it does not connect to GitHub or require a remote repository) and is intentionally designed as a **safe, reviewable engineering tool** rather than an unrestricted autonomous shell agent:

- **Local folder confinement & read-only by default**: inspects your selected local project directory, searches relevant files, and reads bounded line slices on demand.
- **Reviewable file changes**: verifies file hashes against external edits, preserves unrelated code, displays a unified diff (`--- a/...`, `+++ b/...`), and requires explicit approval before writing.
- **Allowlisted command execution**: never invokes a shell (`shell=False`); accepts only explicit argument arrays (`argv`) matching allowlisted test, lint, typecheck, format, or read-only Git inspection commands, and asks explicit approval showing the exact executable, arguments, working directory, and reason.
- **Strict secret & boundary protection**: respects `.gitignore` (if present) and configurable exclusions, confines paths to the selected local project folder, excludes secret-like files (`.env`, private keys, credentials), hard-blocks reading secret-like files when using hosted `:free` endpoints, and redacts API keys across all logs and errors.

---

## Installing on Your Computer

Requires **Python 3.11+** on macOS, Linux, or Windows.

### Option A: Install globally for your user with `pipx` (recommended for daily use across any local folder)

```bash
pipx install .
```

After this, you can `cd` into **any folder on your computer** (or pass `--path /path/to/your/codebase`) and run `NA` or `nemotron-agent`. Both executable names launch the same interactive environment; `nemotron-agent` remains available on platforms where an uppercase executable name is inconvenient.

### Option B: Install in a local virtual environment (for development & running tests)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Verify installation:

```bash
NA --version
nemotron-agent --version
nemotron-agent --help
```

---

## Selectable Model Presets & Provider Notes

Out of the box, `nemotron-agent` provides four selectable NVIDIA Nemotron presets configured for **OpenRouter** (`https://openrouter.ai/api/v1`):

| # | Display Name | Model ID | Default Tool Mode |
|---|--------------|----------|-------------------|
| 1 | **Nemotron 3 Super** | `nvidia/nemotron-3-super-120b-a12b:free` | Native tool calling |
| 2 | **Nemotron 3.5 Lightning** | `nvidia/nemotron-3.5-lightning:free` | Native tool calling |
| 3 | **Nemotron 3.5 Super VL** | `nvidia/nemotron-3.5-super-vl:free` | Validated structured-prompt fallback (`<tool_call>`) |
| 4 | **Nemotron 3 Ultra** | `nvidia/nemotron-3-ultra-550b-a55b:free` | Native tool calling |

### Important Provider & Availability Notice

- The `:free` model IDs above are **OpenRouter-specific identifiers** and do **not** imply that free endpoints with those IDs are offered directly by NVIDIA or by other providers.
- Free-tier availability, rate limits, context windows, supported API parameters (such as native `tools` vs. prompt-based tool requests), model IDs, and data-retention/training policies on hosted `:free` endpoints can change at any time. Always check your selected provider's current documentation, terms, and privacy settings before sending repository code.
- Preset definitions and per-model capability profiles (`supports_tool_calling`, `supports_structured_output`, `supports_vision`, `max_context_chars`) live in user configuration so they can be updated without modifying core agent code.
- `nemotron-agent` **never silently switches models or providers** if a request fails or hits a rate limit.

---

## First-Run Setup & Credential Handling

The interactive environment opens even before a provider is configured. Run setup explicitly with `/setup` inside the REPL or with the existing shell command:

```bash
nemotron-agent setup
```

Start with `/models` to see the choices, then select one (for example, `/models 3`). On first selection, setup immediately guides you through connecting that model without asking you to select it again. A preset uses its listed provider; if its key is missing, setup asks for it before switching. Custom model IDs stay on the current provider, and mock mode remains offline. You can also start setup with your first model request. Local commands such as `/help` and `/exit` work before setup; one-shot requests retain their existing setup behavior.

During first-run setup:

1. You choose one of the four Nemotron presets (or enter a custom provider, base URL, and model ID).
2. If no key is configured, open the displayed provider link. For the built-in OpenRouter presets, create and copy a key at **https://openrouter.ai/settings/keys**. The key belongs to the provider, so one OpenRouter key can be used for its supported models. For a custom provider, use that provider's own dashboard.
3. Paste the key into the hidden prompt and press Enter. Nothing appears while you paste or type; that is normal. The key is sent in the `Authorization` header to the configured provider and is never echoed or stored in conversation history.
4. Before saving, the CLI performs a lightweight connection test (`Reply with OK.`, `max_tokens=8`) **without sending any repository files**. If validation fails or is cancelled, you can retry with updated settings or exit cleanly.
5. **Non-secret settings** (`provider`, `model_id`, `base_url`, limits, exclusions, capability overrides) are saved in your user-level configuration directory (`~/.config/nemotron-agent/config.json` on Linux/macOS, `%APPDATA%\nemotron-agent\config.json` on Windows) — never inside a project repository.
6. **API credentials** are stored in your **operating system credential store** (`keyring`: macOS Keychain, Windows Credential Locker, or Linux Secret Service) when available. If an OS keyring is unavailable (for example, in a headless container or SSH session), the CLI explains the fallback and asks explicit permission before writing to `~/.config/nemotron-agent/credentials.json` with `0600` owner-only permissions (inside a `0700` directory). If declined, the key can be held in memory for the current session only or provided via environment variables.

### Environment Variables & Precedence

Configuration is resolved in strict order of precedence:

**`CLI flags` > `Environment variables` > `Saved user configuration / keyring` > `Built-in defaults`**

An explicit model selected in the REPL overrides those settings for the current session. A newly entered key is also used for the current session; if an older key is set in an environment variable, update or unset it for future launches. API-key entry requires a terminal that can hide input; otherwise use the documented environment variable. Setup displays the endpoint receiving your key and asks for extra confirmation for HTTP or a non-OpenRouter host.

| Environment Variable | Purpose |
|----------------------|---------|
| `NEMOTRON_AGENT_PROVIDER` | Active provider (`openrouter`, `openai_compatible`, `mock`) |
| `NEMOTRON_AGENT_MODEL` | Active model ID or preset name/number (`1`–`4`) |
| `NEMOTRON_AGENT_BASE_URL` | Provider OpenAI-compatible API base URL |
| `NEMOTRON_AGENT_API_KEY` | API key for the active provider (highest env key priority) |
| `OPENROUTER_API_KEY` | OpenRouter API key (used when provider is `openrouter`) |
| `NEMOTRON_AGENT_MOCK` | Set to `1` or `true` to run offline using the mock model client |
| `NEMOTRON_AGENT_CONFIG_DIR` | Override user-level config directory (must be outside target repo) |
| `NEMOTRON_AGENT_MAX_TURNS` | Max model turns per task (default: `8`) |
| `NEMOTRON_AGENT_MAX_TOOL_CALLS` | Max tool invocations per task (default: `12`) |
| `NEMOTRON_AGENT_TIMEOUT` | HTTP request timeout in seconds (default: `30.0`) |

---

## Usage on Local Codebases

### Selecting & Working with Any Local Folder on Your Computer

```bash
# Option 1: cd into any local project folder (Git is NOT required) and launch interactive chat
cd ~/Projects/my-local-app
NA  # or: nemotron-agent

# Option 2: Point directly at any local folder on your computer using --path / -p / --dir / --project
nemotron-agent --path ~/Desktop/my-local-app
nemotron-agent -p "C:\Users\You\Projects\my-local-app" "Explain how this project is structured."

# Option 3: Prompt interactively to pick a local folder on launch
nemotron-agent --select-dir

# One-shot prompt (either via `ask` or direct positional prompt)
nemotron-agent --path ~/Projects/my-local-app "Where are authentication errors handled?"
nemotron-agent ask --read-only "Summarize the main entrypoints."

# Control file inclusion/exclusion in prompts
nemotron-agent --exclude "docs/**" --exclude "*.md" "Summarize core Python modules."
```

*(Note: If you run `nemotron-agent` directly from your OS home folder `~` without `--path`, it will prompt you to enter the local project folder you want to inspect rather than scanning your entire home directory.)*

### Interactive environment

`NA`, bare `nemotron-agent`, and `nemotron-agent chat` use the same persistent, repository-aware prompt:

```text
Nemotron Agent 0.1.0
Repository: /path/to/project
Model: <provider> / <model>
NA >
```

Type `/` to display commands immediately, then use **Tab** to complete or cycle through matches. Arrow keys navigate in-memory input history; **Alt+Enter** inserts a newline. `/help` lists commands from the same registry that handles execution and completion, and `/help <command>` shows detailed usage. Unknown slash commands are handled locally, never sent to the model. Any input not beginning with `/` remains an ordinary model request.

| Command | Purpose |
|---------|---------|
| `/help [command]` | List commands or show detailed usage. |
| `/clear` | Clear conversation and tracked file context only; preserve settings, plan, learned notes, skills, repository files, and previous saved sessions. |
| `/resume [number\|id-prefix]` | Restore a session and send a short, tool-free recap to its selected model. |
| `/agents [list\|info <name>\|use <name>]` | Inspect/select a persona; these are not autonomous workers. |
| `/skills [list\|info <name>\|use <name>]` | Inspect or activate installed skills. `/skills use code-review` also registers `/review-code`. |
| `/models [list\|current\|<model>\|use <model>]` | List or choose a model, e.g. `/models 3`; image-capable presets are marked. First-time selection opens the key setup. Use `/config set-model` to change the saved default. |
| `/efforts [low\|medium\|high]` | Set **local** turn and tool-call caps, never above configured limits. No unsupported provider reasoning parameter is claimed or sent. |
| `/plan [create [goal]\|show\|approve\|clear]` | Produce a read-only plan; approval records a decision but never executes it. `/plan` shows an existing plan or creates one for the last prompt; `create` can also ask for a goal. |
| `/learn [show\|paths\|global <instruction>\|project <instruction>\|list\|add <note>\|search <query>]` | View or add approved instruction-file entries, or manage explicit repository notes. `/learn <instruction>` asks which scope to use. |
| `/artifacts [list\|open <name>\|path]` | Inspect files **created by the agent in this session**, not arbitrary project files. Open uses the existing safe read tool. |
| `/search <query>`, `/read <file>`, `/files [directory]`, `/tools` | Convenience interfaces over the existing validated repository tools. A local `/read` does not automatically send the file to the model. |
| `/config ...`, `/auth ...`, `/setup`, `/ping` | Use the existing configuration, credential, setup, and connectivity functions without leaving the REPL. Key entry remains hidden. |
| `/paste`, `/detach` | Attach the current clipboard image to the next prompt, or remove it before sending. Ctrl+V also attaches an image when the terminal passes it through. |
| `/status`, `/context`, `/history`, `/copy`, `/retry`, `/stop`, `/version` | Inspect state, see recent turns, copy the last answer using terminal OSC 52 where supported, or retry safely. |
| `/project [path]`, `/cd <path>`, `/provider`, `/key`, `/clear-key`, `/readonly [on\|off]` | Existing interactive project, provider, credential, and mode commands remain available. A startup `--read-only` lock cannot be disabled. |
| `/exit`, `/quit` | Save the current resumable session and exit. Ctrl+D also exits. |

The agent executes a task synchronously. **Press Ctrl+C while a provider call or tool is running** to interrupt it and recover the conversation; a blocking operation may take time to return. `/stop` at the prompt reports that no operation is running. A cooperative stop request also prevents further tool steps after the current blocking step returns. `/retry` refuses to replay a request automatically after edits, command execution, interruption, or an image request whose image was discarded.

#### Add an image to a prompt

1. Copy an **image or screenshot** to your desktop clipboard (not just its filename).
2. At `NA >`, press **Ctrl+V**. If your terminal handles Ctrl+V itself, type `/paste` instead. Right-click or your terminal's normal paste shortcut still pastes text as usual.
3. Check the displayed image details, type your question, and press Enter. Use `/detach` to discard it before sending.

The currently selected model must support image input. `/models` marks the vision preset; for a custom model, enable `--vision on` only after verifying that the provider supports it. The image is limited to **5,000,000 bytes (5 MB)** and 16 megapixels; PNG, JPEG, WebP, and non-animated GIF are supported. On Windows, an uncompressed clipboard bitmap is also converted to PNG, provided the clipboard representation and converted image meet the limit. Audio and video are not supported.

Clipboard access reads only the **current** image, not clipboard history. It requires a local desktop clipboard: macOS and Windows use their system clipboard; Linux needs `wl-paste` (`wl-clipboard`) on Wayland or `xclip` on X11. Remote/headless sessions may not have access to your local clipboard. An image is sent to the configured provider with your prompt; inspect it yourself for sensitive information. It is kept only for that task, is **not saved** in session files, and must be pasted again for a retry or after resuming. Mock mode cannot actually analyze images.

#### Repository references

Include files or directories in a normal prompt with `@`:

```text
NA > explain @README.md
NA > review @src/nemotron_agent/agent.py
NA > summarize @src/nemotron_agent/tools/
NA > explain @"docs/file with spaces.md"
```

Files are read in bounded slices and directories are listed through the existing tools; their contents are framed as untrusted repository data. Prompts are limited to 100,000 characters, with at most eight references and 48,000 characters of referenced content. The reads count against that task's tool-call budget. Relative partial filenames are matched against the safe repository index; ambiguous matches require a choice. References cannot escape the project, follow external symlinks, access internal Git or secret-like files, or bypass ignored-file and binary protections.

#### Sessions, knowledge, and artifacts

Resumable session JSON is saved under the user configuration directory in `sessions/`, outside the project. Files use owner-only permissions where supported and known credentials are redacted. **Sessions may still contain prompts and repository source code**; they are not encrypted, and pattern-based redaction cannot recognize every possible secret. Do not put secrets in prompts. `/clear` archives the previous conversation before starting a new one, and does not delete saved sessions.

When you select a session with `/resume 1` (or an ID prefix), NA restores its conversation and plan, then sends a bounded recap of recent turns and plan details together with your current instructions to the selected provider. This reconnect request has **no tools**, so it cannot make edits or run commands. The recap also accompanies later requests because model APIs are stateless; the saved conversation remains available. This may use provider tokens. If the reconnect request fails, the local session is still restored and the recap will be sent with the next request. Resuming never restores prior approvals or file hashes: files must be re-read before editing. A saved session cannot silently redirect credentials to a different provider endpoint.

#### Instructions for every session

NA uses two UTF-8 Markdown files, re-reading them before each model request (including one-shot `ask`):

- **Global:** `<user-config-dir>/INSTRUCTIONS.md`. A starter file is created outside the repository if it does not exist; you can edit it directly.
- **Project:** `<project-root>/NA_INSTRUCTIONS.md`. Create or edit it yourself, or ask NA to add a rule. Opening a project does **not** create or change this file.

Both sets of instructions are included in the prompt. Project instructions take priority over conflicting global instructions, while core safety rules, user requests, approvals, and path protections remain higher priority than either. The project file can be controlled by the repository, so it cannot grant permissions or override security. Each file is limited to 16 KiB; symlinks and project files excluded by repository rules are not loaded. The instruction contents are sent to the provider. Do not put credentials in them: known secret patterns are redacted, but redaction cannot recognize every secret.

```text
NA > /learn show                              # show both sets
NA > /learn paths                             # see their exact paths
NA > /learn global Prefer concise explanations
NA > /learn project Always add a regression test
NA > /learn Use descriptive test names         # asks whether project or global
```

`/learn global` and `/learn project` show a diff and require approval before changing a file. Strict read-only and non-interactive restrictions still apply. The existing `/learn add <note>` and `/learn search <query>` manage separate, explicitly saved repository-scoped notes. Neither conversation history nor discovered repository facts are automatically promoted to notes or instructions.

Artifacts are agent-created project files recorded in session metadata; ordinary existing files and edits are not automatically labeled artifacts. They stay at their original project paths. `/artifacts open` applies the same path, secret, binary, and output protections as the read tool.

#### Installed skills

Skills are declarative JSON manifests in `<user-config-dir>/skills/`. For example:

```json
{
  "name": "python-review",
  "description": "Review Python changes",
  "instructions": "Look for regressions and missing tests.",
  "context": "Prefer focused checks.",
  "tools": ["read_file", "search_files"],
  "workflows": {"audit": "Review the requested changes and cite evidence."},
  "commands": {"python-audit": "audit"}
}
```

Use `/skills use python-review` to activate the guidance and register `/python-audit [task]`. Declared tools must already exist in the bounded registry; workflow commands send ordinary requests through that registry. Skills cannot grant shell access, relax approvals, or override read-only mode. Skill text is treated as untrusted guidance.

### Configuration & Credential Management Commands

```bash
# Show active provider, model, capabilities, and credential status (redacted)
nemotron-agent config show

# List selectable NVIDIA Nemotron presets
nemotron-agent config presets

# Change saved default model (by preset number 1-4, slug, display name, or model ID)
nemotron-agent config set-model 2
nemotron-agent config set-model "Nemotron 3 Ultra" --skip-check

# Configure a custom OpenAI-compatible provider
nemotron-agent config set-provider openai_compatible --base-url https://api.example.com/v1 --model my-org/model

# Override capability flags for a model (e.g., use structured-prompt tool fallback)
nemotron-agent config set-capability --for-model nvidia/nemotron-3-super-120b-a12b:free --tool-calling off

# Update or clear saved API credentials
nemotron-agent auth set-key
nemotron-agent auth clear

# Test provider connectivity without sending any repository files
nemotron-agent ping
```

### Running in Offline Mock Mode

You can run the full CLI and repository tool workflow completely offline without API keys or network calls:

```bash
nemotron-agent --mock "Inspect this repository and summarize its structure"
nemotron-agent --mock "Read pyproject.toml"
nemotron-agent --mock chat

# Or via environment variable:
NEMOTRON_AGENT_MOCK=1 nemotron-agent "search for ToolRegistry"
```

---

## Permissions, Tool Workflow & Safety Boundaries

1. **Bounded Tools Only**: The model can request only six validated tools (`repo_overview`, `list_files`, `search_files`, `read_file`, `edit_file`, `run_command`). Every tool request is validated against strict Pydantic schemas (`extra="forbid"`) in application code. Unknown tools and malformed JSON are rejected.
2. **Untrusted Data Framing**: Repository files, comments, `README` files, and command outputs are wrapped in `<untrusted_repo_data>` tags, and all path/permission checks are enforced in Python code rather than relying on prompt obedience.
3. **Path Confinement & Symlink Protection**: Paths resolving outside the selected project root (including `..` traversal and symlinks pointing outside the repository) and internal `.git/` files are blocked.
4. **Secret Protection**: `.env`, `.env.*` (excluding `.env.example`/`.env.sample`/`.env.template`), private keys (`*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`), and credential files (`credentials.json`, `.netrc`, `.npmrc`, `.pypirc`) are excluded from discovery and **hard-blocked** when using `:free` hosted endpoints.
5. **Conflict-Checked Unified Diffs**: `edit_file` requires reading an existing file first, verifies its SHA-256 hash has not changed externally, ensures `old_text` matches uniquely so unrelated user edits are preserved, shows a unified diff, and writes atomically only after `[y/N]` approval.
6. **Allowlisted `argv` Commands (`shell=False`)**: `run_command` blocks shells (`sh`, `bash`, `powershell`), inline code flags (`python -c`, `node -e`), destructive/network binaries (`rm`, `curl`, `ssh`, `sudo`, `git push`, `git reset`), strips API key env vars from child processes, enforces timeouts and output byte caps, and requires `[y/N]` approval after displaying exact executable, `argv`, working directory, and reason.

---

## Architecture

```text
src/nemotron_agent/
├── cli.py             # CLI parsing, existing config/auth commands & REPL entry
├── repl/
│   ├── commands.py    # Slash-command metadata, validation, execution & help
│   ├── autocomplete.py # prompt_toolkit completion and keyboard shortcuts
│   ├── clipboard.py   # Bounded, in-memory clipboard image capture
│   ├── session.py     # Interactive lifecycle and bounded request handling
│   ├── handlers/      # Session, repository, configuration, and general commands
│   ├── instructions.py # Approved global/project instruction files
│   ├── recap.py       # Small, redacted session-resume recap
│   ├── mentions.py    # Confined @ references using existing read tools
│   ├── capabilities.py # Personas and declarative skill/workflow registry
│   └── storage.py     # Private resumable sessions and explicit learned notes
├── setup_wizard.py    # Interactive setup, hidden key input & validation
├── config.py          # Presets, capability profiles, user config & env precedence
├── credentials.py     # OS keyring adapter + explicit-consent local fallback
├── logging_utils.py   # SecretRedactor, logging filters & safe error hierarchy
├── permissions.py     # Read-only mode & explicit user approval gates
├── agent.py           # Bounded tool loop, turn/call budgets & cancellation
├── providers/         # Provider abstraction, OpenAI-compatible and offline mock
├── repo/              # PathGuard, discovery, bounded search and context tracking
└── tools/             # Schemas, validated registry, conflict-checked edits & argv runner
```

---

## Running Tests

All unit and integration tests run completely offline using `MockProviderClient`, `httpx.MockTransport`, fake keyrings, and temporary repositories — no API keys or internet access required:

```bash
.venv/bin/pytest -v
```

---

## Known Limitations

- **Single-file atomic edits**: Each `edit_file` call modifies one file at a time (multi-file changes are proposed and approved file-by-file rather than as a multi-file transaction rollback).
- **Root `.gitignore` focus**: `.gitignore` parsing loads the repository root `.gitignore` plus built-in and user-configured `--exclude` / `--include` patterns; deeply nested per-subfolder `.gitignore` overrides are not recursively merged in v0.1.
- **Text-first context**: Binary files and images are excluded from repository context even when selecting the `Nemotron 3.5 Super VL` preset; image attachment workflows are out of scope for v0.1.
- **Provider endpoint variance**: On OpenRouter `:free` tiers, upstream availability, latency, and native tool-calling support vary over time; use `nemotron-agent config set-capability --tool-calling off` if an upstream endpoint rejects native `tools` parameters.
