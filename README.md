# Nemotron Agent

`nemotron-agent` is a bounded command-line coding assistant for **local codebases on your computer**, powered by NVIDIA Nemotron models through an OpenAI-compatible API provider (initially configured for **OpenRouter**).

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

After this, you can `cd` into **any folder on your computer** (or pass `--path /path/to/your/codebase`) and run `nemotron-agent`.

### Option B: Install in a local virtual environment (for development & running tests)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Verify installation:

```bash
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

Run interactive setup explicitly or simply launch `nemotron-agent` for the first time:

```bash
nemotron-agent setup
```

During first-run setup:

1. You choose one of the four Nemotron presets (or enter a custom provider, base URL, and model ID).
2. If no API key is configured, the CLI explains how the key is used (sent only in the HTTPS `Authorization: Bearer ...` header to your configured provider base URL) and how to create one in your OpenRouter account's API Keys settings.
3. Key input is hidden (`getpass`) and never echoed or written to logs, errors, prompts, or conversation history.
4. Before saving, the CLI performs a lightweight connection test (`Reply with OK.`, `max_tokens=8`) **without sending any repository files**. If validation fails or is cancelled, you can retry with updated settings or exit cleanly.
5. **Non-secret settings** (`provider`, `model_id`, `base_url`, limits, exclusions, capability overrides) are saved in your user-level configuration directory (`~/.config/nemotron-agent/config.json` on Linux/macOS, `%APPDATA%\nemotron-agent\config.json` on Windows) — never inside a project repository.
6. **API credentials** are stored in your **operating system credential store** (`keyring`: macOS Keychain, Windows Credential Locker, or Linux Secret Service) when available. If an OS keyring is unavailable (for example, in a headless container or SSH session), the CLI explains the fallback and asks explicit permission before writing to `~/.config/nemotron-agent/credentials.json` with `0600` owner-only permissions (inside a `0700` directory). If declined, the key can be held in memory for the current session only or provided via environment variables.

### Environment Variables & Precedence

Configuration is resolved in strict order of precedence:

**`CLI flags` > `Environment variables` > `Saved user configuration / keyring` > `Built-in defaults`**

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
nemotron-agent

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

### Interactive Chat Slash Commands

Inside `nemotron-agent chat`:

- `/help` — Show interactive command reference
- `/project [local-path]` — Show or switch the active local codebase folder on your computer
- `/cd <local-path>` — Switch to another local codebase folder (clears prior file context)
- `/status` or `/config` — Show active provider, model, local codebase path, capabilities, mode, and redacted credential status
- `/presets` — Display the four selectable Nemotron presets
- `/model [1-4 | slug | model_id]` — View or switch the active model
- `/provider <name> [base_url]` — Switch provider explicitly
- `/key` — Update the API key for the active provider
- `/clear-key` — Clear saved credentials for the active provider
- `/context` — Inspect conversation turns, character budget usage, and exact `file:start-end` spans in context
- `/clear` — Clear conversation history and tracked file context
- `/readonly [on|off]` — Inspect or toggle strict read-only lock
- `/exit` or `/quit` — Exit interactive chat (`Ctrl+C` cancels an in-progress task without losing the session)

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
├── cli.py             # CLI parser, config/auth subcommands, one-shot runner & chat REPL
├── setup_wizard.py    # Interactive first-run preset selection, hidden key input & validation
├── config.py          # 4 Nemotron presets, capability profiles, user config & env precedence
├── credentials.py     # OS keyring adapter + explicit-consent 0600 local fallback
├── logging_utils.py   # Process-wide SecretRedactor, logging filters & safe error hierarchy
├── permissions.py     # Read-only default & explicit user approval gates
├── agent.py           # Bounded tool loop, turn/call budgets, Ctrl+C recovery & task summary
├── providers/
│   ├── base.py        # ProviderClient ABC, normalized models & <tool_call> fallback parser
│   ├── openrouter.py  # OpenAI-compatible HTTP client, non-repo ping & bounded retries
│   └── mock.py        # Offline scriptable & heuristic MockProviderClient
├── repo/
│   ├── discovery.py   # PathGuard, symlink confinement, .gitignore & secret exclusions
│   └── context.py     # Targeted search, line-ranged reads, SHA-256 & budget tracking
└── tools/
    ├── schemas.py     # Strict Pydantic schemas (extra="forbid")
    ├── registry.py    # Schema validation & <untrusted_repo_data> framing
    ├── files.py       # Conflict check, unified diff preview & atomic narrow writes
    └── commands.py    # Allowlisted argv command execution (shell=False, timeouts, output caps)
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
