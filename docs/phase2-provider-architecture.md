# Using AI Coding CLIs as Inference Backends

> How we made Claude Code, OpenCode, Codex, and Kilo work as LLM providers for a RAG app — without API keys.

## The Problem

Every RAG tutorial ends the same way:

```python
from openai import OpenAI
client = OpenAI(api_key="sk-...")
```

Paste your API key, pay per token, done. But our users are developers who already pay $20-200/month for AI coding subscriptions — Claude Code, Codex, ChatGPT Plus. Asking them to pay again for API access is bad product design.

What if we could use the subscriptions they already have?

## The Idea

Claude Code, OpenCode, Codex, and Kilo all have CLI tools with non-interactive modes. They're designed for CI/CD pipelines and scripts. Each one:

1. Is already installed and authenticated on the developer's machine
2. Has a headless mode that accepts a prompt and returns a response
3. Outputs structured JSON we can parse

We built a provider abstraction that detects which CLIs are installed, wraps them as LLM backends, and lets the user pick one from the UI. Zero API keys required.

## What's Available

We surveyed every AI coding CLI and tested which ones actually work as subprocess inference backends:

| CLI | Install | Auth | Free Models | Headless Command |
|-----|---------|------|-------------|-----------------|
| **Claude Code** | `npm i -g @anthropic-ai/claude-code` | OAuth (Max/Pro sub) | Via subscription | `claude -p "..." --output-format json` |
| **OpenCode** | `npm i -g opencode-ai` | OAuth or API key | 6 free models | `opencode run "..." --format json` |
| **Codex CLI** | `npm i -g @openai/codex` | OAuth (ChatGPT sub) | Via subscription | `codex exec "..." --json` |
| **Kilo CLI** | `npm i -g @kilocode/cli` | Auto on install | 10 free models | `kilo --auto --json "..."` |

Plus a Custom endpoint option for anyone who has their own OpenAI-compatible server.

## The Provider Interface

Every provider implements the same contract:

```python
class Provider(ABC):
    def detect(self) -> bool:
        """Is this CLI installed on the system?"""

    def status(self) -> ProviderStatus:
        """Installed? Authenticated? What models? Version?"""

    def chat(self, messages: list[dict], model: str = None) -> str:
        """Send messages, get response text."""

    def stream(self, messages: list[dict], model: str = None) -> Iterator[str]:
        """Send messages, yield response chunks."""

    def free_models(self) -> list[ProviderModel]:
        """Which models cost nothing?"""

    def install(self) -> bool:
        """Install the CLI tool (with user permission)."""
```

The RAG pipeline doesn't care which provider is active. It calls `provider.chat(messages)` and gets text back. Switching providers is one line: `registry.active = "kilo"`.

## Parsing Four Different Output Formats

Each CLI has its own JSONL event format. Here's what we had to figure out by reading source code and testing:

### Claude Code

```json
{"type":"assistant","message":{"content":[{"type":"text","text":"the answer"}]}}
{"type":"result","result":""}
```

The `result` field in the final event is often empty. The actual answer is in `type=assistant` → `message.content[0].text`. We discovered this by running `--output-format stream-json --verbose` and inspecting every event.

### OpenCode

```json
{"type":"step_start","timestamp":...,"part":{...}}
{"type":"text","timestamp":...,"part":{"text":"the answer"}}
{"type":"step_finish","timestamp":...,"part":{...}}
```

Clean and simple. Answer is in `type=text` → `part.text`.

### Codex CLI

```json
{"type":"thread.started","thread_id":"..."}
{"type":"item.completed","item":{"type":"agent_message","text":"the answer"}}
{"type":"turn.completed","usage":{...}}
```

Answer is in `type=item.completed` where `item.type=agent_message`. Watch out: specifying `--model o4-mini` fails on ChatGPT accounts — use the default model.

### Kilo CLI

```
]0;Kilo Code - tutorialvault[2J[3J[H{"timestamp":1,"type":"welcome",...}
[2K[1A[2K[G{"say":"reasoning","partial":true,"content":"thinking..."}
[2K[1A[2K[G{"say":"text","partial":false,"content":"the answer"}
```

The messiest output. ANSI escape codes (`[2K[1A[2K[G`) are mixed into every line because Kilo uses a TUI framework (Ink/React) that writes terminal control sequences to stdout even in `--json` mode. We strip them with regex before parsing:

```python
import re
ANSI_RE = re.compile(r'(\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07)')

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text)
```

The answer is in the last event where `say=text` and `partial=false`. There's also `say=completion_result` in some cases. Ignore `say=reasoning` (thinking tokens) and `say=text` where `partial=true` (incremental updates).

## The Claude Code Speed Problem

This was the biggest challenge. Claude Code loads everything on startup:

```
Load binary          ~1s
Read config          ~0.5s
Connect MCP servers  ~30-60s  ← blender, houdini, gmail, etc.
Run hooks            ~5-10s   ← claude-mem plugin
Read CLAUDE.md       ~1s
Send prompt          ~2-3s
```

Total: 40-70 seconds per query. Unusable.

The `--bare` flag skips all that but also skips OAuth — and Max subscriptions use OAuth. So `--bare` = fast but "Not logged in."

### The Fix: Isolated Config Directory

Claude Code reads its config from `~/.claude/`. If you set `CLAUDE_CONFIG_DIR` to a different path, it reads from there instead. The trick:

1. Create an empty directory
2. Copy ONLY `~/.claude/.credentials.json` into it (the OAuth tokens)
3. Set `CLAUDE_CONFIG_DIR` to that directory
4. No MCP servers, no hooks, no plugins — just auth

```python
env = os.environ.copy()
env["CLAUDE_CONFIG_DIR"] = str(isolated_dir)

result = subprocess.run(
    ["claude", "-p", prompt, "--output-format", "json",
     "--no-session-persistence", "--max-turns", "1"],
    env=env, capture_output=True, text=True, timeout=60,
)
```

Result: **60+ seconds → 6 seconds.** The user's main Claude Code setup is completely untouched.

### The Authorization Flow

We can't just copy someone's credentials without asking. The flow in our app:

1. Detect `claude` is installed → check `claude auth status`
2. If logged in, show: "TutorialVault can use your Claude subscription. We copy your auth to an isolated config. Your main setup is not affected."
3. User clicks "Allow" → we copy `.credentials.json`
4. User clicks "Deny" → they use a different provider
5. "Revoke" button removes our copy

The OAuth tokens include a refresh token, so they auto-renew. The user does this once.

## Provider Detection and Setup UX

On first launch, the app scans for installed CLIs:

```
Detected:
✅ Claude Code v2.1.83 (Max — user@email.com)     [Authorize]
✅ OpenCode v1.1.53 (6 free models)                 Ready
✅ Codex CLI v0.116.0 (logged in)                   Ready
✅ Kilo CLI v0.26.1 (10 free models)                Ready
❌ Custom Endpoint                                  [Configure]

Active: Kilo CLI (kilo-auto/free)
```

If nothing is installed:

```
No AI providers detected. Install one to get started:

[Install Kilo CLI]     FREE — 10 models, no API key needed
[Install OpenCode]     FREE — 6 models, no API key needed
[Install Claude Code]  Requires Claude Max/Pro subscription
[Install Codex CLI]    Requires ChatGPT subscription
[Configure Custom]     Any OpenAI-compatible endpoint
```

Kilo and OpenCode are recommended first because they have free models — the user can start chatting immediately with zero cost.

## Performance Results

Tested on Windows 11, all on the same machine, same prompt ("Say hello in one word"):

| Provider | Model | Latency | Cost |
|----------|-------|---------|------|
| Codex CLI | default | 2.6s | ChatGPT subscription |
| OpenCode | minimax-m2.5-free | 4.4s | Free |
| Claude Code | sonnet (isolated) | 5.2s | Max subscription |
| Kilo CLI | kilo-auto/free | 19.9s | Free |

For RAG chat where we're sending 5 sources + a question, add ~1-2 seconds for the larger payload.

## The Registry

All providers are managed by a singleton registry:

```python
from tutorialvault.providers.registry import get_registry

registry = get_registry()

# See what's available
for name, info in registry.all_status().items():
    print(f"{name}: installed={info['installed']}, free={info['free_model_count']}")

# Switch provider
registry.active = "opencode"

# Chat
answer = registry.chat([{"role": "user", "content": "What is box modeling?"}])
```

The registry is exposed via the FastAPI server:

| Endpoint | What it does |
|----------|-------------|
| `GET /api/providers` | List all providers with status, models, free counts |
| `POST /api/providers/active` | Switch the active provider |
| `POST /api/providers/install` | Install a CLI (with permission) |
| `POST /api/providers/authorize` | Copy credentials (Claude Code) |
| `POST /api/providers/test` | Test connection + measure latency |

## What We Learned

1. **Every CLI has a different JSONL format.** There's no standard. You have to read the source code or test empirically.

2. **ANSI escape codes in JSON output are common.** TUI frameworks (Ink, React for terminal) write control sequences even in "machine-readable" modes. Always strip before parsing.

3. **Startup overhead dominates response time.** Claude Code's 60-second startup wasn't the model being slow — it was loading 5 MCP servers and running hook scripts. The actual API call was 2-3 seconds.

4. **OAuth tokens are portable.** Claude Code's credentials file works in any config directory. The isolated config trick exploits this: same auth, zero overhead.

5. **Free models exist and they're good enough.** OpenCode and Kilo both offer free models that handle RAG-grounded Q&A well. Not every query needs Claude Opus.

6. **Anthropic prohibits third-party OAuth.** OpenCode had Claude Max support, Anthropic told them to remove it (v1.3.0). The only legit way to use a Max subscription programmatically is through Claude Code CLI or the Agent SDK. Our isolated config approach works because we're still using Claude Code — just with a clean config.
