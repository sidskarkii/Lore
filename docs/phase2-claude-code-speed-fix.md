# 60 Seconds to 6: Making Claude Code CLI Usable as a Subprocess

> How we discovered that Claude Code's slow headless mode isn't the model — it's the MCP servers — and fixed it with one environment variable.

## The Setup

We're building a RAG knowledge base app. The user asks a question, we search our vector database for relevant sources, then send the question + sources to an LLM for a grounded answer.

For the LLM, we wanted to use whatever the developer already has. Most AI developers have Claude Code installed with a Max subscription ($100/mo). If we could call `claude -p "prompt"` as a subprocess and parse the response, they'd get Claude inference for free.

## The Problem

```bash
$ time claude -p "Say hello" --output-format json
# ... 62 seconds later ...
{"result":"Hello!","duration_ms":62341,...}
```

62 seconds to say "Hello!" That's not the model being slow — Claude Sonnet responds in 2-3 seconds. Something else is eating 60 seconds.

## What's Actually Happening

We ran `--output-format stream-json --verbose` to see every event:

```json
{"type":"system","subtype":"hook_started","hook_name":"SessionStart:startup"}
{"type":"system","subtype":"hook_started","hook_name":"SessionStart:startup"}
{"type":"system","subtype":"hook_started","hook_name":"SessionStart:startup"}
{"type":"system","subtype":"init","mcp_servers":[
  {"name":"blender","status":"connected"},
  {"name":"houdini","status":"connected"},
  {"name":"tutorialvault","status":"connected"},
  {"name":"claude.ai Gmail","status":"connected"},
  {"name":"unrealClaude","status":"connected"}
]}
```

Before sending the prompt, Claude Code:

1. **Runs all SessionStart hooks** — in our case, the claude-mem plugin that loads conversation history and injects context
2. **Connects to every MCP server** — Blender, Houdini, TutorialVault, Gmail, Unreal Engine — 5 servers, each with its own startup time
3. **Reads CLAUDE.md files** — project instructions, memory files
4. **Loads plugins** — claude-mem, any other installed plugins

All of this makes sense for interactive use — you want your full environment when you're coding. But for a subprocess call where you just need "answer this question," it's 60 seconds of wasted setup.

## Attempt 1: `--bare` Flag

The `--bare` flag skips all auto-discovery:

```bash
$ claude --bare -p "Say hello" --output-format json
{"result":"Not logged in · Please run /login"}
```

Fast! But `--bare` also skips OAuth credential loading. On a Max subscription, authentication is OAuth-based (browser login → token stored in keychain/credential file). No OAuth = no auth = "Not logged in."

The docs confirm this:

> "Bare mode skips OAuth and keychain reads. Anthropic authentication must come from `ANTHROPIC_API_KEY` or an `apiKeyHelper`."

But Max subscribers don't have an `ANTHROPIC_API_KEY`. They have OAuth tokens. Dead end.

## Attempt 2: `CLAUDE_CODE_OAUTH_TOKEN` Environment Variable

The research suggested setting `CLAUDE_CODE_OAUTH_TOKEN` with the token from `~/.claude/.credentials.json`:

```bash
$ CLAUDE_CODE_OAUTH_TOKEN='sk-ant-oat01-...' claude --bare -p "Say hello"
{"result":"Not logged in · Please run /login"}
```

Still doesn't work. We tried the full JSON blob with refresh token. We tried `ANTHROPIC_AUTH_TOKEN`. Nothing.

## Attempt 3: `--strict-mcp-config --mcp-config '{}'`

Skip MCP servers but keep OAuth:

```bash
$ claude -p "Say hello" --strict-mcp-config --mcp-config '{}'
# ... 45 seconds later, still loading hooks ...
```

This skips MCP servers but hooks and plugins still load. On our machine, the claude-mem plugin's SessionStart hook alone takes 10+ seconds.

## Attempt 4: Claude Agent SDK

The Python SDK (`pip install claude-agent-sdk`) uses `ClaudeSDKClient` with a persistent connection:

```python
client = ClaudeSDKClient(options=ClaudeAgentOptions(
    allowed_tools=[], mcp_servers={}, plugins=[], setting_sources=[]
))
await client.connect(prompt="Say hello")  # connects in 1s!
async for msg in client.receive_messages():  # hangs for 60s+
    ...
```

Connected in 1 second, but `receive_messages()` hung — the underlying Claude Code process was still loading hooks. The SDK options weren't propagating to prevent it.

## The Fix: Isolated Config Directory

Claude Code reads all its configuration from a directory (default: `~/.claude/`). The `CLAUDE_CONFIG_DIR` environment variable overrides this.

The insight: **create an empty config directory with only the credentials file**. No `.mcp.json`, no hooks, no plugins, no CLAUDE.md. Just auth.

```bash
# Create isolated config
$ mkdir /tmp/claude-isolated
$ cp ~/.claude/.credentials.json /tmp/claude-isolated/

# Use it
$ CLAUDE_CONFIG_DIR=/tmp/claude-isolated claude -p "Say hello" --output-format json
{"result":"Hello!","duration_ms":5923,...}
```

**6 seconds.** Down from 62. The OAuth tokens in `.credentials.json` work fine — Claude Code reads them, authenticates against the Max subscription, and responds.

No MCP servers load (no `.mcp.json` to find). No hooks run (no hook configs). No plugins activate (no plugin directory). No CLAUDE.md gets read (empty directory). Just auth → prompt → response.

## The Implementation

In our provider, we copy the credentials on first authorization:

```python
def authorize(self) -> bool:
    source = find_credentials()  # ~/.claude/.credentials.json
    if source is None:
        return False
    isolated = get_isolated_dir()  # <our_app>/data/claude_isolated/
    shutil.copy2(str(source), str(isolated / ".credentials.json"))
    return True

def chat(self, messages, model=None):
    env = os.environ.copy()
    env["CLAUDE_CONFIG_DIR"] = str(get_isolated_dir())

    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json",
         "--no-session-persistence", "--max-turns", "1"],
        env=env, capture_output=True, text=True, timeout=60,
    )
    return json.loads(result.stdout)["result"]
```

The user's actual `~/.claude/` is never touched. Their MCP servers, hooks, plugins, conversation history — all untouched.

## Why This Works

The credentials file contains:

```json
{
  "claudeAiOauth": {
    "accessToken": "sk-ant-oat01-...",
    "refreshToken": "sk-ant-ort01-...",
    "expiresAt": 1774589122999,
    "scopes": ["user:inference", "user:profile", ...],
    "subscriptionType": "max"
  }
}
```

It has everything needed to authenticate: access token, refresh token (for auto-renewal), and subscription info. Claude Code just needs to find this file — it doesn't care what else is or isn't in the config directory.

## Before and After

| Metric | Before (full config) | After (isolated) |
|--------|---------------------|-------------------|
| MCP servers loaded | 5 | 0 |
| Hooks executed | 3+ | 0 |
| Plugins loaded | 1 | 0 |
| Total latency | 62s | 5.9s |
| API call time | ~3s | ~3s |
| Overhead | ~59s | ~3s |

The API call itself is identical — same model, same quality, same tokens. We just removed 59 seconds of unnecessary startup.

## Caveats

1. **Token expiry.** The OAuth access token expires. The credentials file includes a refresh token that Claude Code uses to auto-renew. If the refresh token expires (months later), the user needs to re-authorize.

2. **Anthropic's terms.** We're still using the Claude Code CLI — just with a different config directory. This is the documented, supported way to customize Claude Code's behavior. We're not bypassing OAuth or extracting tokens for use with a different client.

3. **Per-call process.** Each `claude -p` call spawns a new process, authenticates, responds, and exits. There's no persistent connection. The 6-second overhead includes process startup, credential loading, and HTTP connection to the API. For a chat app where users wait for responses anyway, this is acceptable.

4. **Not parallelizable for free.** You can run multiple `claude -p` calls in parallel, but each one counts against your rate limit. Max subscribers have generous limits, but you can still hit them with aggressive parallel calls.

## For Other CLI Tools

This pattern — isolated config dir — might work for other CLI tools that have slow startup due to plugin/extension loading. The general approach:

1. Find where the CLI reads its config
2. Check if there's an env var to override the config path
3. Create a minimal config with only authentication
4. Point the env var at your minimal config

We didn't need this for OpenCode, Codex, or Kilo — they were already fast enough (2-5 seconds) without MCP servers loading. But if any of them develop the same problem as they add more features, the same trick should work.
