"""Claude Code provider — uses the `claude` CLI with the user's existing subscription.

Speed fix: Claude Code normally loads all MCP servers and hooks on every call,
which can take 30-60+ seconds. We solve this by creating an isolated config
directory with ONLY the credentials file — no hooks, no MCP, no plugins.
This brings response time from 60s+ down to ~6s.

The user's main Claude Code setup is never modified.

Flow for new users:
1. Detect `claude` is installed and logged in
2. Ask permission to use their subscription
3. Copy credentials to our isolated config dir
4. All calls use CLAUDE_CONFIG_DIR pointing to our isolated dir

Invocation:
    CLAUDE_CONFIG_DIR=<isolated_dir> claude -p "{prompt}" \
        --output-format json --no-session-persistence --max-turns 1
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from .base import Provider, ProviderModel, ProviderStatus
from ..core.config import get_config

# Where we store the isolated Claude config
_ISOLATED_DIR: Path | None = None


def _get_isolated_dir() -> Path:
    """Get or create the isolated config directory for fast Claude calls."""
    global _ISOLATED_DIR
    if _ISOLATED_DIR is not None:
        return _ISOLATED_DIR

    cfg = get_config()
    data_dir = cfg.resolve_path("store.path").parent
    isolated = data_dir / "claude_isolated"
    isolated.mkdir(parents=True, exist_ok=True)
    _ISOLATED_DIR = isolated
    return isolated


def _find_credentials() -> Path | None:
    """Find the user's Claude Code credentials file."""
    # Check CLAUDE_CONFIG_DIR env var first
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        creds = Path(config_dir) / ".credentials.json"
        if creds.exists():
            return creds

    # Default locations
    home = Path.home()
    for candidate in [
        home / ".claude" / ".credentials.json",
        home / ".claude" / "credentials.json",
    ]:
        if candidate.exists():
            return candidate

    return None


def _is_authorized() -> bool:
    """Check if we've already copied credentials to our isolated dir."""
    isolated = _get_isolated_dir()
    creds = isolated / ".credentials.json"
    return creds.exists()


def _authorize() -> bool:
    """Copy credentials to our isolated config dir. Returns True on success."""
    source = _find_credentials()
    if source is None:
        return False

    isolated = _get_isolated_dir()
    dest = isolated / ".credentials.json"

    try:
        shutil.copy2(str(source), str(dest))
        return True
    except Exception:
        return False


def _revoke():
    """Remove copied credentials."""
    isolated = _get_isolated_dir()
    creds = isolated / ".credentials.json"
    if creds.exists():
        creds.unlink()


class ClaudeCodeProvider(Provider):
    name = "claude_code"
    display_name = "Claude Code"
    install_command = "npm install -g @anthropic-ai/claude-code"

    def _bin(self) -> str | None:
        return shutil.which("claude")

    def _env(self) -> dict:
        """Build env with isolated config dir."""
        env = os.environ.copy()
        env["CLAUDE_CONFIG_DIR"] = str(_get_isolated_dir())
        return env

    def detect(self) -> bool:
        return self._bin() is not None

    def status(self) -> ProviderStatus:
        binary = self._bin()
        if not binary:
            return ProviderStatus(installed=False)

        version = None
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
            version = r.stdout.strip().split("\n")[0] if r.returncode == 0 else None
        except Exception:
            pass

        # Check if user is logged in to Claude Code (their main install)
        authenticated = False
        user = None
        try:
            r = subprocess.run(
                [binary, "auth", "status", "--text"],
                capture_output=True, text=True, timeout=10,
            )
            authenticated = r.returncode == 0
            if authenticated:
                for line in r.stdout.splitlines():
                    if "email" in line.lower():
                        user = line.split(":", 1)[1].strip()
        except Exception:
            pass

        # Check if we have authorization (credentials copied)
        authorized_for_app = _is_authorized()

        models = [
            ProviderModel(id="sonnet", name="Claude Sonnet", context_window=200_000),
            ProviderModel(id="opus", name="Claude Opus", context_window=200_000),
            ProviderModel(id="haiku", name="Claude Haiku", context_window=200_000),
        ]

        error = None
        if authenticated and not authorized_for_app:
            error = "Permission needed — click Authorize to use your Claude subscription"

        return ProviderStatus(
            installed=True,
            authenticated=authenticated and authorized_for_app,
            version=version,
            user=user,
            models=models,
            error=error,
        )

    def authorize(self) -> bool:
        """Copy credentials from user's Claude install to our isolated dir.

        Call this after getting user permission in the UI.
        Returns True on success.
        """
        return _authorize()

    def revoke(self):
        """Remove our copy of credentials."""
        _revoke()

    def install(self) -> bool:
        """Install Claude Code CLI."""
        if not self.install_command:
            return False
        result = subprocess.run(self.install_command, shell=True, capture_output=True, text=True, timeout=120)
        return result.returncode == 0

    def _build_prompt(self, messages: list[dict]) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        return "\n\n".join(parts)

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Claude Code CLI not found")
        if not _is_authorized():
            raise RuntimeError("Claude Code not authorized. Call authorize() first.")

        prompt = self._build_prompt(messages)

        result = subprocess.run(
            [
                binary, "-p", prompt,
                "--output-format", "json",
                "--no-session-persistence",
                "--max-turns", "1",
            ],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace",
            env=self._env(),
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude Code error (exit {result.returncode}): {result.stderr[:300]}")

        # Parse JSON output
        try:
            data = json.loads(result.stdout)
            answer = data.get("result", "")
            if answer:
                return answer
        except json.JSONDecodeError:
            pass

        # Fallback: parse stream-json style if result field is empty
        # (some versions return empty result but have content in assistant messages)
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
                if event.get("type") == "assistant":
                    content = event.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "text" and block.get("text"):
                            return block["text"]
            except json.JSONDecodeError:
                continue

        if result.stdout.strip():
            return result.stdout.strip()

        raise RuntimeError("Claude Code returned empty response")

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Claude Code CLI not found")
        if not _is_authorized():
            raise RuntimeError("Claude Code not authorized. Call authorize() first.")

        prompt = self._build_prompt(messages)

        proc = subprocess.Popen(
            [
                binary, "-p", prompt,
                "--output-format", "stream-json",
                "--verbose",
                "--no-session-persistence",
                "--max-turns", "1",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            env=self._env(),
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "assistant":
                    content = event.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            yield block["text"]
            except json.JSONDecodeError:
                continue

        proc.wait()
