"""Kilo CLI provider — 10 free models, no API key needed.

Invocation (non-interactive):
    kilo --auto --json --nosplash --yolo -M "kilo-auto/free" "{prompt}"

Output format (--json): JSONL mixed with ANSI escape codes.
    Lines contain ANSI control sequences that must be stripped.
    Extract JSON with regex, then parse.

    Answer events: {"say":"text","partial":false,"content":"the answer"}
    Reasoning:     {"say":"reasoning","partial":true,"content":"..."}  — ignore
    API events:    {"say":"api_req_started",...}  — ignore

Auth: Auto-authenticated on install. `kilo auth` to manage.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Iterator

from .base import Provider, ProviderModel, ProviderStatus
from ..core.config import get_config

_FREE_MODELS = [
    ProviderModel(id="kilo-auto/free", name="Kilo Auto Free", free=True, context_window=204_800),
    ProviderModel(id="xiaomi/mimo-v2-pro:free", name="MiMo V2 Pro (free)", free=True, context_window=1_048_576),
    ProviderModel(id="nvidia/nemotron-3-super-120b-a12b:free", name="Nemotron 3 Super (free)", free=True, context_window=262_144),
    ProviderModel(id="minimax/minimax-m2.5:free", name="MiniMax M2.5 (free)", free=True, context_window=204_800),
    ProviderModel(id="xiaomi/mimo-v2-omni:free", name="MiMo V2 Omni (free)", free=True, context_window=262_144),
    ProviderModel(id="x-ai/grok-code-fast-1:optimized:free", name="Grok Code Fast (free)", free=True, context_window=256_000),
    ProviderModel(id="stepfun/step-3.5-flash:free", name="Step 3.5 Flash (free)", free=True, context_window=256_000),
    ProviderModel(id="arcee-ai/trinity-large-preview:free", name="Trinity Large (free)", free=True, context_window=131_000),
    ProviderModel(id="corethink:free", name="CoreThink (free)", free=True, context_window=78_000),
    ProviderModel(id="openrouter/free", name="OpenRouter Free", free=True, context_window=200_000),
]

# Regex to strip ANSI escape codes
_ANSI_RE = re.compile(r'(\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b\]0;[^\x07]*)')


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub('', text)


def _extract_json_objects(text: str) -> list[dict]:
    """Extract JSON objects from text that may contain ANSI codes."""
    cleaned = _strip_ansi(text)
    results = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            # Try to extract JSON from within the line
            match = re.search(r'\{.*\}', line)
            if match:
                try:
                    results.append(json.loads(match.group()))
                except json.JSONDecodeError:
                    continue
    return results


class KiloProvider(Provider):
    name = "kilo"
    display_name = "Kilo CLI"
    install_command = "npm install -g @kilocode/cli"

    def _bin(self) -> str | None:
        return shutil.which("kilo") or shutil.which("kilocode")

    def detect(self) -> bool:
        return self._bin() is not None

    def status(self) -> ProviderStatus:
        binary = self._bin()
        if not binary:
            return ProviderStatus(installed=False)

        version = None
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
            version = r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            pass

        return ProviderStatus(
            installed=True, authenticated=True,
            version=version, models=list(_FREE_MODELS),
        )

    def _parse_output(self, output: str) -> str:
        """Extract final answer from Kilo output.

        Answer is the last non-partial text event:
        {"say":"text","partial":false,"content":"Hello"}
        """
        events = _extract_json_objects(output)

        # Prefer completion_result, fall back to last non-partial text
        for event in reversed(events):
            if event.get("say") == "completion_result":
                return event.get("content", "").strip()

        for event in reversed(events):
            if event.get("say") == "text" and not event.get("partial", True):
                content = event.get("content", "")
                # Skip if it's the original user prompt echoed back
                if content and event.get("source") != "user":
                    return content.strip()

        return ""

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Kilo CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.kilo.model", "kilo-auto/free")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        result = subprocess.run(
            [
                binary,
                "--auto", "--json", "--nosplash", "--yolo",
                "-M", model,
                "--timeout", "300",
                prompt,
            ],
            capture_output=True, text=True, timeout=360,
            encoding="utf-8", errors="replace",
        )

        if result.returncode != 0:
            raise RuntimeError(f"Kilo error (exit {result.returncode}): {result.stderr[:300]}")

        answer = self._parse_output(result.stdout)
        if not answer:
            raise RuntimeError("Kilo returned empty response")
        return answer

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        binary = self._bin()
        if not binary:
            raise RuntimeError("Kilo CLI not found")

        cfg = get_config()
        model = model or cfg.get("provider.kilo.model", "kilo-auto/free")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        proc = subprocess.Popen(
            [
                binary,
                "--auto", "--json", "--nosplash", "--yolo",
                "-M", model,
                "--timeout", "300",
                prompt,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )

        yielded = 0
        for line in proc.stdout:
            cleaned = _strip_ansi(line).strip()
            if not cleaned or not cleaned.startswith('{'):
                continue
            try:
                event = json.loads(cleaned)
                if event.get("say") == "text":
                    content = event.get("content", "")
                    if content:
                        yielded += 1
                        yield content
            except json.JSONDecodeError:
                continue

        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read(500) if proc.stderr else ""
            print(f"  [kilo] stream exit={proc.returncode} stderr={stderr!r}")
            if yielded == 0:
                raise RuntimeError(f"Kilo stream failed (exit {proc.returncode}): {stderr[:200]}")
        else:
            print(f"  [kilo] stream done, {yielded} chunks")
