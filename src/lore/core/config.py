"""Configuration loader — single source of truth for all settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DATA_DIR = Path(os.environ.get("LORE_DATA_DIR", "")) if os.environ.get("LORE_DATA_DIR") else Path.home() / ".lore"

_DEFAULTS = {
    "server": {
        "host": "127.0.0.1",
        "port": 52105,
    },
    "store": {
        "path": "store",
        "table": "chunks",
        "summaries_table": "episode_summaries",
    },
    "embedding": {
        "model": "onnx-community/embeddinggemma-300m-ONNX",
        "variant": "q4",
        "dim": 768,
        "device": "auto",
        "batch_size": 32,
    },
    "chunking": {
        "target_sec": 90,
        "overlap_sec": 15,
    },
    "search": {
        "reranker_model": "ms-marco-MiniLM-L-12-v2",
        "parent_window_sec": 150,
        "candidate_count": 30,
        "rrf_k": 60,
        "multi_hop_max_queries": 4,
        "multi_hop_relevance_threshold": 0.1,
    },
    "transcription": {
        "model": "whisper-medium",
        "language": "en",
    },
    "provider": {
        "active": "custom",
    },
}

#recursively merges two dicts. If both sides have a nested dict for the same key, it recurses in. Otherwise the override wins.
def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursing into nested dicts."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


class Config:
    """Loads config.yaml and provides typed access to settings."""

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            config_path = self._find_config()
        self._path = Path(config_path) if config_path else None
        self._root = self._path.parent if self._path else Path.cwd()

        file_cfg = {}
        if self._path and self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}

        # Merge config.local.yaml on top if present (gitignored, for secrets)
        local_cfg = {}
        if self._path:
            local_path = self._path.parent / "config.local.yaml"
            if local_path.exists():
                with open(local_path, encoding="utf-8") as f:
                    local_cfg = yaml.safe_load(f) or {}

        self._data = _deep_merge(_deep_merge(_DEFAULTS, file_cfg), local_cfg)

    @staticmethod
    def _find_config() -> Path | None:
        """Find config.yaml — check package root first, then walk up from cwd."""
        pkg_root = Path(__file__).resolve().parents[3]
        pkg_config = pkg_root / "config.yaml"
        if pkg_config.exists():
            return pkg_config

        cur = Path.cwd()
        for _ in range(10):
            candidate = cur / "config.yaml"
            if candidate.exists():
                return candidate
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
        return None

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Get a value by dot-separated path: cfg.get('embedding.model')."""
        keys = dotpath.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def resolve_path(self, dotpath: str) -> Path:
        """Get a path value, resolving relative paths against data dir."""
        raw = self.get(dotpath, "")
        p = Path(raw)
        if p.is_absolute():
            return p
        return self.data_dir / p

    @property
    def data_dir(self) -> Path:
        """Global data directory (~/.lore/ or LORE_DATA_DIR override)."""
        d = _DATA_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def archive_dir(self) -> Path:
        """Source-segregated archive directory."""
        d = self.data_dir / "archive"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def project_root(self) -> Path:
        return self._root

    @property
    def embed_device(self) -> str:
        dev = self.get("embedding.device", "auto")
        if dev == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return dev


# Module-level singleton — import and use directly
_cfg: Config | None = None


def get_config(config_path: str | Path | None = None) -> Config:
    """Get or create the global config singleton."""
    global _cfg
    if _cfg is None or config_path is not None:
        _cfg = Config(config_path)
    return _cfg
