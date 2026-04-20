"""Embedding — EmbeddingGemma-300M via ONNX Runtime. No PyTorch required.

Downloads the q4 quantized ONNX model (~188 MB) from HuggingFace on first
run, then loads it with onnxruntime for fast CPU/GPU inference.

Model: google/embeddinggemma-300m (ONNX export by onnx-community)
  - #1 on MTEB for models under 500M params
  - 768-dim dense vectors
  - 100+ languages
  - q4 quantization: identical accuracy, 6x smaller than fp32
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .config import get_config

_session = None
_tokenizer = None

# Default ONNX variant filenames (configurable in config.yaml)
_VARIANT_FILES = {
    "fp32":  ("model.onnx",           "model.onnx_data"),
    "q8":    ("model_quantized.onnx", "model_quantized.onnx_data"),
    "q4":    ("model_q4.onnx",        "model_q4.onnx_data"),
    "q4f16": ("model_q4f16.onnx",     "model_q4f16.onnx_data"),
}


def _download_model() -> Path:
    """Download ONNX model files from HuggingFace (only the selected variant)."""
    cfg = get_config()
    repo = cfg.get("embedding.model", "onnx-community/embeddinggemma-300m-ONNX")
    variant = cfg.get("embedding.variant", "q4")

    onnx_name, data_name = _VARIANT_FILES.get(variant, _VARIANT_FILES["q4"])

    # Download only the specific variant + tokenizer (not the whole repo)
    model_dir = Path(hf_hub_download(repo, f"onnx/{onnx_name}")).parent.parent
    hf_hub_download(repo, f"onnx/{data_name}")
    # Tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        try:
            hf_hub_download(repo, fname)
        except Exception:
            pass

    return model_dir


def _get_session_and_tokenizer():
    """Lazy-load ONNX session + tokenizer."""
    global _session, _tokenizer
    if _session is not None:
        return _session, _tokenizer

    cfg = get_config()
    device = cfg.embed_device
    variant = cfg.get("embedding.variant", "q4")
    onnx_name, _ = _VARIANT_FILES.get(variant, _VARIANT_FILES["q4"])

    print(f"  Loading embedding model ({variant}) on {device}...")
    model_path = _download_model()

    onnx_file = model_path / "onnx" / onnx_name
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

    # Set up providers
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session = ort.InferenceSession(str(onnx_file), sess_opts, providers=providers)

    # Load tokenizer
    tok_file = model_path / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {model_path}")
    _tokenizer = Tokenizer.from_file(str(tok_file))
    _tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    _tokenizer.enable_truncation(max_length=2048)

    return _session, _tokenizer


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode texts into normalized vectors using ONNX runtime.

    Returns a list of float lists, one per input text.
    """
    if not texts:
        return []

    cfg = get_config()
    batch_size = cfg.get("embedding.batch_size", 32)
    session, tokenizer = _get_session_and_tokenizer()
    input_names = [inp.name for inp in session.get_inputs()]

    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer.encode_batch(batch)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = session.run(None, feeds)
        emb = outputs[0]

        # Handle both (batch, dim) and (batch, seq_len, dim) output shapes
        if emb.ndim == 3:
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            emb = (emb * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1).clip(min=1e-9)

        # L2 normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        all_embeddings.append(emb / norms)

    result = np.concatenate(all_embeddings, axis=0)
    return result.tolist()


def embed_dim() -> int:
    """Return the configured embedding dimension."""
    return get_config().get("embedding.dim", 768)
