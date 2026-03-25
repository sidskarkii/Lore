"""Embedding — BGE-M3 via ONNX Runtime. No PyTorch required (~50MB vs ~2GB).

Downloads the ONNX model from HuggingFace on first run, then loads it
with onnxruntime for fast CPU/GPU inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import get_config

_session = None
_tokenizer = None


def _model_dir() -> Path:
    """Resolve where ONNX model files are cached."""
    from huggingface_hub import snapshot_download

    cfg = get_config()
    model_name = cfg.get("embedding.model", "BAAI/bge-m3")
    # Download only the ONNX files + tokenizer (skip pytorch bins)
    path = snapshot_download(
        model_name,
        allow_patterns=[
            "onnx/*",
            "tokenizer*",
            "special_tokens_map.json",
            "vocab.txt",
            "sentencepiece*",
        ],
    )
    return Path(path)


def _get_session_and_tokenizer():
    """Lazy-load ONNX session + tokenizer."""
    global _session, _tokenizer
    if _session is not None:
        return _session, _tokenizer

    import onnxruntime as ort
    from tokenizers import Tokenizer

    cfg = get_config()
    device = cfg.embed_device

    print(f"  Loading embedding model (ONNX) on {device}...")
    model_path = _model_dir()

    # Find ONNX file
    onnx_dir = model_path / "onnx"
    onnx_file = onnx_dir / "model.onnx"
    if not onnx_file.exists():
        # Some models use model_optimized.onnx or similar
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if onnx_files:
            onnx_file = onnx_files[0]
        else:
            raise FileNotFoundError(f"No .onnx file found in {onnx_dir}")

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
    if tok_file.exists():
        _tokenizer = Tokenizer.from_file(str(tok_file))
    else:
        # Fallback: try loading from the model dir
        from tokenizers import Tokenizer as Tok
        _tokenizer = Tok.from_pretrained(str(model_path))

    # Enable padding and truncation
    from tokenizers import processors
    _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    _tokenizer.enable_truncation(max_length=8192)

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

    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer.encode_batch(batch)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Some ONNX models expect token_type_ids
        input_names = [inp.name for inp in session.get_inputs()]
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = token_type_ids

        outputs = session.run(None, feeds)
        # BGE models output last_hidden_state as first output
        # Use CLS token (index 0) or mean pooling
        last_hidden = outputs[0]  # (batch, seq_len, dim)

        # Mean pooling over non-padded tokens
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (last_hidden * mask_expanded).sum(axis=1)
        counts = mask_expanded.sum(axis=1).clip(min=1e-9)
        pooled = summed / counts

        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        normalized = pooled / norms

        all_embeddings.append(normalized)

    result = np.concatenate(all_embeddings, axis=0)
    return result.tolist()


def embed_dim() -> int:
    """Return the configured embedding dimension."""
    return get_config().get("embedding.dim", 1024)
