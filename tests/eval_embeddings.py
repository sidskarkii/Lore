"""Eval: EmbeddingGemma-300M at all quantization levels (ONNX, no PyTorch)

Variants tested:
- fp32  (~1.2 GB)
- fp16  (~617 MB)   — NOTE: Gemma warns fp16 activations unsupported, may fail
- q8    (~309 MB)   — model_quantized.onnx
- q4    (~197 MB)
- q4f16 (~175 MB)   — smallest
"""

import time
import numpy as np

# ── Test data ────────────────────────────────────────────────────────────
CORPUS = [
    "To enter edit mode in Blender, select the object and press Tab",
    "The subdivision surface modifier smooths geometry by subdividing faces",
    "Weight painting controls how much a bone influences nearby vertices",
    "In Houdini, SOPs are surface operators that modify geometry at the object level",
    "UV unwrapping maps 3D surface coordinates to a 2D texture space",
    "Keyframes define the start and end points of an animation transition",
    "The node editor in Blender connects shader nodes to create materials",
    "Rigging involves creating a skeleton of bones inside a mesh for animation",
    "Sculpting mode uses brushes to push, pull, and smooth mesh geometry",
    "Particle systems simulate hair, fur, rain, and other particle effects",
    "Boolean operations combine or subtract meshes using union, intersection, or difference",
    "Motion capture data can be retargeted onto different character rigs",
    "Normal maps fake surface detail by altering how light bounces off faces",
    "The graph editor displays animation curves for fine-tuning keyframe interpolation",
    "Lattice deformation wraps a control grid around an object for broad shape changes",
]

QUERIES = [
    "how do I enter edit mode",
    "what is weight painting used for",
    "how to create materials with nodes",
    "sculpting tools in blender",
    "how does motion capture work with rigs",
]

EXPECTED = [0, 2, 6, 8, 11]

# All variants to test
VARIANTS = [
    ("fp32",  "model.onnx",              "model.onnx_data"),
    ("q8",    "model_quantized.onnx",    "model_quantized.onnx_data"),
    ("q4",    "model_q4.onnx",           "model_q4.onnx_data"),
    ("q4f16", "model_q4f16.onnx",        "model_q4f16.onnx_data"),
    ("fp16",  "model_fp16.onnx",         "model_fp16.onnx_data"),
]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def download_all():
    """Download all ONNX variants + tokenizer in one shot."""
    from huggingface_hub import snapshot_download
    from pathlib import Path

    print("Downloading all EmbeddingGemma-300M ONNX variants...")
    t0 = time.time()
    model_path = Path(snapshot_download(
        "onnx-community/embeddinggemma-300m-ONNX",
        allow_patterns=[
            "onnx/*.onnx",
            "onnx/*.onnx_data",
            "tokenizer*",
            "special_tokens_map.json",
        ],
    ))
    print(f"  Downloaded in {time.time()-t0:.1f}s\n")
    return model_path


def make_embed_fn(model_path, onnx_name, data_name):
    """Create an embed function for a specific ONNX variant."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    onnx_file = model_path / "onnx" / onnx_name
    data_file = model_path / "onnx" / data_name

    if not onnx_file.exists():
        return None, 0, 0

    # Compute size
    size_mb = onnx_file.stat().st_size / 1024 / 1024
    if data_file.exists():
        size_mb += data_file.stat().st_size / 1024 / 1024

    # Load session
    t0 = time.time()
    try:
        sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"    FAILED to load: {e}")
        return None, size_mb, 0
    load_time = time.time() - t0

    # Load tokenizer (shared across variants)
    tok_file = model_path / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_file))
    tok.enable_padding(pad_id=0, pad_token="<pad>")
    tok.enable_truncation(max_length=2048)

    input_names = [inp.name for inp in sess.get_inputs()]

    def embed(texts):
        encoded = tok.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
        outputs = sess.run(None, feeds)
        emb = outputs[0]
        if emb.ndim == 3:
            mask_exp = attention_mask[:, :, np.newaxis].astype(np.float32)
            pooled = (emb * mask_exp).sum(axis=1) / mask_exp.sum(axis=1).clip(min=1e-9)
            emb = pooled
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        return (emb / norms).tolist()

    return embed, size_mb, load_time


def eval_variant(name, embed_fn, size_mb, load_time):
    """Run eval for a variant."""
    # Embed corpus
    t0 = time.time()
    corpus_vecs = embed_fn(CORPUS)
    corpus_time = time.time() - t0

    # Embed queries
    t0 = time.time()
    query_vecs = embed_fn(QUERIES)
    query_time = time.time() - t0

    # Retrieval accuracy
    hits = 0
    for i, (q, q_vec, expected_idx) in enumerate(zip(QUERIES, query_vecs, EXPECTED)):
        sims = [cosine_sim(q_vec, c_vec) for c_vec in corpus_vecs]
        top_idx = int(np.argmax(sims))
        if top_idx == expected_idx:
            hits += 1

    accuracy = hits / len(QUERIES) * 100
    total_texts = len(CORPUS) + len(QUERIES)
    avg_ms = (corpus_time + query_time) / total_texts * 1000

    return {
        "name": name,
        "size_mb": size_mb,
        "load_time": load_time,
        "accuracy": accuracy,
        "hits": hits,
        "avg_ms": avg_ms,
        "dim": len(corpus_vecs[0]),
    }


if __name__ == "__main__":
    model_path = download_all()

    results = []
    for variant_name, onnx_name, data_name in VARIANTS:
        print(f"--- {variant_name} ---")
        embed_fn, size_mb, load_time = make_embed_fn(model_path, onnx_name, data_name)
        if embed_fn is None:
            print(f"  SKIPPED (file missing or load failed)\n")
            continue
        print(f"  Size: {size_mb:.0f} MB | Load: {load_time:.2f}s")
        try:
            r = eval_variant(variant_name, embed_fn, size_mb, load_time)
            results.append(r)
            print(f"  Accuracy: {r['hits']}/{len(QUERIES)} | Speed: {r['avg_ms']:.1f}ms/text | Dim: {r['dim']}")
        except Exception as e:
            print(f"  EVAL FAILED: {e}")
        print()

    # Summary table
    print(f"\n{'='*70}")
    print(f"  COMPARISON — EmbeddingGemma-300M ONNX Variants")
    print(f"{'='*70}")
    print(f"  {'Variant':<10} {'Size':>8} {'Load':>8} {'Accuracy':>10} {'Speed':>12} {'Dim':>6}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*6}")
    for r in results:
        print(
            f"  {r['name']:<10} {r['size_mb']:>7.0f}M {r['load_time']:>7.2f}s "
            f"{r['hits']}/{len(QUERIES)} ({r['accuracy']:.0f}%) "
            f"{r['avg_ms']:>9.1f}ms/t {r['dim']:>6}"
        )
