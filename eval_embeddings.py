"""Quick eval: BGE-M3 (ONNX) vs EmbeddingGemma-300M (ONNX)

Compares:
- Download size
- Load time
- Embedding speed
- Retrieval quality on test queries
- Dependencies (torch-free?)
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

# Expected best match index for each query (0-indexed into CORPUS)
EXPECTED = [0, 2, 6, 8, 11]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def eval_model(name, embed_fn):
    """Run eval for a given embedding function."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Embed corpus
    t0 = time.time()
    corpus_vecs = embed_fn(CORPUS)
    corpus_time = time.time() - t0
    print(f"  Corpus ({len(CORPUS)} texts): {corpus_time:.2f}s")
    print(f"  Vector dim: {len(corpus_vecs[0])}")

    # Embed queries
    t0 = time.time()
    query_vecs = embed_fn(QUERIES)
    query_time = time.time() - t0
    print(f"  Queries ({len(QUERIES)} texts): {query_time:.2f}s")

    # Retrieval accuracy
    hits = 0
    for i, (q, q_vec, expected_idx) in enumerate(zip(QUERIES, query_vecs, EXPECTED)):
        sims = [cosine_sim(q_vec, c_vec) for c_vec in corpus_vecs]
        top_idx = int(np.argmax(sims))
        top_score = sims[top_idx]
        match = "HIT" if top_idx == expected_idx else "MISS"
        if match == "HIT":
            hits += 1
        print(f"  Q: '{q}'")
        print(f"    Top: [{top_idx}] {CORPUS[top_idx][:60]}... (sim={top_score:.4f}) [{match}]")

    accuracy = hits / len(QUERIES) * 100
    print(f"\n  Accuracy: {hits}/{len(QUERIES)} = {accuracy:.0f}%")
    print(f"  Avg speed: {(corpus_time + query_time) / (len(CORPUS) + len(QUERIES)) * 1000:.1f}ms/text")

    return {"accuracy": accuracy, "corpus_time": corpus_time, "query_time": query_time, "dim": len(corpus_vecs[0])}


# ── BGE-M3 via ONNX ─────────────────────────────────────────────────────

def setup_bge_m3():
    from huggingface_hub import snapshot_download
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from pathlib import Path

    print("\n--- Downloading BGE-M3 ONNX ---")
    t0 = time.time()
    model_path = Path(snapshot_download(
        "aapot/bge-m3-onnx",
        allow_patterns=["*.onnx", "tokenizer*", "special_tokens_map.json", "sentencepiece*"],
    ))
    print(f"  Downloaded in {time.time()-t0:.1f}s")

    onnx_file = model_path / "model.onnx"
    if not onnx_file.exists():
        for f in model_path.rglob("*.onnx"):
            onnx_file = f
            break

    print(f"  ONNX file: {onnx_file}")
    print(f"  Size: {onnx_file.stat().st_size / 1024 / 1024:.0f} MB")

    t0 = time.time()
    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    load_time = time.time() - t0
    print(f"  Session loaded in {load_time:.2f}s")

    tok_file = model_path / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_file))
    tok.enable_padding(pad_id=0, pad_token="<pad>")
    tok.enable_truncation(max_length=8192)

    input_names = [inp.name for inp in sess.get_inputs()]

    def embed(texts):
        encoded = tok.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
        outputs = sess.run(None, feeds)
        last_hidden = outputs[0]
        # Mean pooling
        mask_exp = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (last_hidden * mask_exp).sum(axis=1) / mask_exp.sum(axis=1).clip(min=1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).tolist()

    return embed


# ── EmbeddingGemma-300M via ONNX ────────────────────────────────────────

def setup_gemma():
    from huggingface_hub import hf_hub_download, snapshot_download
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from pathlib import Path

    print("\n--- Downloading EmbeddingGemma-300M ONNX ---")
    t0 = time.time()
    model_path = Path(snapshot_download(
        "onnx-community/embeddinggemma-300m-ONNX",
        allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data", "tokenizer*", "special_tokens_map.json", "sentencepiece*"],
    ))
    print(f"  Downloaded in {time.time()-t0:.1f}s")

    onnx_file = model_path / "onnx" / "model.onnx"
    if not onnx_file.exists():
        for f in model_path.rglob("*.onnx"):
            onnx_file = f
            break

    print(f"  ONNX file: {onnx_file}")
    # Size might be split across model.onnx + model.onnx_data
    total_size = sum(f.stat().st_size for f in onnx_file.parent.glob("model.onnx*"))
    print(f"  Size: {total_size / 1024 / 1024:.0f} MB")

    t0 = time.time()
    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    load_time = time.time() - t0
    print(f"  Session loaded in {load_time:.2f}s")

    tok_file = model_path / "tokenizer.json"
    if tok_file.exists():
        tok = Tokenizer.from_file(str(tok_file))
    else:
        # Gemma might use sentencepiece directly
        print("  WARNING: no tokenizer.json, trying sentencepiece...")
        raise FileNotFoundError("No tokenizer.json found for EmbeddingGemma")

    tok.enable_padding(pad_id=0, pad_token="<pad>")
    tok.enable_truncation(max_length=2048)

    input_names = [inp.name for inp in sess.get_inputs()]
    print(f"  ONNX inputs: {input_names}")

    def embed(texts):
        encoded = tok.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
        outputs = sess.run(None, feeds)
        # EmbeddingGemma outputs embeddings directly (or last_hidden_state)
        emb = outputs[0]
        if emb.ndim == 3:
            # Mean pooling if we got hidden states
            mask_exp = attention_mask[:, :, np.newaxis].astype(np.float32)
            pooled = (emb * mask_exp).sum(axis=1) / mask_exp.sum(axis=1).clip(min=1e-9)
            emb = pooled
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        return (emb / norms).tolist()

    return embed


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    try:
        embed_bge = setup_bge_m3()
        results["BGE-M3 (ONNX)"] = eval_model("BGE-M3 (ONNX, 1024-dim)", embed_bge)
    except Exception as e:
        print(f"\nBGE-M3 FAILED: {e}")

    try:
        embed_gemma = setup_gemma()
        results["EmbeddingGemma-300M (ONNX)"] = eval_model("EmbeddingGemma-300M (ONNX, 768-dim)", embed_gemma)
    except Exception as e:
        print(f"\nEmbeddingGemma FAILED: {e}")

    if len(results) == 2:
        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        for name, r in results.items():
            print(f"  {name}:")
            print(f"    Accuracy:    {r['accuracy']:.0f}%")
            print(f"    Dim:         {r['dim']}")
            print(f"    Corpus time: {r['corpus_time']:.2f}s")
            print(f"    Query time:  {r['query_time']:.2f}s")
