"""Eval rerankers v2 — FlashRank vs GTE-multilingual vs Qwen3-Reranker on real data.

Uses old TutorialVault LanceDB (211K chunks) for BM25 candidate retrieval,
then each reranker re-scores the same candidates.
"""

import time
import numpy as np
import lancedb

OLD_DB = "F:/TutorialVault/.lancedb"

QUERIES = [
    "how to rig a character skeleton in blender",
    "weight painting bone influence",
    "houdini vop noise pattern",
    "sculpting detailed wrinkles on a face",
    "how to retopologize a high poly mesh",
    "particle system for hair and fur",
    "shader nodes for realistic skin material",
    "animation keyframe graph editor curves",
]

N_CANDIDATES = 20
N_SHOW = 5


def get_bm25_candidates(query, n=20):
    db = lancedb.connect(OLD_DB)
    tbl = db.open_table("tutorials")
    try:
        return tbl.search(query, query_type="fts").limit(n).to_pandas().to_dict("records")
    except Exception as e:
        print(f"  FTS error: {e}")
        return []


def fmt_result(rank, c, score=None):
    src = f"{c.get('tutorial_display','')} ep{c.get('episode_num',0):02d}"
    ts = c.get("timestamp", "?")
    score_str = f"score={score:.4f} | " if score is not None else ""
    text = c.get("text", c.get("_text", ""))[:120]
    return f"    [{rank}] {score_str}{src} @ {ts}\n        {text}..."


# ── FlashRank rerankers ──────────────────────────────────────────────────

def test_flashrank(model_name, display_name):
    from flashrank import Ranker, RerankRequest

    print(f"\n  Loading {display_name}...")
    t0 = time.time()
    ranker = Ranker(model_name=model_name)
    print(f"  Loaded in {time.time()-t0:.2f}s")

    times = []
    for query in QUERIES:
        candidates = get_bm25_candidates(query, N_CANDIDATES)
        if not candidates:
            continue
        passages = [{"id": str(i), "text": c["text"][:500]} for i, c in enumerate(candidates)]

        t0 = time.time()
        reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        dt = time.time() - t0
        times.append(dt)

        print(f"\n  Q: \"{query}\"  ({dt*1000:.0f}ms)")
        for j, r in enumerate(reranked[:N_SHOW]):
            orig_idx = int(r["id"])
            print(fmt_result(j+1, candidates[orig_idx], r["score"]))

    avg = np.mean(times) * 1000
    print(f"\n  >> {display_name}: avg {avg:.0f}ms/query")
    return avg


# ── GTE-multilingual-reranker-base (ONNX) ────────────────────────────────

def test_gte_onnx():
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    import onnxruntime as ort
    from tokenizers import Tokenizer

    display = "GTE-multilingual-reranker-base int8 (341MB)"
    print(f"\n  Loading {display}...")

    repo = "onnx-community/gte-multilingual-reranker-base"

    t0 = time.time()
    onnx_path = hf_hub_download(repo, "onnx/model_int8.onnx")
    tok_path = Path(hf_hub_download(repo, "tokenizer.json"))
    print(f"  Downloaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"  Session loaded in {time.time()-t0:.2f}s")

    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_padding(pad_id=0, pad_token="<pad>")
    tok.enable_truncation(max_length=512)

    input_names = [i.name for i in sess.get_inputs()]
    print(f"  ONNX inputs: {input_names}")

    def rerank(query, texts):
        # Cross-encoder: encode (query, text) pairs
        pairs = [f"{query} [SEP] {t}" for t in texts]
        encoded = tok.encode_batch(pairs)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(ids, dtype=np.int64)
        out = sess.run(None, feeds)
        # Cross-encoder outputs logits — take the relevance score
        logits = out[0]
        if logits.ndim == 2 and logits.shape[1] == 1:
            return logits.flatten().tolist()
        elif logits.ndim == 2:
            return logits[:, 0].tolist()  # first class = relevant
        return logits.flatten().tolist()

    times = []
    for query in QUERIES:
        candidates = get_bm25_candidates(query, N_CANDIDATES)
        if not candidates:
            continue
        texts = [c["text"][:500] for c in candidates]

        t0 = time.time()
        scores = rerank(query, texts)
        dt = time.time() - t0
        times.append(dt)

        ranked = sorted(zip(scores, range(len(candidates))), reverse=True)
        print(f"\n  Q: \"{query}\"  ({dt*1000:.0f}ms)")
        for j, (score, idx) in enumerate(ranked[:N_SHOW]):
            print(fmt_result(j+1, candidates[idx], score))

    avg = np.mean(times) * 1000
    print(f"\n  >> {display}: avg {avg:.0f}ms/query")
    return avg


# ── Qwen3-Reranker-0.6B seq-cls (ONNX) ───────────────────────────────────

def test_qwen3_onnx():
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    import onnxruntime as ort
    from tokenizers import Tokenizer

    display = "Qwen3-Reranker-0.6B seq-cls (1.19GB)"
    print(f"\n  Loading {display}...")

    repo = "zhiqing/Qwen3-Reranker-0.6B-seq-cls-ONNX"

    t0 = time.time()
    onnx_path = hf_hub_download(repo, "model.onnx")
    tok_path = Path(hf_hub_download(repo, "tokenizer.json"))
    print(f"  Downloaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    load_time = time.time() - t0
    print(f"  Session loaded in {load_time:.2f}s")

    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_padding(pad_id=0, pad_token="<|endoftext|>")
    tok.enable_truncation(max_length=512)

    input_names = [i.name for i in sess.get_inputs()]
    print(f"  ONNX inputs: {input_names}")

    def rerank(query, texts):
        # Qwen3-Reranker uses instruct format for scoring
        pairs = [f"Query: {query}\nDocument: {t}\nIs this document relevant?" for t in texts]
        encoded = tok.encode_batch(pairs)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "position_ids" in input_names:
            pos = np.broadcast_to(np.arange(ids.shape[1])[np.newaxis, :], ids.shape).astype(np.int64)
            feeds["position_ids"] = pos
        out = sess.run(None, feeds)
        logits = out[0]
        if logits.ndim == 2 and logits.shape[1] >= 2:
            # seq-cls: logits[:, 1] = "yes" score
            return logits[:, 1].tolist()
        elif logits.ndim == 2 and logits.shape[1] == 1:
            return logits.flatten().tolist()
        return logits.flatten().tolist()

    times = []
    for query in QUERIES:
        candidates = get_bm25_candidates(query, N_CANDIDATES)
        if not candidates:
            continue
        texts = [c["text"][:500] for c in candidates]

        t0 = time.time()
        scores = rerank(query, texts)
        dt = time.time() - t0
        times.append(dt)

        ranked = sorted(zip(scores, range(len(candidates))), reverse=True)
        print(f"\n  Q: \"{query}\"  ({dt*1000:.0f}ms)")
        for j, (score, idx) in enumerate(ranked[:N_SHOW]):
            print(fmt_result(j+1, candidates[idx], score))

    avg = np.mean(times) * 1000
    print(f"\n  >> {display}: avg {avg:.0f}ms/query")
    return avg


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Reranker Eval v2 — Real TutorialVault Data (211K chunks)")
    print("=" * 70)

    results = {}

    # FlashRank (already proven)
    print(f"\n{'='*70}")
    print("  FlashRank MiniLM-L12 (34MB)")
    print(f"{'='*70}")
    try:
        results["FlashRank MiniLM-L12 (34MB)"] = test_flashrank("ms-marco-MiniLM-L-12-v2", "MiniLM-L12")
    except Exception as e:
        print(f"  FAILED: {e}")

    # GTE multilingual ONNX
    print(f"\n{'='*70}")
    print("  GTE-multilingual-reranker-base int8 (341MB)")
    print(f"{'='*70}")
    try:
        results["GTE-reranker int8 (341MB)"] = test_gte_onnx()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    # Qwen3 ONNX
    print(f"\n{'='*70}")
    print("  Qwen3-Reranker-0.6B seq-cls (1.19GB)")
    print(f"{'='*70}")
    try:
        results["Qwen3-Reranker-0.6B (1.19GB)"] = test_qwen3_onnx()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("  SPEED SUMMARY")
    print(f"{'='*70}")
    # Include FlashRank TinyBERT from previous eval
    print(f"  {'FlashRank TinyBERT (4MB)':<40} {'11':>6}ms / query (prev eval)")
    for name, ms in results.items():
        print(f"  {name:<40} {ms:>6.0f}ms / query")
