"""Search — hybrid retrieval with RRF fusion, reranking, and parent expansion."""

from __future__ import annotations

from .chunk import fmt_timestamp
from .config import get_config
from .embed import embed_texts
from .store import Store

_reranker_session = None
_reranker_tokenizer = None


def _get_reranker():
    """Lazy-load cross-encoder reranker via ONNX."""
    global _reranker_session, _reranker_tokenizer
    if _reranker_session is not None:
        return _reranker_session, _reranker_tokenizer

    cfg = get_config()
    model_name = cfg.get("search.reranker_model")
    if model_name is None:
        return None, None

    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    from pathlib import Path

    print(f"  Loading reranker {model_name} (ONNX)...")
    model_path = Path(snapshot_download(
        model_name,
        allow_patterns=["onnx/*", "tokenizer*", "special_tokens_map.json", "vocab.txt", "sentencepiece*"],
    ))

    onnx_dir = model_path / "onnx"
    onnx_file = onnx_dir / "model.onnx"
    if not onnx_file.exists():
        onnx_files = list(onnx_dir.glob("*.onnx"))
        onnx_file = onnx_files[0] if onnx_files else None
    if onnx_file is None:
        print(f"  Reranker ONNX not found for {model_name}, skipping reranking")
        return None, None

    device = cfg.embed_device
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "cuda" else ["CPUExecutionProvider"])
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _reranker_session = ort.InferenceSession(str(onnx_file), sess_opts, providers=providers)

    max_len = cfg.get("search.reranker_max_length", 512)
    tok_file = model_path / "tokenizer.json"
    _reranker_tokenizer = Tokenizer.from_file(str(tok_file))
    _reranker_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    _reranker_tokenizer.enable_truncation(max_length=max_len)

    return _reranker_session, _reranker_tokenizer


def _rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Score query-text pairs with the cross-encoder reranker."""
    import numpy as np

    session, tokenizer = _get_reranker()
    if session is None:
        return [0.0] * len(texts)

    # Cross-encoders take paired input: (query, text)
    encoded = tokenizer.encode_batch([(query, t) for t in texts])
    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

    feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
    input_names = [inp.name for inp in session.get_inputs()]
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

    outputs = session.run(None, feeds)
    # Reranker outputs logits — higher = more relevant
    logits = outputs[0].flatten()
    return logits.tolist()


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple ranked ID lists."""
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return scores


def _build_where(topic: str | None, subtopic: str | None) -> str | None:
    """Build a LanceDB WHERE clause from filters."""
    parts = []
    if topic:
        parts.append(f"topic = '{topic.lower()}'")
    if subtopic:
        parts.append(f"subtopic = '{subtopic.lower()}'")
    return " AND ".join(parts) if parts else None


class SearchEngine:
    """Hybrid search with RRF fusion, cross-encoder reranking, parent expansion."""

    def __init__(self, store: Store | None = None):
        self.store = store or Store()
        self._cfg = get_config()

    def search(
        self,
        query: str,
        n_results: int = 5,
        topic: str | None = None,
        subtopic: str | None = None,
    ) -> list[dict]:
        """Run hybrid search and return ranked, expanded results.

        Pipeline:
        1. Embed query
        2. Vector search (semantic)
        3. FTS search (BM25 keyword)
        4. RRF fusion
        5. Cross-encoder reranking
        6. Parent window expansion
        """
        cfg = self._cfg
        candidate_count = cfg.get("search.candidate_count", 30)
        rrf_k = cfg.get("search.rrf_k", 60)
        where = _build_where(topic, subtopic)

        # 1. Embed query
        query_vec = embed_texts([query])[0]

        # 2. Vector search
        vec_results = self.store.vector_search(query_vec, n=candidate_count, where=where)

        # 3. FTS search
        fts_results = self.store.fts_search(query, n=candidate_count, where=where)

        # 4. RRF fusion
        vec_ids = [r["id"] for r in vec_results] if vec_results else []
        fts_ids = [r["id"] for r in fts_results] if fts_results else []
        rrf_scores = _rrf([vec_ids, fts_ids], k=rrf_k)

        # Merge and deduplicate
        all_by_id: dict[str, dict] = {}
        for r in vec_results + fts_results:
            if r["id"] not in all_by_id:
                all_by_id[r["id"]] = r

        # Sort by RRF score
        candidates = sorted(
            [r for r in all_by_id.values() if r["id"] in rrf_scores],
            key=lambda r: rrf_scores.get(r["id"], 0),
            reverse=True,
        )[:n_results * 4]

        # 5. Cross-encoder reranking
        if candidates:
            scores = _rerank_scores(query, [c["text"] for c in candidates])
            if any(s != 0.0 for s in scores):
                ranked = sorted(
                    zip(scores, candidates),
                    key=lambda x: x[0],
                    reverse=True,
                )
                results = [c for _, c in ranked[:n_results]]
            else:
                results = candidates[:n_results]
        else:
            results = candidates[:n_results]

        # 6. Parent window expansion
        results = [self._expand_to_parent(r) for r in results]

        return results

    def _expand_to_parent(self, chunk: dict) -> dict:
        """Expand a matched chunk to its surrounding parent window."""
        window = self._cfg.get("search.parent_window_sec", 150)
        try:
            center = int(chunk["start_sec"])
            win_start = max(0, center - window)
            win_end = center + window

            neighbors = self.store.get_neighbors(
                collection=chunk["collection"],
                episode_num=chunk["episode_num"],
                start_sec=win_start,
                end_sec=win_end,
            )

            if not neighbors:
                return chunk

            # Deduplicate overlapping text, preserve order
            seen: set[tuple[int, int]] = set()
            parts: list[str] = []
            for row in neighbors:
                key = (row["start_sec"], row["end_sec"])
                if key not in seen:
                    seen.add(key)
                    parts.append(row["text"])

            expanded = dict(chunk)
            expanded["text"] = "\n\n".join(parts)
            expanded["start_sec"] = int(neighbors[0]["start_sec"])
            expanded["end_sec"] = int(neighbors[-1]["end_sec"])
            expanded["timestamp"] = fmt_timestamp(neighbors[0]["start_sec"])
            return expanded
        except Exception:
            return chunk
