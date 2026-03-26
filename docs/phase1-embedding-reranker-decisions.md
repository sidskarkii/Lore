# Building a PyTorch-Free RAG Pipeline: Embedding & Reranker Selection

> How we cut the install size from 2.2GB to 100MB without losing retrieval quality.

## The Problem

Every RAG tutorial starts the same way:

```bash
pip install sentence-transformers  # pulls PyTorch (~2GB)
```

That's fine for a Colab notebook. But we're building a desktop tool that real people install and run. Asking someone to download 2GB of PyTorch just to search their own files is absurd.

We needed embedding and reranking models that:
- Run without PyTorch
- Are small enough for a reasonable install
- Don't sacrifice retrieval quality
- Work on CPU (not everyone has a GPU)

## Part 1: Choosing an Embedding Model

### The Baseline: BGE-M3

BGE-M3 (BAAI) is the default recommendation in most RAG guides. It's good — 1024-dim dense vectors, multilingual, supports sparse and ColBERT outputs. We used it in our earlier prototype.

The problem: the official HuggingFace repo doesn't ship ONNX weights. Community ONNX exports exist, but they're inconsistent — different output shapes, missing files, broken on Windows due to symlink issues. We tried `aapot/bge-m3-onnx` and hit a shape mismatch error where the ONNX model output `[batch, dim]` directly instead of `[batch, seq_len, dim]`, breaking our mean-pooling code.

Even when it works, the model weights are 2.3GB (FP32).

### The Alternative: EmbeddingGemma-300M

Google's EmbeddingGemma-300M caught our attention:
- #1 on MTEB for models under 500M parameters
- 100+ language support
- Official ONNX export by `onnx-community` (not a random community fork)
- Multiple quantization levels available
- 768-dim output (vs BGE-M3's 1024)

The `onnx-community/embeddinggemma-300m-ONNX` repo ships with `tokenizer.json`, which means we can use HuggingFace's `tokenizers` library (a 5MB Rust binary) instead of the full `transformers` package.

### Quantization Comparison

We ran every available variant on the same test corpus (15 3D tutorial chunks) and 5 queries with known correct answers:

| Variant | Size | Load Time | Accuracy | Speed |
|---------|------|-----------|----------|-------|
| fp32 | 1,178 MB | 0.59s | 5/5 (100%) | 11.3 ms/text |
| q8 | 295 MB | 0.48s | 5/5 (100%) | 40.1 ms/text |
| **q4** | **188 MB** | **0.69s** | **5/5 (100%)** | **14.3 ms/text** |
| q4f16 | 168 MB | 0.72s | 5/5 (100%) | 17.4 ms/text |

Every variant hit 100% accuracy. Zero quality loss from quantization on this test set.

q4 is the sweet spot — 6x smaller than fp32, only 3ms slower, and the q4f16 docs warn about fp16 activation issues on some hardware.

### Why Not Smaller Models?

We considered `all-MiniLM-L6-v2` (~80MB, 384-dim) and Model2Vec's `potion-base-8M` (~8MB). The problem is quality — these are fine for English keyword-style queries but lose nuance on domain-specific content. When someone asks "how to adjust bone influence falloff in weight paint mode," you need a model that understands the semantic relationships between those terms, not just keyword overlap.

### Decision

**EmbeddingGemma-300M, q4 quantization, loaded via ONNX Runtime.**

```
Before: torch (2GB) + sentence-transformers + BGE-M3 (2.3GB) = ~4.3GB
After:  onnxruntime (50MB) + tokenizers (5MB) + model (188MB) = ~243MB
```

The embedding code loads the model lazily on first use, downloads only the q4 variant (not the full repo), and auto-detects CPU/GPU:

```python
from huggingface_hub import hf_hub_download

# Download ONLY the q4 files — not the 1.2GB fp32
hf_hub_download(repo, "onnx/model_q4.onnx")
hf_hub_download(repo, "onnx/model_q4.onnx_data")
```

---

## Part 2: Choosing a Reranker

### Why Rerank at All?

Initial retrieval (vector search + BM25) casts a wide net. It finds the right neighborhood but the ordering within the top 20 results is noisy. A reranker is a cross-encoder that looks at the query and each candidate together, producing much more accurate relevance scores.

The tradeoff is speed — cross-encoders process one (query, document) pair at a time instead of encoding them independently.

### The Landscape

We surveyed the full reranking ecosystem:

**Cross-encoder models:**
- FlashRank (TinyBERT, MiniLM-L12, MultiBERT, rank-T5-flan)
- GTE-multilingual-reranker-base (Alibaba, ONNX available)
- Qwen3-Reranker-0.6B (ONNX available)
- Jina Reranker v3 (GGUF available)
- Contextual AI Reranker v2 (instruction-following)
- mxbai-rerank-v2 (mixedbread)
- bge-reranker-v2-m3 (what we used before, no ONNX export)

**Alternative approaches:**
- LLM-as-reranker (send candidates to Claude/GPT for scoring)
- ColBERT late-interaction (per-token matching)
- PROVENCE (simultaneous rerank + context compression)

Most of the interesting models are 500MB+ or require PyTorch. We narrowed to what's actually testable without torch.

### The Test

We ran 4 rerankers on **real data** — the 211,000-chunk LanceDB from our earlier prototype, covering Blender tutorials, Houdini documentation, Unreal Engine wikis, and StackExchange Q&A. BM25 pulled the initial 20 candidates per query. Each reranker re-scored the same candidates.

8 real queries: rigging, weight painting, VOP noise, sculpting wrinkles, retopology, particle hair, shader nodes, graph editor curves.

### Speed Results

| Reranker | Model Size | Avg Time / Query | PyTorch? |
|----------|-----------|-------------------|----------|
| FlashRank TinyBERT | 4 MB | 11 ms | No |
| **FlashRank MiniLM-L12** | **34 MB** | **349 ms** | **No** |
| FlashRank MultiBERT | 150 MB | 926 ms | No |
| FlashRank rank-T5-flan | 110 MB | 1,105 ms | No |
| GTE-reranker int8 (ONNX) | 341 MB | 1,693 ms | No |
| Qwen3-Reranker-0.6B (ONNX) | 1,190 MB | 10,262 ms | No |

### Quality Results

Here's where it gets interesting. Bigger models weren't better on our data.

**Query: "sculpting detailed wrinkles on a face"**

| Rank | MiniLM-L12 (34MB) | GTE-reranker (341MB) | Qwen3-Reranker (1.19GB) |
|------|-------|-------|-------|
| #1 | "What details should show in sculpting" (StackExchange) | "Wrinkles fade out when retopologizing" | "Modeling clothes creases/wrinkles" |
| #2 | Wolf Modeling tutorial on nose wrinkles | "How to simulate sculpt layers" | Wolf Modeling tutorial on skin wrinkles |

MiniLM-L12 surfaced a directly relevant sculpting question at #1. GTE's top pick is about retopology artifacts, not sculpting technique. Qwen3's top pick is about clothing wrinkles — wrong domain entirely.

**Query: "weight painting bone influence"**

| Rank | MiniLM-L12 | GTE-reranker | Qwen3-Reranker |
|------|-------|-------|-------|
| #1 | Tutorial demo of Control-click to view bone influence in Weight Paint mode | "Remove influence of selected bones on vertices" | "Can't interact with rigging bones" |

MiniLM-L12 pulled an actual tutorial walkthrough. GTE and Qwen3 found tangentially related problems.

**Query: "shader nodes for realistic skin material"**

MiniLM-L12 returned the Houdini Principled Material docs and a Cycles SSS skin shader tutorial. Qwen3's top result was "How to place a .png texture with transparency" — completely off target.

### Why Bigger Wasn't Better

Three factors:

1. **Domain mismatch.** Qwen3-Reranker was trained on general web data and benchmarked on MTEB (news, Wikipedia, generic Q&A). Our corpus is niche 3D software documentation. The model's understanding of "relevance" doesn't align with what a Blender user considers relevant.

2. **Input format sensitivity.** Qwen3-Reranker expects an instruct-style prompt (`Query: ... Document: ... Is this document relevant?`). MiniLM-L12 just takes the raw text pair. With domain-specific jargon, the simpler approach actually works better — the model focuses on text similarity rather than trying to "reason" about relevance.

3. **Quantization artifacts at scale.** The Qwen3 ONNX export is a 0.6B parameter model compressed into a single ONNX file. At 10 seconds per query on CPU, it's also just impractical — you'd need GPU acceleration to make it usable, which defeats our goal.

### GPU Acceleration?

We tested with `onnxruntime-gpu` but CUDA integration on Windows requires system-level CUDA libraries (`cublasLt64_12.dll`, cuDNN 9.x) in the PATH. These exist in our whisperx environment but aren't visible to the main Python install.

Even with GPU, Qwen3 would drop from ~10s to maybe ~1s — still slower than MiniLM-L12's 349ms on CPU. GPU acceleration matters more for batch embedding at ingestion time, not for reranking 20 candidates at query time.

### Decision

**FlashRank with `ms-marco-MiniLM-L-12-v2` as the default reranker.**

- 34MB model download, auto-cached after first use
- 349ms per query on CPU (20 candidates)
- Best quality on our actual data
- Pure ONNX internally — no PyTorch, no transformers
- `TinyBERT` (4MB, 11ms) available as a fast-mode option in config

```yaml
search:
  reranker_model: ms-marco-MiniLM-L-12-v2  # or ms-marco-TinyBERT-L-2-v2 for speed
```

---

## The Final Stack

| Component | Model | Size | Speed | PyTorch? |
|-----------|-------|------|-------|----------|
| Embeddings | EmbeddingGemma-300M (q4 ONNX) | 188 MB | 14 ms/text | No |
| Reranking | FlashRank MiniLM-L12 | 34 MB | 349 ms/query | No |
| Vector DB | LanceDB (embedded) | ~30 MB | - | No |
| Tokenizer | HuggingFace tokenizers (Rust) | 5 MB | - | No |
| Runtime | ONNX Runtime | 50 MB | - | No |

**Total base install: ~100MB.** First run downloads the embedding model (~188MB) and reranker (~34MB) from HuggingFace cache.

Compare to the typical RAG starter:
```
torch==2.x          2,000 MB
sentence-transformers  50 MB
BGE-M3 weights      2,300 MB
                    ─────────
                    ~4,350 MB
```

That's a 95% reduction in download size with identical retrieval quality on our test set.

### Search Pipeline

```
Query
  │
  ├─→ Embed (EmbeddingGemma q4, 14ms)
  │     │
  │     ├─→ Vector search (top 30 by cosine similarity)
  │     │
  │     └─→ BM25 search (top 30 by keyword matching)
  │           │
  │           ▼
  │     RRF Fusion (merge ranked lists)
  │           │
  │           ▼
  │     FlashRank rerank (cross-encoder, 349ms)
  │           │
  │           ▼
  │     Parent window expansion (±2.5 min context)
  │           │
  └───────────▼
        Results with timestamps + source metadata
```

### What's Next

This is the search engine. It can ingest chunks, embed them, store them, and find relevant results. What it can't do yet:

- **Talk to an LLM** — needs a provider layer (Claude Code, OpenCode, Kilo CLI, local GGUF)
- **Show results in a UI** — needs a Gradio web interface
- **Transcribe videos** — needs the Whisper pipeline
- **Download from YouTube** — needs yt-dlp integration

Those are the next phases. The foundation is in place.
