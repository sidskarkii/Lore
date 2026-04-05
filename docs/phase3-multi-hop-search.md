# Stealing from a 20B Model: Multi-Hop Search Without the 20B Model

> How we took the core idea from Chroma Context-1 and implemented it in 80 lines on top of our existing search pipeline.

## The Trigger

Chroma released Context-1 — a 20 billion parameter "search agent" that decomposes complex queries into sub-queries, iteratively searches, and prunes its own context window. John Schulman co-signed it. The technical report is real: RL training via CISPO/GRPO, self-editing context windows, staged curriculum from recall to precision.

The question was whether we needed it for TutorialVault.

## What We Already Had

Our single-hop search pipeline handles queries like "what sculpting brushes are available in Blender" perfectly:

```
Query
  -> Embed (EmbeddingGemma q4, 14ms)
  -> Vector search (top 30 by cosine similarity)
  -> BM25 search (top 30 by keyword matching)
  -> RRF fusion (merge ranked lists)
  -> FlashRank rerank (cross-encoder, 349ms)
  -> Parent window expansion (+/- 2.5 min context)
  -> Results with timestamps
```

One query in, N results out. Works great for direct questions.

## Where It Breaks

Now consider: "Compare the sculpting workflow to the mesh editing approach and which is better for beginners."

This query needs information from multiple tutorials, multiple episodes, multiple topics. Single-hop embeds the whole thing as one vector and hopes the results cover enough ground. In practice, the embedding collapses the query into a single semantic direction — it'll find chunks that mention sculpting AND mesh editing together (like transition sentences between tutorials), but miss the detailed breakdowns that live in separate episodes.

This is the multi-hop problem. Context-1 solves it by training a 20B model to decompose, search, and prune iteratively. We solve it by making one cheap LLM call and reusing the search pipeline we already have.

## The Insight

Context-1's architecture is three steps:

1. **Decompose** — break a complex query into simpler sub-queries
2. **Search** — run each sub-query independently
3. **Prune** — drop irrelevant results before sending to the reasoning model

Steps 2 and 3 are already in our pipeline. Vector search + BM25 + RRF + FlashRank is the search. FlashRank scores are the pruning signal. The only missing piece is step 1 — decomposition — and we already have an LLM provider layer sitting right there.

## The Implementation

### Query Decomposition

One LLM call. The prompt is deliberately simple:

```python
_DECOMPOSE_PROMPT = (
    "Break this question into {max_queries} or fewer simpler search queries "
    "that together would find all the information needed to answer it. "
    "Return one query per line, nothing else. If the question is already "
    "simple, return it unchanged on one line."
)
```

No JSON schema. No chain-of-thought. No few-shot examples. Just "give me sub-queries, one per line." This works because:

- The decomposition task is trivial for any LLM — it's not reasoning, it's paraphrasing
- Plain text output is robust to parse (split on newlines, strip numbering)
- Even free models (Kilo, OpenCode) handle it correctly
- The prompt is ~50 tokens in, ~40 tokens out — fast and cheap

The parser strips common formatting the LLM might add:

```python
_STRIP_NUMBERING = re.compile(r"^\s*[\d]+[.):\-]\s*")

def _parse_sub_queries(text: str, max_queries: int) -> list[str]:
    queries = []
    for line in text.strip().splitlines():
        line = _STRIP_NUMBERING.sub("", line).strip().strip('"').strip("'")
        if line and len(line) > 5:
            queries.append(line)
    return queries[:max_queries]
```

If the LLM returns "1. sculpting tools\n2. mesh editing\n3. comparison", we get `["sculpting tools", "mesh editing", "comparison"]`. If it returns the original query on one line (simple query), we fall back to single-hop. If the call fails entirely, we fall back to single-hop. Every failure mode degrades gracefully.

### Multi-Pass Search

Each sub-query runs through the full existing pipeline — embedding, vector search, BM25, RRF fusion, FlashRank reranking. We're not doing lightweight retrieval here. Each sub-query gets the same quality treatment as a standalone query.

```python
for sq in sub_queries:
    results = self.search(sq, n_results=per_query_n, topic=topic, subtopic=subtopic)
    ranked_ids = []
    for r in results:
        ranked_ids.append(r["id"])
        if r["id"] not in all_by_id:
            all_by_id[r["id"]] = r
    all_ranked_ids.append(ranked_ids)
```

This gives us N ranked lists (one per sub-query) and a deduplicated pool of all candidates.

### Cross-Query Fusion

Here's where it gets elegant. We already have RRF (Reciprocal Rank Fusion) for merging vector search and BM25 results. The same function works for merging sub-query results:

```python
rrf_scores = _rrf(all_ranked_ids, k=rrf_k)
```

A chunk that appears highly ranked in multiple sub-queries gets a higher combined score than one that only appears in one. This naturally surfaces results that are relevant to the overall question, not just one aspect of it.

### Final Reranking Against the Original Query

This is the key step that Context-1 gets right and naive implementations miss. After combining results from sub-queries, we rerank against the **original** query, not the sub-queries:

```python
scored = _rerank_with_scores(query, candidates, n_results * 2)
```

Why? The sub-queries are approximations. A chunk that scored well for "mesh editing tools" might be completely irrelevant to "Compare sculpting to mesh editing for beginners." FlashRank cross-encoder scoring against the original question catches this.

### Self-Editing: Relevance Threshold

Context-1's headline feature is "self-editing context" — the model learns to drop documents that aren't helping. Our version is simpler: use the FlashRank scores from the final reranking step.

```python
threshold = cfg.get("search.multi_hop_relevance_threshold", 0.1)
results = [c for c, score in scored if score >= threshold]
```

If FlashRank gives a chunk a score below 0.1 against the original query, it's noise. Drop it. This means multi-hop can return fewer than N results — and that's the right behavior. Stuffing irrelevant context into the LLM's prompt degrades answer quality. Returning 3 highly relevant results is better than 5 where 2 are noise.

If the threshold prunes everything (too aggressive for the domain), we fall back to the top N regardless:

```python
if not results:
    results = [c for c, _ in scored[:n_results]]
```

## What We Didn't Do

**No iterative search.** Context-1 does multiple rounds — search, read results, decide what to search for next. We do one round. For tutorial content where queries are "compare X to Y" or "how does A relate to B," one decomposition pass is enough. Iterative search matters more for legal discovery or patent analysis where you don't know what you're looking for until you start reading.

**No LLM-based pruning.** We considered a second LLM call: "which of these 10 results are actually relevant to the original question?" But FlashRank already does this with scores, costs zero tokens, and runs in <400ms. An LLM pruning pass would add 5-20 seconds depending on the provider and wouldn't meaningfully improve results for our corpus.

**No new model.** Context-1 is 20B parameters (~10-40GB depending on quantization). Our entire search stack is 222MB (188MB embeddings + 34MB reranker). Adding a 20B model would blow up the install size by 50-200x and require a GPU. The whole point of using EmbeddingGemma q4 and FlashRank MiniLM was to stay lightweight. We're not going to throw that away for a feature that works fine with one cheap LLM call.

## The Pipeline

```
Complex Query: "Compare sculpting to mesh editing for beginners"
  |
  v
[1] Decompose (1 LLM call, ~50 tokens)
  -> "sculpting brushes and tools in Blender"
  -> "mesh editing tools and techniques"
  -> "differences between sculpting and mesh editing workflows"
  |
  v
[2] Search each sub-query (full pipeline x3)
  -> Sub-query 1: 5 results (sculpting episodes)
  -> Sub-query 2: 5 results (mesh editing episodes)
  -> Sub-query 3: 5 results (mixed)
  |
  v
[3] Deduplicate + RRF across all sub-query rankings
  -> 10 unique candidates (some appeared in multiple sub-queries)
  |
  v
[4] Final FlashRank rerank against ORIGINAL query
  -> Scored and sorted by relevance to the actual question
  |
  v
[5] Threshold pruning (drop score < 0.1)
  -> 5 results that are genuinely relevant
  |
  v
[6] Parent window expansion (+/- 2.5 min)
  -> Each result expanded to surrounding context

Single-hop would give you 5 results, probably all from one episode.
Multi-hop gives you 5 results across multiple episodes, covering different facets.
```

## Test Results

We ingested 3 episodes from a Blender creature modeling course (Introduction, Sculpting Tools, Mesh Editing Tools — 15 chunks total) and ran a comparison query:

**Query:** "Compare the sculpting workflow to the mesh editing approach and which is better for beginners"

**Multi-hop (StubProvider):**
- Decomposed into 3 sub-queries
- 5 results from 10 unique candidates
- Covered episodes 2 and 3 (sculpting AND mesh editing)
- 2.5 seconds

**Single-hop:**
- 5 results
- Also covered episodes 2 and 3
- 0.5 seconds

With only 15 chunks across 3 episodes, the difference is modest — both approaches find the same content because there isn't much to search. The real value shows with a larger corpus spanning dozens of tutorials across different tools and workflows, where single-hop's embedding collapse becomes a real problem.

## Graceful Degradation

Every failure mode falls back to single-hop:

| Scenario | Behavior |
|----------|----------|
| No provider available | Falls back to `search()` |
| LLM call fails (timeout, error) | Falls back to `search()` |
| LLM returns 0 or 1 sub-queries | Falls back to `search()` |
| All results pruned by threshold | Returns top N regardless |
| Multi-hop exception anywhere | Caught in chat route, falls back |

The user never sees an error from multi-hop. Worst case, they get single-hop results — which are already good.

## Integration

The chat API gains one optional field:

```json
POST /api/chat
{
    "messages": [...],
    "multi_hop": true
}
```

`multi_hop` defaults to `false`. Existing clients don't break. The WebSocket streaming endpoint adds a `{"type": "status"}` message ("Decomposing query...") that clients can display or ignore.

## The Portfolio Argument

When someone asks "why didn't you just use Context-1?" the answer is:

1. **It's 20B parameters.** Our total stack is 222MB. A quantized Context-1 would be 10-40GB. We're building a desktop tool, not a cloud service.

2. **We don't need iterative search.** Tutorial content has a finite, known structure. One decomposition pass handles "compare X to Y" queries, which are the main multi-hop use case.

3. **The pattern is more interesting than the model.** Decompose, search per sub-query, RRF combine, rerank against the original, prune by score. That's the idea. You can implement it with any LLM and any search pipeline. We did it in 80 lines on top of existing infrastructure.

4. **We know when NOT to use it.** Single-hop is faster and sufficient for 90% of queries. Multi-hop is opt-in. Shipping a 20B model for every query when most queries are "how do I X" is poor engineering judgment.

Understanding the research, extracting the useful pattern, and adapting it to your constraints — that's the skill. Not downloading a model.
