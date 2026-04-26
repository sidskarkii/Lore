# Wiki Layer Design

## Goal

Build a local-first, agent-native wiki layer on top of Lore's existing chunk store and archive pipeline.

The wiki is not a second search engine. It is a synthesis layer:

- Search finds evidence.
- Wiki pages distill stable knowledge from that evidence.
- Every wiki claim stays tied to underlying chunks.

This design assumes Lore's current architecture:

- Archive-first ingest artifacts in `~/.lore/archive/<collection>/`
- LanceDB as the operational search layer
- MCP tools as the primary UI
- Existing derived indexes: `EntityIndex`, `EntityGraph`, `KeywordTagGraph`, `CrossSourceIndex`, `JaccardIndex`
- Existing LLM routing: cheap model for narrow extraction, stronger model for synthesis

## Core Recommendation

Use a two-layer wiki:

1. Markdown files in `~/.lore/wiki/` are the source of truth.
2. A lightweight derived wiki index is embedded into LanceDB for retrieval and ranking.

This is the right fit for Lore because:

- It matches the archive-first design already used for source summaries.
- It keeps pages inspectable, editable, diffable, and easy to back up.
- It still lets agents search wiki pages through the same retrieval flow.
- It avoids turning LanceDB rows into the only canonical representation of long synthesized documents.

## MVP vs Full Vision

### MVP

Ship only three wiki page types:

1. `entity`
   - Canonical page for a resolved entity cluster from `EntityIndex`
   - Best for people, organizations, places, named works, products
   - High leverage because entity detection and fuzzy merging already exist

2. `concept`
   - Page for a canonical concept tag or strongly recurring keyword/tag cluster
   - Best for ideas, mechanisms, themes, tactics, patterns
   - This becomes the main "Karpathy-style wiki" surface

3. `source`
   - Thin wiki wrapper around existing `book_summary.json`
   - Do not regenerate from scratch initially
   - Convert existing source summaries into wiki page format so all pages share one system

### Phase 2

Add one synthesis type:

4. `comparison`
   - Focused pages comparing 2-5 sources or viewpoints around a concept/entity
   - Example: "How different books treat power, influence, and deception"
   - These should be generated on demand, not exhaustively prebuilt

### Skip for now

- Auto-generating pages for every section
- Auto-generating pages for every graph community
- Full claim graph / knowledge triples / ontology extraction
- Automatic page hierarchies with parent-child taxonomies
- Multi-page debate/consensus tracking
- Continuous background regeneration daemon

For 4-20 sources, those are complexity traps.

## Page Types

### 1. Entity Pages

Source of candidates:

- `EntityIndex` canonical clusters
- Filter to clusters that are meaningful enough to deserve a page

Creation threshold:

- Mentioned in at least 2 chunks, or
- Appears in at least 2 collections, or
- Is a high-centrality / bridge node in `EntityGraph`

Purpose:

- Explain who/what the entity is in the corpus
- Summarize roles across sources
- List related concepts and entities
- Surface disagreements or framing differences across sources

These pages should not become biographies. They are corpus-grounded summaries.

### 2. Concept Pages

Source of candidates:

- Stage 2 `concept_tags`
- Stage 3 `key_concepts` / section ledgers
- `KeywordTagGraph` communities
- Search-query demand signals later

Creation threshold:

- Appears in at least 3 chunks, or
- Appears across at least 2 sources, or
- Is requested explicitly by agent via tool

Purpose:

- Define the concept in Lore's corpus terms
- Explain mechanisms, applications, tensions, and related concepts
- Synthesize across books instead of restating one source

Concept pages are the highest-value page type and should be the default wiki surface.

### 3. Source Pages

Source:

- Existing `book_summary.json`
- Existing `section_summaries.json`
- Existing `meta.json`

Purpose:

- Normalize source summaries into the wiki schema
- Provide a stable browse entrypoint
- Link source pages to concept/entity pages they contribute to

Do not spend LLM budget regenerating these during MVP unless the archive summary is missing.

### 4. Comparison Pages

Do not precompute broadly.

Generate only when:

- An agent asks for comparison between named sources
- An agent asks how multiple sources differ on a concept/entity
- A concept spans enough sources to justify a dedicated synthesis

These pages are expensive and lower-frequency. Treat them as cached derived products.

## Storage

## Recommendation

Choose Option C:

- Markdown in `~/.lore/wiki/` as canonical pages
- LanceDB rows as derived search/index entries

### Directory layout

```text
~/.lore/wiki/
  pages/
    entity/
      napoleon-bonaparte.md
      robert-cialdini.md
    concept/
      social-proof.md
      deception-as-advantage.md
    source/
      influence.md
      art-of-war.md
    comparison/
      influence-vs-48-laws-on-persuasion.md
  manifests/
    pages.json
    backlinks.json
  state/
    generation_jobs.json
    dirty_queue.json
```

### Why markdown should be canonical

- Human-readable and inspectable
- Easy to patch manually if the generated page is close but imperfect
- Natural fit for local-first backup and git workflows
- Easier for agents to quote, inspect, and revise
- Matches the existing archive artifact philosophy

### Why LanceDB should still index wiki pages

- The wiki must appear in normal retrieval
- Embedding summaries and sections improves answer quality
- Agents should not need a separate retrieval mental model for wiki vs chunks

### What gets indexed

Do not index the whole markdown page as one long row.

Index page fragments:

- page summary / lead
- claim bullets or sections
- related concepts/entities section
- source coverage metadata

This keeps retrieval granular and ranking sane.

## Page Schema

Use frontmatter plus structured sections.

```md
---
page_id: concept/social-proof
page_type: concept
title: Social Proof
slug: social-proof
status: generated
version: 1
created_at: 2026-04-26T00:00:00Z
updated_at: 2026-04-26T00:00:00Z
source_collections:
  - influence
  - 48_laws_of_power
source_chunk_count: 18
supporting_source_count: 2
corroboration_level: moderate
confidence: medium
generation:
  strategy: synthesized
  model: sonnet
  based_on:
    entity_clusters: []
    concept_tags:
      - social-proof
  inputs_hash: abc123
  source_versions:
    influence: 9f3e...
    48_laws_of_power: 2aa1...
related_pages:
  - concept/conformity
  - entity/robert-cialdini
canonical_sources:
  - collection: influence
    weight: primary
---

# Social Proof

## Summary
...

## Key Claims
- Claim text.
  Sources: `influence_p12_0012`, `influence_p14_0016`
  Support: 2 chunks, 1 collection

## Cross-Source Synthesis
...

## Tensions
...

## Related Pages
- [[concept/conformity]]
- [[entity/robert-cialdini]]

## Provenance
- `influence_p12_0012`
- `48_laws_of_power_ch_x_0042`
```

### Required metadata fields

- `page_id`
- `page_type`
- `title`
- `slug`
- `updated_at`
- `source_collections`
- `source_chunk_count`
- `supporting_source_count`
- `generation.inputs_hash`
- `generation.source_versions`

### Optional but useful

- `canonical_entity` for entity pages
- `canonical_concept` for concept pages
- `status`: `generated|draft|stale|failed`
- `confidence`
- `corroboration_level`

## Generation Triggers

## Recommendation

Use a hybrid trigger model:

1. Auto-generate a small set after ingest
2. Generate the rest on demand
3. Cache and mark stale for later regeneration

### After ingest

Automatically create or refresh:

- source page for the new collection
- top N concept pages touched by the new collection
- top N entity pages touched by the new collection

Recommended defaults:

- top 10 concepts
- top 10 entities
- only if they cross threshold

This gives the corpus immediate wiki coverage without exploding cost.

### On demand

Agents should be able to ask for:

- a page by exact slug
- a page for an entity or concept name
- a comparison page
- regeneration of a stale page

### Lazy generation

If a requested page does not exist:

- generate it synchronously if the candidate set is small
- otherwise return a preview plus a job id and let the agent poll

This matches the MCP pattern better than generating everything upfront.

## Cross-References

Pages should link in three directions:

1. Page -> page
2. Page -> source collection
3. Page -> exact supporting chunks

### Page links

Use stable `page_id` references internally:

- `concept/social-proof`
- `entity/napoleon-bonaparte`

Render markdown links in the page body, but store structured `related_pages` in frontmatter too.

### Source links

Every page should list:

- contributing collections
- dominant collections
- exact supporting chunk ids

### Backlinks

Maintain a derived `backlinks.json` manifest so the MCP layer can answer:

- what pages mention this page
- what wiki pages cite this chunk

Do not compute backlinks by scanning all files on every request.

## Provenance and Confidence

This is where the wiki layer must be stricter than normal summarization.

## Recommendation

Track provenance at the claim level, not only at the page level.

### Claim unit

Represent each important bullet/claim in structured form during generation:

```json
{
  "claim_id": "social-proof-claim-03",
  "text": "People often treat observed group behavior as evidence of correctness under uncertainty.",
  "chunk_ids": ["influence_p12_0012", "influence_p14_0016"],
  "collections": ["influence"],
  "support_count": 2,
  "corroboration_count": 1,
  "confidence": "medium"
}
```

Then render those into markdown.

### Corroboration rubric

- `low`: 1 chunk or 1 source
- `moderate`: 2+ chunks across 1-2 sources
- `high`: 3+ chunks across 2+ sources with consistent framing
- `mixed`: multiple sources but conflicting framing

### Confidence rubric

Confidence should reflect quality of synthesis, not truth in the abstract:

- `high`: repeated, direct, consistent support
- `medium`: good support but some abstraction or limited source diversity
- `low`: weak or sparse support
- `mixed`: evidence exists but sources disagree materially

### What not to do

- Do not force sentence-level citations inline everywhere
- Do not build a full attribution parser
- Do not claim numerical certainty beyond support counts

For Lore, support counts and collection counts are enough.

## Incremental Updates

## Recommendation

Use dirty-page invalidation plus targeted regeneration. Do not append blindly.

### Why full append is wrong

Concept/entity pages are synthesized narratives. Appending new content will accumulate contradictions, stale framing, and duplicated claims.

### Update model

When a new source is ingested:

1. Determine affected concepts/entities from its chunks
2. Mark matching wiki pages as `stale`
3. Record why they are stale and which collection changed
4. Regenerate only when:
   - the page is requested, or
   - it is in the auto-refresh top N set

### Regeneration strategy

For MVP:

- full page regeneration from selected evidence set

Not:

- token-level patching
- LLM diff-and-merge against old page

Full regeneration is simpler and safer at this corpus size.

### Evidence set selection

Do not regenerate from all chunks in all sources if a concept is broad.

Select evidence by:

- chunks tagged with the concept/entity
- top related chunks from `CrossSourceIndex`
- optional Jaccard neighbors for coverage
- cap total chunk input per page

Recommended cap:

- 12-30 chunks for concept/entity pages
- 20-40 chunks for comparison pages

## LLM Usage

## Recommendation

Use the cheap model for evidence distillation, strong model for final synthesis.

### Model assignment

- source pages: no new LLM in MVP if `book_summary.json` exists
- entity pages:
  - evidence distillation: haiku
  - final page synthesis: sonnet only if cross-source, otherwise haiku is fine
- concept pages:
  - evidence distillation: haiku
  - final synthesis: sonnet
- comparison pages:
  - final synthesis: sonnet

### Pipeline

For concept/entity pages:

1. Gather candidate chunks
2. Deduplicate and rank evidence
3. Distill each chunk or small batch into compact notes
4. Synthesize final page from distilled notes

This is cheaper than feeding raw chunks directly into a large synthesis prompt.

### Cost controls

- Reuse existing chunk summaries and titles heavily
- Prefer section summaries when many chunks come from one section
- Cache intermediate evidence bundles by `inputs_hash`
- Batch small page generations where possible for the cheap model
- Do not batch final synthesis pages aggressively; quality matters more there

### Batch guidance

Good batch candidates:

- entity pages with small evidence sets
- concept pages from same ingest event

Bad batch candidates:

- comparison pages
- pages with mixed or conflicting evidence

## Search Integration

## Recommendation

Wiki pages should be searchable through the existing search tool, but ranked differently from raw chunks.

### Indexing approach

Add wiki page fragments into a separate LanceDB table or a namespaced row type in a shared table.

Recommendation:

- separate table: `wiki_pages`

Reason:

- different schema
- different ranking features
- easier rebuild
- lower risk to chunk-search regressions

### Retrieval behavior

Existing `search` should optionally include wiki results:

- default: include both
- compact result type identifies `result_type: chunk|wiki`

### Ranking policy

For direct factual/source-grounded queries:

- prefer chunks first

For broad explanatory queries:

- let wiki pages rank above chunks if strong match

### Practical heuristic

Boost wiki pages when:

- query looks conceptual (`what is`, `explain`, `overview`, `compare`, `themes of`)
- page has high corroboration
- page has cross-source support

Penalize wiki pages when:

- query asks for exact wording
- query asks for local detail, page number, or quote
- wiki page support is sparse

### Result presentation

Compact wiki search results should include:

- `page_id`
- `page_type`
- `title`
- `summary`
- `supporting_source_count`
- `source_chunk_count`
- `confidence`
- `corroboration_level`

Then the agent can call a page-read tool for full content.

## MCP Tools

Lore does not need many new tools. Add a small, sharp set.

## Recommended tools

### `wiki_search`

Search only wiki pages or prefer wiki pages.

```python
def wiki_search(
    query: str,
    page_type: str | None = None,   # entity|concept|source|comparison
    n_results: int = 8,
    include_stale: bool = False,
) -> dict
```

Returns compact wiki hits.

### `wiki_get_page`

Read a page by id, slug, or approximate title.

```python
def wiki_get_page(
    page_id: str | None = None,
    slug: str | None = None,
    title: str | None = None,
    include_claims: bool = True,
    include_provenance: bool = True,
    max_section_tokens: int = 1600,
) -> dict
```

Returns page metadata, rendered markdown, structured claims, related pages, and provenance.

### `wiki_generate_page`

Explicitly generate or refresh a page.

```python
def wiki_generate_page(
    page_type: str,                 # entity|concept|comparison|source
    target: str,                    # canonical name, slug, collection, etc.
    force: bool = False,
    sync: bool = True,
) -> dict
```

Use for on-demand generation and admin refresh.

### `wiki_related`

Browse the wiki graph.

```python
def wiki_related(
    page_id: str,
    relation: str = "all",          # all|concepts|entities|sources|backlinks
    n_results: int = 10,
) -> dict
```

### `wiki_claims`

Inspect provenance without dumping the full page.

```python
def wiki_claims(
    page_id: str,
    min_support: int = 1,
    corroboration: str | None = None,   # low|moderate|high|mixed
    n_results: int = 20,
) -> dict
```

This matters because agents often need support inspection more than full prose.

### `wiki_queue`

Operational tool for stale/missing pages.

```python
def wiki_queue(
    action: str = "list",           # list|rebuild|clear
    limit: int = 20,
) -> dict
```

## Tool surface to avoid

- No `wiki_edit_page` in MVP
- No `wiki_diff_page`
- No separate `wiki_cite_claim` tool if `wiki_claims` already returns chunk ids
- No large admin surface area

## Candidate Selection

Page quality depends more on evidence selection than prompt cleverness.

## Recommendation

Build deterministic candidate selectors before generation.

### For entity pages

Seed from:

- resolved entity cluster mentions

Expand with:

- `CrossSourceIndex.find_by_entity`
- top `EntityGraph` neighbors for related entities

### For concept pages

Seed from:

- exact `concept_tags`
- Stage 3 `key_concepts`

Expand with:

- `KeywordTagGraph` neighbors
- chunks sharing concept tags / semantic keys
- `JaccardIndex` neighbors for breadth

### Ranking formula

Use a simple weighted score:

- concept/entity exact match: high weight
- cross-source coverage: boost
- chunk `importance`: boost
- source diversity: boost
- near-duplicate penalty

Do not make the generator responsible for selecting evidence from 200 chunks.

## Update and Rebuild Mechanics

Add a small wiki manager with three responsibilities:

1. page registry
2. dirty tracking
3. generation orchestration

Suggested files:

- `src/lore/core/wiki.py`
- `src/lore/core/wiki_index.py`
- `src/lore/core/wiki_generate.py`

### Dirty tracking model

When ingest or delete happens:

- recompute affected concepts/entities from changed collection
- update `dirty_queue.json`
- mark pages stale in `pages.json`

### Source versioning

Each page should store `source_versions` as hashes of contributing source archives.

Recommended source hash basis:

- `chunks.json`
- `section_summaries.json`
- `book_summary.json`

When hashes change, the page is stale even if the title/slug is unchanged.

## What to Skip From The "LLM Wiki" Pattern

Karpathy-style wiki ideas are useful, but Lore does not need the full research-lab version.

Skip these initially:

- Page decomposition into many micro-subpages
- Recursive wiki generation loops
- Autonomous "discover and write all missing pages"
- Rich semantic graph database under the pages
- Fine-grained trust scoring learned from user behavior
- Claim contradiction resolution with adversarial multi-agent debate

Why:

- Lore's corpus is small
- Agents are the UI, so browseability matters more than autonomous worldbuilding
- Local-first cost and latency matter more than maximal automation
- Most value comes from 30-200 good pages, not 5,000 mediocre ones

## Build Order

### Phase 1: Infrastructure

- Create wiki directory layout
- Add page manifest and dirty queue
- Add markdown page schema with frontmatter
- Add `source` page adapter from existing archive summaries

### Phase 2: Concept + Entity generation

- Deterministic candidate selectors
- Evidence bundle builder
- Page generator
- `wiki_get_page` and `wiki_generate_page`

### Phase 3: Search integration

- Index page fragments into LanceDB
- Add `wiki_search`
- Blend chunk and wiki ranking in existing search tool

### Phase 4: Operations

- Stale page invalidation on ingest/delete
- `wiki_queue`
- `wiki_claims`

### Phase 5: Comparison pages

- On-demand only
- Stronger synthesis prompt
- Explicit support display

## Final Recommendations

If Lore wants the fastest path to a high-value wiki layer, build this:

1. Canonical markdown wiki in `~/.lore/wiki/`
2. Three MVP page types: `source`, `entity`, `concept`
3. Hybrid trigger model: small auto-generation after ingest, broader on-demand generation later
4. Claim-level provenance with chunk ids and source counts
5. Dirty-page invalidation with full regeneration, not append/merge
6. Separate LanceDB wiki index for retrieval
7. A minimal MCP surface: `wiki_search`, `wiki_get_page`, `wiki_generate_page`, `wiki_related`, `wiki_claims`, `wiki_queue`

The main discipline is this: keep the wiki as a synthesis layer anchored to evidence, not a second loose summarization system. If a page cannot point back to the exact chunks it stands on, it should not exist.
