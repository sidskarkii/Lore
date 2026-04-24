"""Entity co-occurrence graph with NPMI weighting and Louvain communities.

Builds a graph where nodes are canonical entities and edges represent
statistically surprising co-occurrence in the same chunk. Community
detection groups related entities into topic clusters.

Persists to ~/.lore/entity_graph.json. Rebuilt on demand.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import louvain_communities

from .config import get_config
from .entities import get_entity_index


def _npmi(cooccur: int, df_a: int, df_b: int, n: int) -> float:
    """Normalized Pointwise Mutual Information, range [-1, 1]."""
    if cooccur == 0 or n == 0:
        return 0.0
    p_ab = cooccur / n
    p_a = df_a / n
    p_b = df_b / n
    if p_a == 0 or p_b == 0 or p_ab == 0:
        return 0.0
    pmi = math.log(p_ab / (p_a * p_b))
    return pmi / -math.log(p_ab)


class EntityGraph:
    """Co-occurrence graph over canonical entities."""

    def __init__(self, min_pair_count: int = 3):
        self.min_pair_count = min_pair_count
        self.graph: nx.Graph = nx.Graph()
        self.communities: dict[str, int] = {}
        self._cfg = get_config()
        self._graph_path = self._cfg.data_dir / "entity_graph.json"

    def build(self) -> "EntityGraph":
        """Build the graph from all chunks in the store."""
        from .store import get_store

        store = get_store()
        idx = get_entity_index()

        chunk_entity_sets: list[set[str]] = []
        df: Counter[str] = Counter()
        cooccur: Counter[tuple[str, str]] = Counter()

        for coll in store.list_collections():
            chunks = store.get_all_chunks(coll["collection"])
            for c in chunks:
                ents_raw = c.get("entities", "")
                if not ents_raw:
                    continue
                try:
                    ents = json.loads(ents_raw) if isinstance(ents_raw, str) else ents_raw
                except (json.JSONDecodeError, TypeError):
                    continue

                canonical_set = set()
                for e in ents:
                    if not isinstance(e, dict):
                        continue
                    cluster = idx.resolve(e.get("name", ""))
                    if cluster:
                        canonical_set.add(cluster.canonical)

                if len(canonical_set) < 2:
                    continue

                chunk_entity_sets.append(canonical_set)
                for ent in canonical_set:
                    df[ent] += 1
                for a in canonical_set:
                    for b in canonical_set:
                        if a < b:
                            cooccur[(a, b)] += 1

        n = len(chunk_entity_sets)
        print(f"  [graph] {n} chunks with 2+ entities, {len(df)} unique entities, {len(cooccur)} pairs")

        self.graph = nx.Graph()

        for ent, count in df.items():
            cluster = idx.resolve(ent)
            self.graph.add_node(ent, type=cluster.entity_type if cluster else "UNKNOWN",
                                docfreq=count, sources=len(cluster.sources) if cluster else 0)

        edges_added = 0
        for (a, b), count in cooccur.items():
            if count < self.min_pair_count:
                continue
            score = _npmi(count, df[a], df[b], n)
            if score > 0:
                self.graph.add_edge(a, b, cooccur=count, npmi=round(score, 4))
                edges_added += 1

        print(f"  [graph] {self.graph.number_of_nodes()} nodes, {edges_added} edges (min_pair={self.min_pair_count}, npmi>0)")

        if self.graph.number_of_nodes() > 0:
            try:
                communities = louvain_communities(self.graph, weight="npmi", seed=42)
                self.communities = {}
                for i, community in enumerate(communities):
                    for node in community:
                        self.communities[node] = i
                        self.graph.nodes[node]["community"] = i
                print(f"  [graph] {len(communities)} communities detected")
            except Exception as e:
                print(f"  [graph] Community detection failed: {e}")

        try:
            self.save()
        except OSError as e:
            print(f"  [graph] Warning: could not persist: {e}")

        return self

    def neighbors(self, entity: str, n: int = 10) -> list[dict]:
        """Get top neighbors of an entity by NPMI."""
        idx = get_entity_index()
        cluster = idx.resolve(entity)
        canonical = cluster.canonical if cluster else entity

        if canonical not in self.graph:
            return []

        edges = []
        for neighbor in self.graph.neighbors(canonical):
            data = self.graph.edges[canonical, neighbor]
            edges.append({
                "entity": neighbor,
                "type": self.graph.nodes[neighbor].get("type", ""),
                "npmi": data.get("npmi", 0),
                "cooccur": data.get("cooccur", 0),
                "community": self.graph.nodes[neighbor].get("community", -1),
            })

        edges.sort(key=lambda x: -x["npmi"])
        return edges[:n]

    def community_members(self, entity: str) -> list[dict]:
        """Get all entities in the same community."""
        idx = get_entity_index()
        cluster = idx.resolve(entity)
        canonical = cluster.canonical if cluster else entity

        community_id = self.communities.get(canonical, -1)
        if community_id == -1:
            return []

        members = []
        for node, cid in self.communities.items():
            if cid == community_id and node != canonical:
                members.append({
                    "entity": node,
                    "type": self.graph.nodes[node].get("type", ""),
                    "docfreq": self.graph.nodes[node].get("docfreq", 0),
                })

        members.sort(key=lambda x: -x["docfreq"])
        return members

    def bridges(self, n: int = 10) -> list[dict]:
        """Find bridge entities that connect different communities."""
        bridge_scores = []
        for node in self.graph.nodes:
            neighbor_communities = set()
            for neighbor in self.graph.neighbors(node):
                nc = self.communities.get(neighbor, -1)
                if nc >= 0:
                    neighbor_communities.add(nc)
            own = self.communities.get(node, -1)
            if own >= 0:
                neighbor_communities.discard(own)
            if len(neighbor_communities) >= 2:
                bridge_scores.append({
                    "entity": node,
                    "type": self.graph.nodes[node].get("type", ""),
                    "communities_bridged": len(neighbor_communities) + 1,
                    "degree": self.graph.degree(node),
                })

        bridge_scores.sort(key=lambda x: (-x["communities_bridged"], -x["degree"]))
        return bridge_scores[:n]

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": len(set(self.communities.values())) if self.communities else 0,
            "bridges": len(self.bridges(50)),
        }

    def save(self):
        data = {
            "nodes": [],
            "edges": [],
            "communities": len(set(self.communities.values())) if self.communities else 0,
        }
        for node, attrs in self.graph.nodes(data=True):
            data["nodes"].append({
                "entity": node, "type": attrs.get("type", ""),
                "docfreq": attrs.get("docfreq", 0), "sources": attrs.get("sources", 0),
                "community": attrs.get("community", -1),
            })
        for a, b, attrs in self.graph.edges(data=True):
            data["edges"].append({
                "a": a, "b": b,
                "cooccur": attrs.get("cooccur", 0), "npmi": attrs.get("npmi", 0),
            })

        self._graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph_path.write_text(json.dumps(data, indent=2))
        print(f"  [graph] Saved to {self._graph_path}")

    def load(self) -> bool:
        if not self._graph_path.exists():
            return False
        try:
            data = json.loads(self._graph_path.read_text())
            self.graph = nx.Graph()
            for n in data.get("nodes", []):
                self.graph.add_node(n["entity"], type=n.get("type", ""),
                                    docfreq=n.get("docfreq", 0), sources=n.get("sources", 0),
                                    community=n.get("community", -1))
                if n.get("community", -1) >= 0:
                    self.communities[n["entity"]] = n["community"]
            for e in data.get("edges", []):
                self.graph.add_edge(e["a"], e["b"], cooccur=e.get("cooccur", 0), npmi=e.get("npmi", 0))
            print(f"  [graph] Loaded {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            return True
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"  [graph] Failed to load: {e}")
            return False


_graph: EntityGraph | None = None


def get_entity_graph(rebuild: bool = False) -> EntityGraph:
    global _graph
    if _graph is None or rebuild:
        _graph = EntityGraph()
        if not rebuild and _graph.load():
            return _graph
        _graph.build()
    return _graph
