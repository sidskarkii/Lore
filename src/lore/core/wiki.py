"""Wiki layer — synthesis pages over the chunk store.

Manages wiki pages as markdown files with YAML frontmatter in ~/.lore/wiki/.
Pages are the source of truth; LanceDB indexes derived fragments for search.

Page types: source, entity, concept, comparison (phase 2).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import get_config


def _slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return s[:max_len].rstrip('-')


def _inputs_hash(chunk_ids: list[str]) -> str:
    """Deterministic hash of the evidence set used to generate a page."""
    return hashlib.sha256("|".join(sorted(chunk_ids)).encode()).hexdigest()[:16]


@dataclass
class WikiPage:
    page_id: str
    page_type: str
    title: str
    slug: str
    status: str = "generated"
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    source_collections: list[str] = field(default_factory=list)
    source_chunk_count: int = 0
    supporting_source_count: int = 0
    corroboration_level: str = "low"
    confidence: str = "medium"
    generation: dict = field(default_factory=dict)
    related_pages: list[str] = field(default_factory=list)
    canonical_sources: list[dict] = field(default_factory=list)
    content: str = ""

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_frontmatter(self) -> dict:
        d = {
            "page_id": self.page_id,
            "page_type": self.page_type,
            "title": self.title,
            "slug": self.slug,
            "status": self.status,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_collections": self.source_collections,
            "source_chunk_count": self.source_chunk_count,
            "supporting_source_count": self.supporting_source_count,
            "corroboration_level": self.corroboration_level,
            "confidence": self.confidence,
            "related_pages": self.related_pages,
        }
        if self.canonical_sources:
            d["canonical_sources"] = self.canonical_sources
        if self.generation:
            d["generation"] = self.generation
        return d

    def to_markdown(self) -> str:
        fm = yaml.dump(self.to_frontmatter(), default_flow_style=False, sort_keys=False)
        return f"---\n{fm}---\n\n{self.content}"

    @classmethod
    def from_markdown(cls, text: str, page_id: str = "") -> "WikiPage":
        if not text.startswith("---"):
            return cls(page_id=page_id, page_type="unknown", title="", slug="", content=text)

        parts = text.split("---", 2)
        if len(parts) < 3:
            return cls(page_id=page_id, page_type="unknown", title="", slug="", content=text)

        try:
            meta = yaml.safe_load(parts[1]) or {}
        except Exception:
            meta = {}

        body = parts[2]
        if body.startswith("\n\n"):
            body = body[2:]
        elif body.startswith("\n"):
            body = body[1:]

        return cls(
            page_id=meta.get("page_id", page_id),
            page_type=meta.get("page_type", "unknown"),
            title=meta.get("title", ""),
            slug=meta.get("slug", ""),
            status=meta.get("status", "generated"),
            version=meta.get("version", 1),
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
            source_collections=meta.get("source_collections", []),
            source_chunk_count=meta.get("source_chunk_count", 0),
            supporting_source_count=meta.get("supporting_source_count", 0),
            corroboration_level=meta.get("corroboration_level", "low"),
            confidence=meta.get("confidence", "medium"),
            generation=meta.get("generation", {}),
            related_pages=meta.get("related_pages", []),
            canonical_sources=meta.get("canonical_sources", []),
            content=body,
        )


class WikiManager:
    """Manages wiki pages: read/write, manifest, dirty tracking."""

    def __init__(self):
        self._cfg = get_config()
        self._wiki_dir = self._cfg.data_dir / "wiki"
        self._pages_dir = self._wiki_dir / "pages"
        self._manifests_dir = self._wiki_dir / "manifests"
        self._state_dir = self._wiki_dir / "state"
        self._ensure_dirs()
        self._manifest: dict[str, dict] = {}
        self._load_manifest()

    def _ensure_dirs(self):
        for d in [
            self._pages_dir / "entity",
            self._pages_dir / "concept",
            self._pages_dir / "source",
            self._pages_dir / "comparison",
            self._manifests_dir,
            self._state_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self) -> Path:
        return self._manifests_dir / "pages.json"

    def _backlinks_path(self) -> Path:
        return self._manifests_dir / "backlinks.json"

    def _dirty_queue_path(self) -> Path:
        return self._state_dir / "dirty_queue.json"

    def _load_manifest(self):
        path = self._manifest_path()
        if path.exists():
            try:
                self._manifest = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._manifest = {}

    def _save_manifest(self):
        self._manifest_path().write_text(json.dumps(self._manifest, indent=2))

    def _page_path(self, page_id: str) -> Path:
        return self._pages_dir / f"{page_id}.md"

    # ── Read/Write ──────────────────────────────────────────────

    def get_page(self, page_id: str) -> WikiPage | None:
        path = self._page_path(page_id)
        if not path.exists():
            return None
        try:
            return WikiPage.from_markdown(path.read_text(), page_id=page_id)
        except OSError:
            return None

    def save_page(self, page: WikiPage):
        page.updated_at = datetime.now(timezone.utc).isoformat()
        path = self._page_path(page.page_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page.to_markdown())

        self._manifest[page.page_id] = {
            "title": page.title,
            "page_type": page.page_type,
            "slug": page.slug,
            "status": page.status,
            "updated_at": page.updated_at,
            "source_collections": page.source_collections,
            "supporting_source_count": page.supporting_source_count,
        }
        self._save_manifest()
        self.clear_stale(page.page_id)
        self._update_backlinks_for(page)

    def delete_page(self, page_id: str) -> bool:
        path = self._page_path(page_id)
        if path.exists():
            path.unlink()
        existed = page_id in self._manifest
        if existed:
            del self._manifest[page_id]
            self._save_manifest()
        dirty = self._load_dirty_queue()
        if page_id in dirty:
            del dirty[page_id]
            self._save_dirty_queue(dirty)
        self._remove_backlinks_for(page_id)
        return existed

    # ── List/Query ──────────────────────────────────────────────

    def list_pages(self, page_type: str | None = None, include_stale: bool = True) -> list[dict]:
        results = []
        for page_id, meta in self._manifest.items():
            if page_type and meta.get("page_type") != page_type:
                continue
            if not include_stale and meta.get("status") == "stale":
                continue
            results.append({"page_id": page_id, **meta})
        return results

    def page_exists(self, page_id: str) -> bool:
        return page_id in self._manifest

    # ── Dirty Tracking ──────────────────────────────────────────

    def mark_stale(self, page_ids: list[str], reason: str = ""):
        dirty = self._load_dirty_queue()
        for pid in page_ids:
            if pid in self._manifest:
                self._manifest[pid]["status"] = "stale"
            dirty[pid] = {
                "reason": reason,
                "marked_at": datetime.now(timezone.utc).isoformat(),
            }
        self._save_manifest()
        self._save_dirty_queue(dirty)

    def get_stale_pages(self) -> list[dict]:
        return [
            {"page_id": pid, **meta}
            for pid, meta in self._manifest.items()
            if meta.get("status") == "stale"
        ]

    def clear_stale(self, page_id: str):
        if page_id in self._manifest:
            self._manifest[page_id]["status"] = "generated"
            self._save_manifest()
        dirty = self._load_dirty_queue()
        dirty.pop(page_id, None)
        self._save_dirty_queue(dirty)

    def _load_dirty_queue(self) -> dict:
        path = self._dirty_queue_path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_dirty_queue(self, dirty: dict):
        self._dirty_queue_path().write_text(json.dumps(dirty, indent=2))

    # ── Backlinks ──────────────────────────────────────────────

    def _load_backlinks(self) -> dict[str, list[str]]:
        path = self._backlinks_path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_backlinks(self, backlinks: dict[str, list[str]]):
        self._backlinks_path().write_text(json.dumps(backlinks, indent=2))

    def rebuild_backlinks(self):
        backlinks: dict[str, list[str]] = {}
        for page_id in self._manifest:
            page = self.get_page(page_id)
            if not page:
                continue
            for related in page.related_pages:
                backlinks.setdefault(related, []).append(page_id)
        self._save_backlinks(backlinks)

    def _update_backlinks_for(self, page: WikiPage):
        """Incrementally update backlinks when a page is saved."""
        bl = self._load_backlinks()
        self._remove_from_backlinks(bl, page.page_id)
        for related in page.related_pages:
            bl.setdefault(related, []).append(page.page_id)
        self._save_backlinks(bl)

    def _remove_backlinks_for(self, page_id: str):
        """Remove all backlink entries for a deleted page."""
        bl = self._load_backlinks()
        self._remove_from_backlinks(bl, page_id)
        self._save_backlinks(bl)

    def _remove_from_backlinks(self, bl: dict[str, list[str]], page_id: str):
        for target in list(bl.keys()):
            if page_id in bl[target]:
                bl[target].remove(page_id)
                if not bl[target]:
                    del bl[target]

    def get_backlinks(self, page_id: str) -> list[str]:
        return self._load_backlinks().get(page_id, [])

    # ── Source Page Adapter ──────────────────────────────────────

    def generate_source_pages(self, force: bool = False) -> int:
        """Convert existing book_summary.json archives into wiki source pages.
        No LLM calls — pure reformatting of existing data.
        Set force=True to regenerate existing pages from updated archives.
        """
        archive_dir = self._cfg.archive_dir
        if not archive_dir.exists():
            return 0

        count = 0
        for coll_dir in sorted(archive_dir.iterdir()):
            if not coll_dir.is_dir():
                continue
            summary_file = coll_dir / "book_summary.json"
            meta_file = coll_dir / "meta.json"
            if not summary_file.exists():
                continue

            try:
                summary = json.loads(summary_file.read_text())
                meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            except (json.JSONDecodeError, OSError):
                continue

            collection = coll_dir.name
            display = meta.get("collection_display", collection.replace("_", " "))
            slug = _slug(display)
            page_id = f"source/{slug}"

            if not force and self.page_exists(page_id):
                continue

            overview = summary.get("overview", "")
            themes = summary.get("main_themes", [])
            takeaways = summary.get("key_takeaways", [])
            tags = summary.get("tags", [])
            cross_patterns = summary.get("cross_section_patterns", [])

            sections = [f"# {display}\n"]

            if overview:
                sections.append(f"## Overview\n\n{overview}\n")

            if themes:
                sections.append("## Main Themes\n")
                for t in themes:
                    if isinstance(t, dict):
                        name = t.get("theme", t.get("title", ""))
                        desc = t.get("description", t.get("summary", ""))
                        if name:
                            sections.append(f"### {name}\n\n{desc}\n")
                    elif isinstance(t, str):
                        sections.append(f"- {t}")

            if takeaways:
                sections.append("## Key Takeaways\n")
                for tk in takeaways:
                    sections.append(f"- {tk}")
                sections.append("")

            if cross_patterns:
                sections.append("## Cross-Section Patterns\n")
                for p in cross_patterns:
                    sections.append(f"- {p}")
                sections.append("")

            section_file = coll_dir / "section_summaries.json"
            if section_file.exists():
                try:
                    sec_data = json.loads(section_file.read_text())
                    if sec_data:
                        sections.append("## Sections\n")
                        for sec in sec_data:
                            if not isinstance(sec, dict):
                                continue
                            heading = sec.get("heading", sec.get("section", ""))
                            sec_summary = sec.get("summary", "")
                            if not heading or not sec_summary:
                                continue
                            sections.append(f"### {heading}\n\n{sec_summary}\n")
                            concepts = sec.get("key_concepts", [])
                            if concepts:
                                sections.append("**Key concepts:** " + ", ".join(concepts) + "\n")
                            entities = sec.get("key_entities", [])
                            if entities:
                                sections.append("**Key entities:** " + ", ".join(entities) + "\n")
                            tensions = sec.get("tensions", [])
                            if tensions:
                                sections.append("**Tensions:** " + ", ".join(tensions) + "\n")
                            notable = sec.get("notable_points", [])
                            if notable:
                                for np in notable:
                                    sections.append(f"- {np}")
                                sections.append("")
                except (json.JSONDecodeError, OSError):
                    pass

            related = [f"concept/{_slug(t)}" for t in tags[:10]] if tags else []

            hash_parts = summary_file.read_bytes()
            if meta_file.exists():
                hash_parts += meta_file.read_bytes()
            if section_file.exists():
                hash_parts += section_file.read_bytes()
            chunks_file = coll_dir / "chunks.json"
            if chunks_file.exists():
                hash_parts += chunks_file.read_bytes()
            archive_hash = hashlib.sha256(hash_parts).hexdigest()[:16]

            existing = self.get_page(page_id)
            version = (existing.version + 1) if existing else 1

            page = WikiPage(
                page_id=page_id,
                page_type="source",
                title=display,
                slug=slug,
                version=version,
                source_collections=[collection],
                source_chunk_count=meta.get("chunk_count", 0),
                supporting_source_count=1,
                corroboration_level="single_source",
                confidence="derived",
                generation={
                    "strategy": "archive_adapter",
                    "model": "none",
                    "inputs_hash": archive_hash,
                    "source_versions": {collection: archive_hash},
                },
                canonical_sources=[{"collection": collection, "weight": "primary"}],
                related_pages=related,
                content="\n".join(sections),
            )

            self.save_page(page)
            count += 1
            print(f"  [wiki] {'Refreshed' if existing else 'Created'} source page: {page_id}")

        return count

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for meta in self._manifest.values():
            t = meta.get("page_type", "unknown")
            s = meta.get("status", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
            by_status[s] = by_status.get(s, 0) + 1
        return {
            "total_pages": len(self._manifest),
            "by_type": by_type,
            "by_status": by_status,
        }


_wiki_manager: WikiManager | None = None


def get_wiki_manager() -> WikiManager:
    global _wiki_manager
    if _wiki_manager is None:
        _wiki_manager = WikiManager()
    return _wiki_manager
