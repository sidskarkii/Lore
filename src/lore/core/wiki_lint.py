"""Wiki lint/audit — deterministic health checks for the wiki layer.

Checks: orphan pages, stale pages, weak claims, broken provenance,
dangling links, generation drift, source gaps. No LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Finding:
    check: str
    severity: str  # error, warning, info
    page_id: str = ""
    claim_id: str = ""
    message: str = ""


_VALID_CHECKS = {
    "stale", "orphan", "weak_claims", "claim_summary",
    "broken_links", "broken_provenance", "generation_drift", "source_gaps",
}


def lint_wiki(checks: list[str] | None = None) -> dict:
    """Run wiki lint checks. Returns structured findings grouped by severity."""
    from .wiki import get_wiki_manager
    wm = get_wiki_manager()

    all_checks = {
        "stale": _check_stale,
        "orphan": _check_orphan,
        "weak_claims": _check_weak_claims,
        "claim_summary": _check_claim_summary,
        "broken_links": _check_broken_links,
        "broken_provenance": _check_broken_provenance,
        "generation_drift": _check_generation_drift,
        "source_gaps": _check_source_gaps,
    }

    run = checks or list(all_checks.keys())
    findings: list[Finding] = []

    unknown = [c for c in run if c not in _VALID_CHECKS]
    if unknown:
        findings.append(Finding(
            check="lint", severity="warning",
            message=f"Unknown checks ignored: {', '.join(unknown)}. Valid: {', '.join(sorted(_VALID_CHECKS))}",
        ))

    for name in run:
        fn = all_checks.get(name)
        if fn:
            findings.extend(fn(wm))

    by_severity: dict[str, list[dict]] = {"error": [], "warning": [], "info": []}
    for f in findings:
        entry = {"check": f.check, "message": f.message}
        if f.page_id:
            entry["page_id"] = f.page_id
        if f.claim_id:
            entry["claim_id"] = f.claim_id
        by_severity.get(f.severity, by_severity["info"]).append(entry)

    return {
        "total_findings": len(findings),
        "errors": len(by_severity["error"]),
        "warnings": len(by_severity["warning"]),
        "info": len(by_severity["info"]),
        "findings": by_severity,
    }


def _check_stale(wm) -> list[Finding]:
    stale = wm.get_stale_pages()
    findings = [
        Finding(
            check="stale", severity="warning", page_id=p["page_id"],
            message="Page is stale and needs regeneration",
        )
        for p in stale
    ]
    if stale:
        findings.append(Finding(
            check="stale", severity="info",
            message=f"{len(stale)} stale pages total",
        ))
    return findings


def _check_orphan(wm) -> list[Finding]:
    pages = wm.list_pages()
    backlinks = wm._load_backlinks()

    all_related: set[str] = set()
    for meta in pages:
        page = wm.get_page(meta["page_id"])
        if page:
            all_related.update(page.related_pages)

    findings = []
    for meta in pages:
        pid = meta["page_id"]
        if meta.get("page_type") == "source":
            continue
        has_backlinks = pid in backlinks and len(backlinks[pid]) > 0
        is_referenced = pid in all_related
        if not has_backlinks and not is_referenced:
            findings.append(Finding(
                check="orphan", severity="warning", page_id=pid,
                message="No backlinks and not referenced by any other page",
            ))
    return findings


def _check_weak_claims(wm) -> list[Finding]:
    findings = []
    for meta in wm.list_pages():
        page = wm.get_page(meta["page_id"])
        if not page or not page.generation:
            continue
        claims = page.generation.get("claims", [])
        for c in claims:
            cid = c.get("claim_id", "")
            status = c.get("status", "")
            support = c.get("support_count", 0)

            if status in ("review", "conflicted"):
                findings.append(Finding(
                    check="weak_claims", severity="warning",
                    page_id=page.page_id, claim_id=cid,
                    message=f"Claim status '{status}': {c.get('text', '')[:80]}",
                ))
            elif support <= 1 and status != "supported":
                findings.append(Finding(
                    check="weak_claims", severity="info",
                    page_id=page.page_id, claim_id=cid,
                    message=f"Low support ({support}): {c.get('text', '')[:80]}",
                ))
    return findings


def _check_claim_summary(wm) -> list[Finding]:
    totals: dict[str, int] = {}
    page_count = 0
    claim_count = 0

    for meta in wm.list_pages():
        page = wm.get_page(meta["page_id"])
        if not page or not page.generation:
            continue
        claims = page.generation.get("claims", [])
        if not claims:
            continue
        page_count += 1
        claim_count += len(claims)
        for c in claims:
            s = c.get("status", "unknown")
            totals[s] = totals.get(s, 0) + 1

    parts = [f"{s}: {n}" for s, n in sorted(totals.items(), key=lambda x: -x[1])]
    return [Finding(
        check="claim_summary", severity="info",
        message=f"{claim_count} claims across {page_count} pages — {', '.join(parts)}",
    )]


def _check_broken_links(wm) -> list[Finding]:
    findings = []

    for meta in wm.list_pages():
        page = wm.get_page(meta["page_id"])
        if not page:
            continue

        seen = set()
        for related in page.related_pages:
            if related == page.page_id:
                findings.append(Finding(
                    check="broken_links", severity="warning",
                    page_id=page.page_id,
                    message="Self-link in related_pages",
                ))
            elif related in seen:
                findings.append(Finding(
                    check="broken_links", severity="warning",
                    page_id=page.page_id,
                    message=f"Duplicate link: {related}",
                ))
            elif not wm.get_page(related):
                findings.append(Finding(
                    check="broken_links", severity="warning",
                    page_id=page.page_id,
                    message=f"Links to nonexistent page: {related}",
                ))
            seen.add(related)
    return findings


def _check_broken_provenance(wm) -> list[Finding]:
    findings = []
    try:
        from .store import get_store
        store = get_store()
    except Exception:
        return [Finding(
            check="broken_provenance", severity="info",
            message="Could not access chunk store — skipping provenance check",
        )]

    for meta in wm.list_pages():
        if meta.get("page_type") == "source":
            continue
        page = wm.get_page(meta["page_id"])
        if not page or not page.generation:
            continue
        claims = page.generation.get("claims", [])
        for c in claims:
            for cid in c.get("chunk_ids", []):
                if not store.get_chunk_by_id(cid):
                    findings.append(Finding(
                        check="broken_provenance", severity="error",
                        page_id=page.page_id, claim_id=c.get("claim_id", ""),
                        message=f"Claim references missing chunk: {cid}",
                    ))
    return findings


def _check_generation_drift(wm) -> list[Finding]:
    findings = []
    for meta in wm.list_pages():
        page = wm.get_page(meta["page_id"])
        if not page:
            continue

        if page.status == "generated" and not page.content.strip():
            findings.append(Finding(
                check="generation_drift", severity="error",
                page_id=page.page_id,
                message="Page has status 'generated' but empty content",
            ))

        if not page.generation:
            if page.page_type != "source":
                findings.append(Finding(
                    check="generation_drift", severity="warning",
                    page_id=page.page_id,
                    message="Non-source page missing generation metadata",
                ))
            continue

        declared = page.generation.get("claim_count", 0)
        actual = len(page.generation.get("claims", []))
        if declared != actual:
            findings.append(Finding(
                check="generation_drift", severity="error",
                page_id=page.page_id,
                message=f"claim_count mismatch: declared {declared}, actual {actual}",
            ))

        if not page.generation.get("inputs_hash") and page.page_type != "source":
            findings.append(Finding(
                check="generation_drift", severity="warning",
                page_id=page.page_id,
                message="Missing inputs_hash in generation metadata",
            ))
    return findings


def _parse_tags(raw) -> list[str]:
    """Parse concept_tags from chunk — handles CSV strings, JSON arrays, lists."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip().lower() for t in raw if t]
    raw = str(raw).strip()
    if raw.startswith("["):
        try:
            import json
            parsed = json.loads(raw)
            return [str(t).strip().lower() for t in parsed if t]
        except (ValueError, TypeError):
            pass
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


def _check_source_gaps(wm) -> list[Finding]:
    findings = []
    existing_entity_slugs: set[str] = set()
    existing_concept_slugs: set[str] = set()

    for meta in wm.list_pages():
        pid = meta["page_id"]
        if pid.startswith("entity/"):
            existing_entity_slugs.add(pid.split("/", 1)[1])
        elif pid.startswith("concept/"):
            existing_concept_slugs.add(pid.split("/", 1)[1])

    try:
        from .entities import get_entity_index
        from .wiki import _slug
        idx = get_entity_index()
        cross_source = idx.get_cross_source_entities()
        for cluster in cross_source:
            slug = _slug(cluster.canonical)
            if slug and slug not in existing_entity_slugs and cluster.count >= 3:
                findings.append(Finding(
                    check="source_gaps", severity="warning",
                    message=f"Entity '{cluster.canonical}' ({cluster.count} mentions, "
                            f"{len(cluster.sources)} sources) has no wiki page",
                ))
    except Exception as e:
        findings.append(Finding(
            check="source_gaps", severity="info",
            message=f"Could not check entity gaps: {e}",
        ))

    try:
        from .store import get_store
        from .wiki import _slug
        store = get_store()
        tag_sources: dict[str, set[str]] = {}
        for coll in store.list_collections():
            coll_id = coll["collection"]
            chunks = store.get_all_chunks(coll_id)
            for ch in chunks:
                for t in _parse_tags(ch.get("concept_tags", "")):
                    if t:
                        tag_sources.setdefault(t, set()).add(coll_id)

        for tag, sources in tag_sources.items():
            if len(sources) >= 2:
                slug = _slug(tag)
                if slug and slug not in existing_concept_slugs:
                    findings.append(Finding(
                        check="source_gaps", severity="info",
                        message=f"Concept tag '{tag}' spans {len(sources)} sources but has no wiki page",
                    ))
    except Exception as e:
        findings.append(Finding(
            check="source_gaps", severity="info",
            message=f"Could not check concept tag gaps: {e}",
        ))

    return findings
