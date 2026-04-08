"""Text/Markdown/RST extraction — structure-aware splitting."""

from __future__ import annotations

import re
from pathlib import Path

from . import ExtractedDocument


def extract_text(path: str) -> ExtractedDocument:
    """Extract text from markdown, RST, plain text, or HTML files."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = p.read_text(encoding="utf-8", errors="replace")
    ext = p.suffix.lower()

    if ext in (".md", ".markdown"):
        sections = _split_markdown(content)
    elif ext in (".html", ".htm"):
        sections = _split_html(content)
    elif ext == ".rst":
        sections = _split_rst(content)
    else:
        sections = [{"title": p.stem, "text": content.strip()}]

    sections = [s for s in sections if s["text"].strip()]

    return ExtractedDocument(
        sections=sections,
        metadata={"filename": p.name},
        source_type="text",
        file_path=str(p),
    )


def _split_markdown(content: str) -> list[dict]:
    """Split markdown by headers."""
    sections: list[dict] = []
    current_title = ""
    current_lines: list[str] = []

    for line in content.split("\n"):
        match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if match:
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append({"title": current_title, "text": text})
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({"title": current_title, "text": text})

    return sections if sections else [{"title": "", "text": content.strip()}]


def _split_html(content: str) -> list[dict]:
    """Extract text from HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    return [{"title": title, "text": text}]


def _split_rst(content: str) -> list[dict]:
    """Split RST by section headers (underlined with =, -, ~)."""
    lines = content.split("\n")
    sections: list[dict] = []
    current_title = ""
    current_lines: list[str] = []

    i = 0
    while i < len(lines):
        if (i + 1 < len(lines)
                and lines[i].strip()
                and re.match(r"^[=\-~^\"]{3,}$", lines[i + 1].strip())
                and len(lines[i + 1].strip()) >= len(lines[i].strip())):
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append({"title": current_title, "text": text})
            current_title = lines[i].strip()
            current_lines = []
            i += 2
        else:
            current_lines.append(lines[i])
            i += 1

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({"title": current_title, "text": text})

    return sections if sections else [{"title": "", "text": content.strip()}]
