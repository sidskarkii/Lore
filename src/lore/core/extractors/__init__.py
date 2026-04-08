from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedDocument:
    sections: list[dict]   # [{"title": str, "text": str}, ...]
    metadata: dict          # source-specific metadata
    source_type: str        # "pdf", "epub", "web", "text", "code", "audio"
    file_path: str = ""


_EXTENSION_MAP: dict[str, str] = {
    # document
    ".pdf": "pdf",
    ".epub": "epub",
    # text / markup
    ".md": "text",
    ".txt": "text",
    ".rst": "text",
    ".html": "text",
    ".htm": "text",
    # code
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".java": "code",
    ".go": "code",
    ".rs": "code",
    ".cpp": "code",
    ".c": "code",
    ".cs": "code",
    ".rb": "code",
    ".php": "code",
    # audio
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".aac": "audio",
    ".wma": "audio",
    # video (audio track extracted)
    ".mp4": "audio",
    ".mkv": "audio",
    ".avi": "audio",
    ".mov": "audio",
    ".webm": "audio",
}


def detect_source_type(path: str) -> str:
    """Return the source type for *path* based on its file extension.

    Falls back to ``"text"`` for unknown extensions.
    """
    ext = Path(path).suffix.lower()
    return _EXTENSION_MAP.get(ext, "text")


def extract(path: str, source_type: str | None = None) -> ExtractedDocument:
    """Extract content from *path*, routing to the appropriate extractor.

    *source_type* is auto-detected from the file extension when not supplied.
    Extractor modules are imported lazily so their heavy optional dependencies
    are only loaded when actually needed.
    """
    if source_type is None:
        source_type = detect_source_type(path)

    if source_type == "pdf":
        from .pdf import extract_pdf  # type: ignore[import]
        return extract_pdf(path)
    elif source_type == "epub":
        from .epub import extract_epub  # type: ignore[import]
        return extract_epub(path)
    elif source_type == "code":
        from .code import extract_code  # type: ignore[import]
        return extract_code(path)
    elif source_type == "audio":
        from .audio import extract_audio  # type: ignore[import]
        return extract_audio(path)
    elif source_type == "web":
        from .web import extract_web  # type: ignore[import]
        return extract_web(path)
    else:
        from .text import extract_text  # type: ignore[import]
        return extract_text(path)


def extract_url(url: str) -> ExtractedDocument:
    """Fetch and extract content from *url* using the web extractor."""
    from .web import extract_web  # type: ignore[import]
    return extract_web(url)
