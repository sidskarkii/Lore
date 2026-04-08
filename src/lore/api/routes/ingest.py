"""Ingestion endpoints — trigger content ingestion from the UI."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["ingest"])


class IngestFolderRequest(BaseModel):
    """Ingest video/audio files from a local folder."""
    path: str = Field(..., description="Path to folder containing video/audio files")
    name: str = Field(..., description="Collection display name")
    topic: str = Field(..., description="Topic category (e.g. '3d', 'ai', 'code')")
    subtopic: str = Field(..., description="Subtopic (e.g. 'blender', 'houdini')")
    language: str | None = Field("en", description="Language code or null for auto-detect")
    contextual: bool = Field(False, description="Generate contextual prefixes (slower, better retrieval)")


class IngestYouTubeRequest(BaseModel):
    """Ingest from a YouTube URL or playlist."""
    url: str = Field(..., description="YouTube video or playlist URL")
    name: str = Field(..., description="Collection display name")
    topic: str = Field(..., description="Topic category")
    subtopic: str = Field(..., description="Subtopic")
    language: str | None = Field("en", description="Language code or null for auto-detect")
    contextual: bool = Field(False, description="Generate contextual prefixes")


class IngestDocumentsRequest(BaseModel):
    """Ingest documents from a file or folder."""
    path: str = Field(..., description="Path to file or folder")
    name: str = Field(..., description="Collection display name")
    topic: str = Field(..., description="Topic category")
    subtopic: str = Field(..., description="Subtopic")
    contextual: bool = Field(False, description="Generate contextual prefixes")


class IngestFileRequest(BaseModel):
    """Ingest any supported file type (auto-detects format)."""
    path: str = Field(..., description="Path to file")
    name: str = Field(..., description="Collection display name")
    topic: str = Field("", description="Topic category")
    subtopic: str = Field("", description="Subtopic")
    source_type: str | None = Field(None, description="Override auto-detection (pdf, epub, text, code, audio)")
    enrich: bool = Field(True, description="Run keyword + entity extraction")


class IngestUrlRequest(BaseModel):
    """Ingest a web page URL."""
    url: str = Field(..., description="Web page URL to ingest")
    name: str = Field(..., description="Collection display name")
    topic: str = Field("", description="Topic category")
    subtopic: str = Field("", description="Subtopic")
    enrich: bool = Field(True, description="Run keyword + entity extraction")


class IngestResponse(BaseModel):
    """Result of an ingestion job."""
    success: bool
    chunks: int = 0
    message: str = ""
    error: str | None = None


@router.post(
    "/api/ingest/folder",
    response_model=IngestResponse,
    summary="Ingest video/audio files from a folder",
    description="Transcribes all media files, chunks, embeds, and stores them. Uses existing SRT files if present.",
)
def ingest_folder(req: IngestFolderRequest):
    from ...core.ingest import Ingester
    try:
        ingester = Ingester()
        chunks = ingester.ingest_folder(
            folder=req.path, name=req.name, topic=req.topic,
            subtopic=req.subtopic, language=req.language, contextual=req.contextual,
        )
        return IngestResponse(success=True, chunks=chunks, message=f"Ingested {chunks} chunks")
    except Exception as e:
        return IngestResponse(success=False, error=str(e))


@router.post(
    "/api/ingest/youtube",
    response_model=IngestResponse,
    summary="Ingest from YouTube URL or playlist",
    description="Downloads auto-subs or audio, transcribes, chunks, embeds, and stores.",
)
def ingest_youtube(req: IngestYouTubeRequest):
    from ...core.ingest import Ingester
    try:
        ingester = Ingester()
        chunks = ingester.ingest_youtube(
            url=req.url, name=req.name, topic=req.topic,
            subtopic=req.subtopic, language=req.language, contextual=req.contextual,
        )
        return IngestResponse(success=True, chunks=chunks, message=f"Ingested {chunks} chunks")
    except Exception as e:
        return IngestResponse(success=False, error=str(e))


@router.post(
    "/api/ingest/documents",
    response_model=IngestResponse,
    summary="Ingest documents (markdown, PDF, text)",
    description="Parses documents, chunks by structure, embeds, and stores.",
)
def ingest_documents(req: IngestDocumentsRequest):
    from ...core.ingest import Ingester
    try:
        ingester = Ingester()
        chunks = ingester.ingest_documents(
            path=req.path, name=req.name, topic=req.topic,
            subtopic=req.subtopic, contextual=req.contextual,
        )
        return IngestResponse(success=True, chunks=chunks, message=f"Ingested {chunks} chunks")
    except Exception as e:
        return IngestResponse(success=False, error=str(e))


@router.post(
    "/api/ingest/file",
    response_model=IngestResponse,
    summary="Ingest any file (auto-detects type)",
    description="Supports PDF, EPUB, markdown, text, code, audio, video. Auto-detects format from extension.",
)
def ingest_file(req: IngestFileRequest):
    from ...core.ingest import Ingester
    try:
        ingester = Ingester()
        chunks = ingester.ingest_file(
            path=req.path, name=req.name, topic=req.topic,
            subtopic=req.subtopic, source_type=req.source_type, enrich=req.enrich,
        )
        return IngestResponse(success=True, chunks=chunks, message=f"Ingested {chunks} chunks")
    except Exception as e:
        return IngestResponse(success=False, error=str(e))


@router.post(
    "/api/ingest/url",
    response_model=IngestResponse,
    summary="Ingest a web page",
    description="Fetches URL, extracts content, chunks, enriches, and stores.",
)
def ingest_url(req: IngestUrlRequest):
    from ...core.ingest import Ingester
    try:
        ingester = Ingester()
        chunks = ingester.ingest_url(
            url=req.url, name=req.name, topic=req.topic,
            subtopic=req.subtopic, enrich=req.enrich,
        )
        return IngestResponse(success=True, chunks=chunks, message=f"Ingested {chunks} chunks")
    except Exception as e:
        return IngestResponse(success=False, error=str(e))
