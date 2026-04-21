"""Ingestion pipeline — orchestrates source -> extract -> chunk -> embed -> store.

Supports:
  - Local video/audio files (mp4, mp3, wav, m4a, etc.)
  - YouTube URLs and playlists (via yt-dlp)
  - Local documents (markdown, RST, PDF, EPUB, code)
  - Web pages (via trafilatura)
  - Pre-existing SRT/transcript files
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunk import chunk_segments, chunk_sections
from .config import get_config
from .store import Store, get_store
from .transcribe import Transcriber


@dataclass
class IngestionProgress:
    """Progress update for the UI."""
    stage: str
    progress: float
    current_item: str
    total_items: int
    completed_items: int
    message: str = ""
    error: str | None = None


ProgressCallback = Callable[[IngestionProgress], None] | None


def _sanitize(name: str, maxlen: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:maxlen]


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
SUB_EXTS = {".srt", ".vtt"}


def _find_yt_dlp() -> str:
    """Find yt-dlp binary."""
    path = shutil.which("yt-dlp")
    if path:
        return path
    raise FileNotFoundError("yt-dlp not found. Install with: pip install yt-dlp")


def _find_ffmpeg() -> str | None:
    """Find ffmpeg binary (optional, for audio extraction)."""
    return shutil.which("ffmpeg")


class Ingester:
    """Orchestrates the full ingestion pipeline."""

    def __init__(self, store: Store | None = None):
        self.store = store or get_store()
        self.transcriber = Transcriber()
        self._cfg = get_config()

    # ── Video/Audio folder ────────────────────────────────────────────

    def ingest_folder(
        self,
        folder: str | Path,
        name: str,
        topic: str,
        subtopic: str,
        language: str | None = None,
        contextual: bool = False,
        on_progress: ProgressCallback = None,
    ) -> int:
        """Ingest all video/audio files from a local folder."""
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in VIDEO_EXTS | AUDIO_EXTS
        ])
        if not files:
            raise ValueError(f"No video/audio files found in {folder}")

        collection = _sanitize(name)
        total_chunks = 0

        for i, media_file in enumerate(files, 1):
            episode_title = media_file.stem
            srt_file = media_file.with_suffix(".srt")

            if on_progress:
                on_progress(IngestionProgress(
                    stage="transcribing",
                    progress=i / len(files),
                    current_item=episode_title,
                    total_items=len(files),
                    completed_items=i - 1,
                ))

            if srt_file.exists():
                print(f"[{i}/{len(files)}] Loading SRT: {episode_title}")
                segments = self.transcriber.load_srt(srt_file)
            else:
                print(f"[{i}/{len(files)}] Transcribing: {episode_title}")
                segments = self.transcriber.transcribe(media_file, language=language)
                self.transcriber.save_srt(segments, srt_file)

            n = self._chunk_and_store_segments(
                segments=segments,
                collection=collection,
                display_name=name,
                topic=topic,
                subtopic=subtopic,
                episode_num=i,
                episode_title=episode_title,
                url=str(media_file),
                source_type="video",
                contextual=contextual,
            )
            total_chunks += n
            print(f"  -> {n} chunks stored")

        if on_progress:
            on_progress(IngestionProgress(
                stage="done", progress=1.0, current_item="",
                total_items=len(files), completed_items=len(files),
                message=f"{total_chunks} chunks from {len(files)} files",
            ))

        return total_chunks

    # ── YouTube ───────────────────────────────────────────────────────

    def ingest_youtube(
        self,
        url: str,
        name: str,
        topic: str,
        subtopic: str,
        language: str | None = None,
        contextual: bool = False,
        on_progress: ProgressCallback = None,
    ) -> int:
        """Ingest from a YouTube URL or playlist."""
        yt_dlp = _find_yt_dlp()
        collection = _sanitize(name)

        print(f"Fetching playlist info: {url}")
        result = subprocess.run(
            [yt_dlp, "--flat-playlist", "--dump-single-json", "--no-warnings", url],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp error: {result.stderr[:300]}")

        data = json.loads(result.stdout)
        entries = data.get("entries", [data])
        print(f"Found {len(entries)} video(s)")

        total_chunks = 0

        for i, entry in enumerate(entries, 1):
            vid_url = entry.get("url") or entry.get("webpage_url") or url
            vid_id = entry.get("id", "")
            title = entry.get("title") or f"episode_{i}"

            if vid_id and "bilibili" in url and not vid_url.startswith("http"):
                vid_url = f"https://www.bilibili.com/video/{vid_id}"

            if on_progress:
                on_progress(IngestionProgress(
                    stage="downloading",
                    progress=i / len(entries),
                    current_item=title,
                    total_items=len(entries),
                    completed_items=i - 1,
                ))

            print(f"\n[{i}/{len(entries)}] {title}")

            with tempfile.TemporaryDirectory() as tmp:
                segments = None

                sub_result = subprocess.run(
                    [yt_dlp, "--no-playlist", "--skip-download",
                     "--write-auto-subs", "--write-subs",
                     "--sub-langs", language or "en",
                     "--sub-format", "srt/best",
                     "--convert-subs", "srt",
                     "-o", os.path.join(tmp, "subs"),
                     "--no-warnings", vid_url],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                )

                srt_files = list(Path(tmp).glob("*.srt"))
                if srt_files:
                    print(f"  Using YouTube subtitles")
                    segments = self.transcriber.load_srt(srt_files[0])

                if not segments:
                    audio_path = os.path.join(tmp, "audio.m4a")
                    ffmpeg_path = _find_ffmpeg()

                    dl_cmd = [
                        yt_dlp, "-x",
                        "--audio-format", "m4a",
                        "--audio-quality", "5",
                        "--no-playlist",
                        "--no-warnings",
                        "-o", audio_path,
                        vid_url,
                    ]
                    if ffmpeg_path:
                        dl_cmd.extend(["--ffmpeg-location", ffmpeg_path])

                    print(f"  Downloading audio...")
                    dl_result = subprocess.run(
                        dl_cmd, capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                    )
                    if dl_result.returncode != 0:
                        print(f"  x Download failed: {dl_result.stderr[:200]}")
                        continue

                    audio_files = list(Path(tmp).glob("audio.*"))
                    if not audio_files:
                        print(f"  x No audio file found after download")
                        continue

                    print(f"  Transcribing...")
                    if on_progress:
                        on_progress(IngestionProgress(
                            stage="transcribing",
                            progress=i / len(entries),
                            current_item=title,
                            total_items=len(entries),
                            completed_items=i - 1,
                        ))
                    segments = self.transcriber.transcribe(audio_files[0], language=language)

                if not segments:
                    print(f"  x No segments produced, skipping")
                    continue

                n = self._chunk_and_store_segments(
                    segments=segments,
                    collection=collection,
                    display_name=name,
                    topic=topic,
                    subtopic=subtopic,
                    episode_num=i,
                    episode_title=title,
                    url=vid_url,
                    source_type="video",
                    contextual=contextual,
                )
                total_chunks += n
                print(f"  -> {n} chunks stored")

        if on_progress:
            on_progress(IngestionProgress(
                stage="done", progress=1.0, current_item="",
                total_items=len(entries), completed_items=len(entries),
                message=f"{total_chunks} chunks from {len(entries)} videos",
            ))

        return total_chunks

    # ── File / URL / Documents (unified) ─────────────────────────────

    def ingest_file(
        self,
        path: str,
        name: str,
        topic: str = "",
        subtopic: str = "",
        source_type: str | None = None,
        enrich: bool = True,
        on_progress: ProgressCallback = None,
    ) -> int:
        """Ingest any supported file type. Auto-detects format."""
        from .extractors import extract, detect_source_type

        if source_type is None:
            source_type = detect_source_type(path)

        doc = extract(path, source_type)
        return self._ingest_extracted(
            doc=doc,
            name=name,
            topic=topic,
            subtopic=subtopic,
            source_path=path,
            enrich=enrich,
            on_progress=on_progress,
        )

    def ingest_url(
        self,
        url: str,
        name: str,
        topic: str = "",
        subtopic: str = "",
        enrich: bool = True,
        on_progress: ProgressCallback = None,
    ) -> int:
        """Ingest a web page URL."""
        from .extractors import extract_url

        doc = extract_url(url)
        return self._ingest_extracted(
            doc=doc,
            name=name,
            topic=topic,
            subtopic=subtopic,
            source_path=url,
            url_override=url,
            enrich=enrich,
            on_progress=on_progress,
        )

    def ingest_documents(
        self,
        path: str | Path,
        name: str,
        topic: str,
        subtopic: str,
        contextual: bool = False,
        on_progress: ProgressCallback = None,
    ) -> int:
        """Ingest documents from a file or folder. Routes through ingest_file."""
        path = Path(path)
        from .extractors import extract, detect_source_type

        if path.is_file():
            return self.ingest_file(
                path=str(path), name=name, topic=topic, subtopic=subtopic,
                enrich=True, on_progress=on_progress,
            )

        if not path.is_dir():
            raise FileNotFoundError(f"Path not found: {path}")

        DOC_EXTS = {".md", ".markdown", ".rst", ".txt", ".pdf", ".epub"}
        files = sorted([f for f in path.rglob("*") if f.suffix.lower() in DOC_EXTS])
        if not files:
            raise ValueError(f"No document files found in {path}")

        total_chunks = 0
        for i, doc_file in enumerate(files, 1):
            if on_progress:
                on_progress(IngestionProgress(
                    stage="extracting", progress=i / len(files),
                    current_item=doc_file.name,
                    total_items=len(files), completed_items=i - 1,
                ))
            try:
                n = self.ingest_file(
                    path=str(doc_file), name=name, topic=topic, subtopic=subtopic,
                    enrich=True,
                )
                total_chunks += n
                print(f"[{i}/{len(files)}] {doc_file.name} -> {n} chunks")
            except Exception as e:
                print(f"[{i}/{len(files)}] {doc_file.name} x {e}")

        if on_progress:
            on_progress(IngestionProgress(
                stage="done", progress=1.0, current_item="",
                total_items=len(files), completed_items=len(files),
                message=f"{total_chunks} chunks from {len(files)} documents",
            ))

        return total_chunks

    # ── Single SRT ────────────────────────────────────────────────────

    def ingest_srt(
        self,
        srt_path: str | Path,
        name: str,
        topic: str,
        subtopic: str,
        episode_num: int = 1,
        episode_title: str | None = None,
        url: str = "",
        contextual: bool = False,
    ) -> int:
        """Ingest a single SRT file."""
        srt_path = Path(srt_path)
        if not srt_path.exists():
            raise FileNotFoundError(f"SRT not found: {srt_path}")

        segments = self.transcriber.load_srt(srt_path)
        collection = _sanitize(name)

        return self._chunk_and_store_segments(
            segments=segments,
            collection=collection,
            display_name=name,
            topic=topic,
            subtopic=subtopic,
            episode_num=episode_num,
            episode_title=episode_title or srt_path.stem,
            url=url or str(srt_path),
            source_type="video",
            contextual=contextual,
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _ingest_extracted(
        self,
        doc,
        name: str,
        topic: str,
        subtopic: str,
        source_path: str,
        url_override: str | None = None,
        enrich: bool = True,
        on_progress: ProgressCallback = None,
    ) -> int:
        """Shared pipeline for file and URL ingestion: chunk -> enrich -> store."""
        from .enrich import enrich_programmatic, enrich_llm
        from ..providers.registry import get_registry

        collection = _sanitize(name)
        existing = [c for c in self.store.list_collections() if c["collection"] == collection]
        if existing:
            print(f"  Collection '{name}' already exists, skipping.")
            return 0

        item_name = Path(source_path).name if not source_path.startswith("http") else source_path

        if on_progress:
            on_progress(IngestionProgress(
                stage="chunking", progress=0.3, current_item=item_name,
                total_items=1, completed_items=0,
                message=f"Chunking {len(doc.sections)} sections...",
            ))

        chunks = chunk_sections(doc.sections, target_tokens=512, source_path=source_path)
        if not chunks:
            return 0

        if enrich:
            if on_progress:
                on_progress(IngestionProgress(
                    stage="enriching", progress=0.5, current_item=item_name,
                    total_items=1, completed_items=0,
                    message="Extracting keywords and entities...",
                ))
            chunks = enrich_programmatic(chunks)

            provider = get_registry().active
            if provider:
                if on_progress:
                    on_progress(IngestionProgress(
                        stage="enriching", progress=0.6, current_item=item_name,
                        total_items=1, completed_items=0,
                        message="LLM enrichment (titles, summaries, tags)...",
                    ))
                chunks = enrich_llm(chunks, provider)

        if on_progress:
            on_progress(IngestionProgress(
                stage="embedding", progress=0.7, current_item=item_name,
                total_items=1, completed_items=0,
                message=f"Embedding {len(chunks)} chunks...",
            ))

        meta = {
            "collection": collection,
            "collection_display": name,
            "topic": topic.lower() if topic else "",
            "subtopic": subtopic.lower() if subtopic else "",
            "episode_num": 1,
            "episode_title": doc.metadata.get("book_title", doc.metadata.get("page_title", Path(source_path).stem)),
            "url": url_override or doc.metadata.get("url", ""),
            "source_type": doc.source_type,
            "file_path": source_path,
        }

        n = self.store.add_chunks(chunks, meta)

        if on_progress:
            on_progress(IngestionProgress(
                stage="done", progress=1.0, current_item=item_name,
                total_items=1, completed_items=1,
                message=f"Done — {n} chunks indexed",
            ))

        return n

    def _chunk_and_store_segments(
        self,
        segments: list[dict],
        collection: str,
        display_name: str,
        topic: str,
        subtopic: str,
        episode_num: int,
        episode_title: str,
        url: str,
        source_type: str,
        contextual: bool = False,
    ) -> int:
        """Chunk temporal segments and store. For video/audio sources."""
        chunks = chunk_segments(segments)

        if contextual:
            chunks = self._apply_contextual_prefixes(chunks, display_name, episode_title)

        meta = {
            "collection": collection,
            "collection_display": display_name,
            "topic": topic.lower(),
            "subtopic": subtopic.lower(),
            "episode_num": episode_num,
            "episode_title": episode_title,
            "url": url,
            "source_type": source_type,
        }

        return self.store.add_chunks(chunks, meta)

    def _apply_contextual_prefixes(
        self,
        chunks: list[dict],
        collection_name: str,
        episode_title: str,
    ) -> list[dict]:
        """Prepend LLM-generated context to each chunk before embedding."""
        from ..providers.registry import get_registry

        registry = get_registry()
        provider = registry.active
        if provider is None or not provider.detect():
            print("  Contextual prefixes: no provider available, skipping")
            return chunks

        print(f"  Generating contextual prefixes for {len(chunks)} chunks...")
        enriched = []

        for i, chunk in enumerate(chunks):
            prompt = (
                f"Tutorial: {collection_name}\n"
                f"Episode: {episode_title}\n\n"
                "Write 1-2 sentences describing the specific concept or technique "
                "covered in this transcript excerpt. Be concrete - mention tool names, "
                "node names, or technical terms if present.\n\n"
                f"Excerpt: {chunk['text']}\n\nContext description:"
            )

            try:
                context = provider.chat(
                    [{"role": "user", "content": prompt}],
                )
                enriched.append({
                    **chunk,
                    "text": f"{context.strip()}\n\n{chunk['text']}",
                })
            except Exception as e:
                print(f"  Context gen failed for chunk {i}: {e}")
                enriched.append(chunk)

        return enriched
