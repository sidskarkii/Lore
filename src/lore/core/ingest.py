"""Ingestion pipeline — orchestrates source → transcribe → chunk → embed → store.

Supports:
  - Local video/audio files (mp4, mp3, wav, m4a, etc.)
  - YouTube URLs and playlists (via yt-dlp)
  - Local documents (markdown, RST, PDF, plain text)
  - Pre-existing SRT/transcript files

Usage:
    from lore.core.ingest import Ingester

    ingester = Ingester()

    # Video folder
    ingester.ingest_folder("D:/Courses/Blender101/", name="Blender 101", topic="3d", subtopic="blender")

    # YouTube playlist
    ingester.ingest_youtube("https://youtube.com/playlist?list=...", name="Houdini VFX", topic="3d", subtopic="houdini")

    # Documents
    ingester.ingest_documents("./docs/", name="Blender Manual", topic="3d", subtopic="blender")

    # Single SRT
    ingester.ingest_srt("transcript.srt", name="Tutorial", topic="3d", subtopic="blender", episode_num=1)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .chunk import chunk_segments
from .config import get_config
from .store import Store
from .transcribe import Transcriber


# ── Progress callback type ────────────────────────────────────────────────

@dataclass
class IngestionProgress:
    """Progress update for the UI."""
    stage: str          # "downloading", "transcribing", "chunking", "embedding", "done", "error"
    progress: float     # 0.0 to 1.0
    current_item: str   # e.g. "Episode 3: Mesh Tools"
    total_items: int
    completed_items: int
    message: str = ""
    error: str | None = None


ProgressCallback = Callable[[IngestionProgress], None] | None


# ── Helpers ───────────────────────────────────────────────────────────────

def _sanitize(name: str, maxlen: int = 80) -> str:
    """Sanitize a string for use as an ID."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:maxlen]


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
DOC_EXTS = {".md", ".markdown", ".rst", ".txt", ".pdf"}
SUB_EXTS = {".srt", ".vtt"}


def _find_yt_dlp() -> str:
    """Find yt-dlp binary."""
    import shutil
    path = shutil.which("yt-dlp")
    if path:
        return path
    # Check common locations
    for candidate in [
        r"H:\Houdini\whisperx-env2\Scripts\yt-dlp.exe",
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("yt-dlp not found. Install with: pip install yt-dlp")


def _find_ffmpeg() -> str | None:
    """Find ffmpeg binary (optional, for audio extraction)."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    candidate = r"C:\Users\siddhant\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
    if os.path.isdir(candidate):
        return candidate
    return None


# ── Ingester ──────────────────────────────────────────────────────────────

class Ingester:
    """Orchestrates the full ingestion pipeline."""

    def __init__(self, store: Store | None = None):
        self.store = store or Store()
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
        """Ingest all video/audio files from a local folder.

        Files are sorted alphabetically and numbered as episodes.
        If an SRT file exists next to a video, it's used instead of transcribing.

        Returns total chunks stored.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        # Find media files
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

            # Use existing SRT if available, otherwise transcribe
            if srt_file.exists():
                print(f"[{i}/{len(files)}] Loading SRT: {episode_title}")
                segments = self.transcriber.load_srt(srt_file)
            else:
                print(f"[{i}/{len(files)}] Transcribing: {episode_title}")
                segments = self.transcriber.transcribe(media_file, language=language)
                # Save SRT next to the file for future runs
                self.transcriber.save_srt(segments, srt_file)

            # Chunk + store
            n = self._chunk_and_store(
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
        """Ingest from a YouTube URL or playlist.

        Downloads auto-subs if available (skips Whisper), falls back to
        audio download + transcription if no subs.

        Returns total chunks stored.
        """
        yt_dlp = _find_yt_dlp()
        collection = _sanitize(name)

        # Get playlist info
        print(f"Fetching playlist info: {url}")
        result = subprocess.run(
            [yt_dlp, "--flat-playlist", "--dump-single-json", "--no-warnings", url],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp error: {result.stderr[:300]}")

        data = json.loads(result.stdout)
        entries = data.get("entries", [data])  # single video has no entries key
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

                # Try to download auto-subs first (skip Whisper if available)
                srt_path = os.path.join(tmp, "subs.srt")
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

                # Find downloaded srt
                srt_files = list(Path(tmp).glob("*.srt"))
                if srt_files:
                    print(f"  Using YouTube subtitles")
                    segments = self.transcriber.load_srt(srt_files[0])

                if not segments:
                    # Download audio and transcribe
                    audio_path = os.path.join(tmp, "audio.m4a")
                    ffmpeg_dir = _find_ffmpeg()

                    dl_cmd = [
                        yt_dlp, "-x",
                        "--audio-format", "m4a",
                        "--audio-quality", "5",
                        "--no-playlist",
                        "--no-warnings",
                        "-o", audio_path,
                        vid_url,
                    ]
                    if ffmpeg_dir:
                        dl_cmd.extend(["--ffmpeg-location", ffmpeg_dir])

                    print(f"  Downloading audio...")
                    dl_result = subprocess.run(
                        dl_cmd, capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                    )
                    if dl_result.returncode != 0:
                        print(f"  x Download failed: {dl_result.stderr[:200]}")
                        continue

                    # Find the actual downloaded file (yt-dlp may add extensions)
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

                n = self._chunk_and_store(
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

    # ── Documents ─────────────────────────────────────────────────────

    def ingest_documents(
        self,
        path: str | Path,
        name: str,
        topic: str,
        subtopic: str,
        contextual: bool = False,
        on_progress: ProgressCallback = None,
    ) -> int:
        """Ingest documents from a file or folder.

        Supports: .md, .rst, .txt, .pdf
        Returns total chunks stored.
        """
        path = Path(path)
        collection = _sanitize(name)

        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted([f for f in path.rglob("*") if f.suffix.lower() in DOC_EXTS])
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        if not files:
            raise ValueError(f"No document files found in {path}")

        total_chunks = 0

        for i, doc_file in enumerate(files, 1):
            if on_progress:
                on_progress(IngestionProgress(
                    stage="chunking",
                    progress=i / len(files),
                    current_item=doc_file.name,
                    total_items=len(files),
                    completed_items=i - 1,
                ))

            print(f"[{i}/{len(files)}] {doc_file.name}")

            # Parse document to text
            text = self._parse_document(doc_file)
            if not text.strip():
                print(f"  x Empty document, skipping")
                continue

            # Create segments from text (no timestamps for docs)
            segments = self._text_to_segments(text)

            n = self._chunk_and_store(
                segments=segments,
                collection=collection,
                display_name=name,
                topic=topic,
                subtopic=subtopic,
                episode_num=i,
                episode_title=doc_file.stem,
                url=str(doc_file),
                source_type="docs",
                contextual=contextual,
            )
            total_chunks += n
            print(f"  -> {n} chunks stored")

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
        """Ingest a single SRT file. Returns chunks stored."""
        srt_path = Path(srt_path)
        if not srt_path.exists():
            raise FileNotFoundError(f"SRT not found: {srt_path}")

        segments = self.transcriber.load_srt(srt_path)
        collection = _sanitize(name)

        return self._chunk_and_store(
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

    def _chunk_and_store(
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
        """Chunk segments and store in LanceDB. Returns chunks stored."""
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

    def _parse_document(self, path: Path) -> str:
        """Parse a document file to plain text."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            try:
                import pymupdf4llm
                return pymupdf4llm.to_markdown(str(path))
            except ImportError:
                # Fallback: try pypdf
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(path))
                    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
                except ImportError:
                    raise ImportError("Install pymupdf4llm or pypdf for PDF support: pip install pymupdf4llm")

        # Plain text formats
        return path.read_text(encoding="utf-8", errors="replace")

    def _text_to_segments(self, text: str) -> list[dict]:
        """Convert document text to fake segments for the chunker.

        Documents don't have real timestamps, so we assign sequential
        positions (1 second per ~10 words) for compatibility with the
        chunk/store pipeline.
        """
        # Split by paragraphs or double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        segments = []
        pos = 0.0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            word_count = len(para.split())
            duration = max(1.0, word_count / 2.5)  # ~2.5 words per second (reading speed)
            segments.append({
                "start": pos,
                "end": pos + duration,
                "text": para,
            })
            pos += duration

        return segments

    def _apply_contextual_prefixes(
        self,
        chunks: list[dict],
        collection_name: str,
        episode_title: str,
    ) -> list[dict]:
        """Prepend LLM-generated context to each chunk before embedding.

        Uses the active provider to generate a 2-3 sentence description
        of what each chunk covers. This improves retrieval accuracy by
        35-67% according to Anthropic's contextual retrieval research.
        """
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
                f"Excerpt: {chunk['text'][:600]}\n\nContext description:"
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
