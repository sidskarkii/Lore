"""Transcription — faster-whisper wrapper with per-sentence timestamps.

Produces segments with start/end timestamps suitable for SRT generation
and timestamp-linked search results. Handles both audio and video files
(extracts audio automatically via ffmpeg/av).

Usage:
    from lore.core.transcribe import Transcriber

    t = Transcriber()
    segments = t.transcribe("video.mp4")
    # [{"start": 0.0, "end": 5.28, "text": "Hello everyone..."}, ...]

    t.save_srt(segments, "output.srt")
    t.save_txt(segments, "output.txt")
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import get_config


def _fmt_ts(secs: float) -> str:
    """Format seconds as MM:SS."""
    return f"{int(secs // 60):02}:{int(secs % 60):02}"


def _srt_time(t: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


class Transcriber:
    """Transcribe audio/video files using faster-whisper.

    Lazy-loads the model on first use. Model size, device, and compute type
    are read from config.yaml.
    """

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        cfg = get_config()
        model_size = cfg.get("transcription.model", "small.en")
        device = cfg.get("transcription.device", "cpu")
        compute_type = cfg.get("transcription.compute_type", "int8")

        # Auto-detect: use CPU with int8 if no CUDA
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
                    compute_type = "int8"
            except ImportError:
                device = "cpu"
                compute_type = "int8"

        print(f"  Loading Whisper {model_size} on {device} ({compute_type})...")
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> list[dict]:
        """Transcribe an audio or video file.

        Args:
            audio_path: Path to audio/video file (mp4, mp3, wav, m4a, etc.)
            language: Language code (e.g. "en"). None = auto-detect.
            word_timestamps: If True, include word-level timestamps in each segment.

        Returns:
            List of segments: [{"start": float, "end": float, "text": str, "words": [...]}, ...]
        """
        cfg = get_config()
        if language is None:
            language = cfg.get("transcription.language")

        model = self._get_model()

        kwargs = {"word_timestamps": word_timestamps}
        if language:
            kwargs["language"] = language

        segments_iter, info = model.transcribe(str(audio_path), **kwargs)

        print(f"  Detected language: {info.language} ({info.language_probability:.0%})")

        segments = []
        for seg in segments_iter:
            entry = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
            if word_timestamps and seg.words:
                entry["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in seg.words
                ]
            segments.append(entry)

        print(f"  Transcribed: {len(segments)} segments, {_fmt_ts(info.duration)} duration")
        return segments

    @staticmethod
    def save_srt(segments: list[dict], path: str | Path):
        """Save segments as an SRT subtitle file."""
        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n")
                f.write(f"{seg['text']}\n\n")

    @staticmethod
    def save_txt(segments: list[dict], path: str | Path):
        """Save segments as a timestamped text file."""
        with open(path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{_fmt_ts(seg['start'])} - {_fmt_ts(seg['end'])}] {seg['text']}\n")

    @staticmethod
    def load_srt(path: str | Path) -> list[dict]:
        """Load an existing SRT file as segments.

        Use this to skip transcription when subtitles already exist.
        """

        segments = []
        content = Path(path).read_text(encoding="utf-8")

        # Parse SRT format
        blocks = re.split(r"\n\n+", content.strip())
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            # Parse timestamp line: 00:00:00,000 --> 00:00:05,280
            ts_match = re.match(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})",
                lines[1],
            )
            if not ts_match:
                continue

            g = ts_match.groups()
            start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
            end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
            text = " ".join(lines[2:]).strip()

            segments.append({"start": start, "end": end, "text": text})

        return segments

    @staticmethod
    def load_txt(path: str | Path) -> list[dict]:
        """Load a timestamped text file as segments.

        Expected format: [MM:SS - MM:SS] text
        """

        segments = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            match = re.match(r"\[(\d+):(\d+)\s*-\s*(\d+):(\d+)\]\s*(.+)", line)
            if match:
                g = match.groups()
                start = int(g[0]) * 60 + int(g[1])
                end = int(g[2]) * 60 + int(g[3])
                segments.append({"start": float(start), "end": float(end), "text": g[4].strip()})

        return segments
