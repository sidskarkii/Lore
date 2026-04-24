"""Transcription — sherpa-onnx offline STT with per-segment timestamps.

Uses ONNX Runtime (shared with embedding model) for efficient inference.
Supports Whisper and Moonshine models. CoreML acceleration on Apple Silicon.

Usage:
    from lore.core.transcribe import Transcriber

    t = Transcriber()
    segments = t.transcribe("video.mp4")
    # [{"start": 0.0, "end": 5.28, "text": "Hello everyone..."}, ...]
"""

from __future__ import annotations

import re
import subprocess
import tempfile
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


def _extract_audio(input_path: str | Path, output_path: str | Path) -> bool:
    """Extract audio from video file using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(input_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", str(output_path), "-y"],
            capture_output=True, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


_HF_BASE = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-{model}/resolve/main"

_MODELS = {
    "whisper-tiny": "tiny.en",
    "whisper-medium": "medium.en",
}

_MODEL_FILES = ["encoder.onnx", "decoder.onnx", "tokens.txt"]


class Transcriber:
    """Transcribe audio/video files using sherpa-onnx.

    Lazy-loads the model on first use. Uses ONNX Runtime with CoreML
    on Apple Silicon. Default model: Whisper tiny.en (~75MB).
    """

    def __init__(self):
        self._recognizer = None
        self._model_dir = None

    def _ensure_model(self) -> Path:
        """Download model files from HuggingFace if not present."""
        cfg = get_config()
        model_name = cfg.get("transcription.model", "whisper-medium")
        model_variant = _MODELS.get(model_name, "medium.en")
        model_dir = cfg.data_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        base_url = _HF_BASE.format(model=model_variant)
        for fname in _MODEL_FILES:
            full_name = f"{model_variant}-{fname}"
            target = model_dir / full_name
            if not target.exists() or target.stat().st_size < 100:
                url = f"{base_url}/{full_name}"
                print(f"  Downloading {full_name}...")
                subprocess.run(
                    ["curl", "-sfL", url, "-o", str(target)],
                    check=True,
                )
                if target.stat().st_size < 1000:
                    target.unlink()
                    raise RuntimeError(f"Download too small, likely failed: {url}")

        self._model_dir = model_dir
        return model_dir

    def _get_recognizer(self):
        if self._recognizer is not None:
            return self._recognizer

        import sherpa_onnx

        model_dir = self._ensure_model()
        cfg = get_config()
        model_name = cfg.get("transcription.model", "whisper-tiny")

        encoder = list(model_dir.glob("*encoder*"))[0]
        decoder = list(model_dir.glob("*decoder*"))[0]
        tokens = list(model_dir.glob("*tokens*"))[0]

        cfg = get_config()
        lang = cfg.get("transcription.language", "en")
        print(f"  Loading {model_name} via sherpa-onnx (lang={lang})...")

        if "moonshine" in model_name:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
                preprocessor=str(model_dir / "preprocess.onnx"),
                encoder=str(encoder),
                uncached_decoder=str(model_dir / "uncached_decoder.onnx"),
                cached_decoder=str(model_dir / "cached_decoder.onnx"),
                tokens=str(tokens),
                num_threads=4,
            )
        else:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(encoder),
                decoder=str(decoder),
                tokens=str(tokens),
                num_threads=4,
                language=lang,
                task="transcribe",
            )

        return self._recognizer

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """Transcribe an audio or video file.

        Extracts audio to WAV if needed, then runs sherpa-onnx offline recognition.
        Returns segments with start/end timestamps (~30s windows).
        """
        audio_path = Path(audio_path)

        if audio_path.suffix.lower() != ".wav":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            if not _extract_audio(audio_path, wav_path):
                raise RuntimeError(f"Failed to extract audio from {audio_path}")
            audio_path = Path(wav_path)

        recognizer = self._get_recognizer()

        import wave
        import numpy as np
        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

            if sample_width == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            if n_channels > 1:
                samples = samples[::n_channels]
        duration = len(samples) / sample_rate

        window_secs = 30
        window_samples = int(window_secs * sample_rate)
        segments = []

        offset = 0
        while offset < len(samples):
            chunk = samples[offset:offset + window_samples]
            s = recognizer.create_stream()
            s.accept_waveform(sample_rate, chunk)
            recognizer.decode_stream(s)

            text = s.result.text.strip()
            if text:
                start_sec = offset / sample_rate
                end_sec = min(start_sec + len(chunk) / sample_rate, duration)
                segments.append({
                    "start": start_sec,
                    "end": end_sec,
                    "text": text,
                })
            offset += window_samples

        if not segments and duration > 0:
            s = recognizer.create_stream()
            s.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(s)
            text = s.result.text.strip()
            if text:
                segments = [{"start": 0.0, "end": duration, "text": text}]

        print(f"  Transcribed: {len(segments)} segments, {_fmt_ts(len(samples)/sample_rate)} duration")
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
        """Load an existing SRT file as segments."""
        segments = []
        content = Path(path).read_text(encoding="utf-8")

        blocks = re.split(r"\n\n+", content.strip())
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

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
        """Load a timestamped text file as segments."""
        segments = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            match = re.match(r"\[(\d+):(\d+)\s*-\s*(\d+):(\d+)\]\s*(.+)", line)
            if match:
                g = match.groups()
                start = int(g[0]) * 60 + int(g[1])
                end = int(g[2]) * 60 + int(g[3])
                segments.append({"start": float(start), "end": float(end), "text": g[4].strip()})

        return segments
