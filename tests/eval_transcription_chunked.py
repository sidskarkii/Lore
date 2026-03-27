"""Long-form ASR benchmark with audio chunking.

The ONNX models (Distil-Whisper, Parakeet) failed on long audio because
they don't chunk internally. This test splits audio into 30s segments,
transcribes each, then concatenates results.

Tests on 14.5 min Blender tutorial (Skeletal Structure episode).
"""

import time
import os
import sys
import numpy as np
import wave

AUDIO = os.path.join(os.path.dirname(__file__), "test_audio_long.wav")
CHUNK_SEC = 30
OVERLAP_SEC = 2  # Small overlap to avoid cutting mid-word


def load_wav(path):
    """Load WAV file as float32 numpy array + sample rate."""
    wf = wave.open(path, "rb")
    assert wf.getsampwidth() == 2 and wf.getnchannels() == 1
    sr = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    wf.close()
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def chunk_audio(samples, sr, chunk_sec=30, overlap_sec=2):
    """Split audio into overlapping chunks. Returns list of (samples, offset_sec)."""
    chunk_len = int(chunk_sec * sr)
    overlap_len = int(overlap_sec * sr)
    step = chunk_len - overlap_len
    chunks = []
    pos = 0
    while pos < len(samples):
        end = min(pos + chunk_len, len(samples))
        chunks.append((samples[pos:end], pos / sr))
        pos += step
    return chunks


def save_temp_wav(samples, sr, path):
    """Save samples to a temp WAV file."""
    wf = wave.open(path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes((samples * 32768).astype(np.int16).tobytes())
    wf.close()


# ── Distil-Whisper (onnx-asr) with chunking ─────────────────────────────

def test_distil_whisper_chunked():
    from onnx_asr import load_model
    import tempfile

    print("\n  Loading Distil-Whisper small.en...")
    model = load_model("distil-whisper/distil-small.en")

    samples, sr = load_wav(AUDIO)
    chunks = chunk_audio(samples, sr, CHUNK_SEC, OVERLAP_SEC)
    print(f"  Audio: {len(samples)/sr:.0f}s -> {len(chunks)} chunks of {CHUNK_SEC}s")

    all_text = []
    t0 = time.time()
    for i, (chunk_samples, offset) in enumerate(chunks):
        tmp_path = os.path.join(tempfile.gettempdir(), f"tv_chunk_{i}.wav")
        save_temp_wav(chunk_samples, sr, tmp_path)
        result = model.recognize(tmp_path)
        text = result.text if hasattr(result, "text") else str(result)
        all_text.append(text.strip())
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass
    elapsed = time.time() - t0

    transcript = " ".join(all_text)
    return transcript, elapsed


# ── Parakeet TDT (onnx-asr) with chunking ───────────────────────────────

def test_parakeet_chunked():
    from onnx_asr import load_model

    print("\n  Loading Parakeet TDT 0.6B...")
    model = load_model("istupakov/parakeet-tdt-0.6b-v3-onnx")

    samples, sr = load_wav(AUDIO)
    chunks = chunk_audio(samples, sr, CHUNK_SEC, OVERLAP_SEC)
    print(f"  Audio: {len(samples)/sr:.0f}s -> {len(chunks)} chunks of {CHUNK_SEC}s")

    all_text = []
    t0 = time.time()
    for i, (chunk_samples, offset) in enumerate(chunks):
        tmp_path = os.path.join(tempfile.gettempdir(), f"tv_chunk_{i}.wav")
        save_temp_wav(chunk_samples, sr, tmp_path)
        try:
            result = model.recognize(tmp_path)
            text = result.text if hasattr(result, "text") else str(result)
            all_text.append(text.strip())
        except Exception as e:
            print(f"    Chunk {i} failed: {e}")
            all_text.append("")
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass
    elapsed = time.time() - t0

    transcript = " ".join(t for t in all_text if t)
    return transcript, elapsed


# ── SenseVoice (sherpa-onnx) with chunking ───────────────────────────────

def test_sensevoice_chunked():
    import sherpa_onnx
    from pathlib import Path

    print("\n  Loading SenseVoice Small...")
    model_dir = Path.home() / ".cache" / "sherpa-onnx" / "sensevoice"
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_dir / "model.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        use_itn=True,
        num_threads=4,
    )

    samples, sr = load_wav(AUDIO)
    chunks = chunk_audio(samples, sr, CHUNK_SEC, OVERLAP_SEC)
    print(f"  Audio: {len(samples)/sr:.0f}s -> {len(chunks)} chunks of {CHUNK_SEC}s")

    all_text = []
    t0 = time.time()
    for i, (chunk_samples, offset) in enumerate(chunks):
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, chunk_samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        all_text.append(text)
    elapsed = time.time() - t0

    transcript = " ".join(t for t in all_text if t)
    return transcript, elapsed


# ── faster-whisper baseline (no chunking needed) ────────────────────────

def test_faster_whisper():
    from faster_whisper import WhisperModel

    print("\n  Loading faster-whisper small.en...")
    model = WhisperModel("small.en", device="cpu", compute_type="int8")

    t0 = time.time()
    segs, info = model.transcribe(AUDIO, language="en")
    segments = list(segs)
    elapsed = time.time() - t0

    transcript = " ".join(s.text.strip() for s in segments)
    return transcript, elapsed


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(AUDIO):
        print(f"ERROR: {AUDIO} not found")
        sys.exit(1)

    samples, sr = load_wav(AUDIO)
    duration = len(samples) / sr
    print("=" * 60)
    print(f"  Long-Form ASR Benchmark (chunked) — {duration/60:.1f} min")
    print("=" * 60)

    results = []

    tests = [
        ("Distil-Whisper small.en (chunked)", test_distil_whisper_chunked, 150),
        ("SenseVoice Small (chunked)", test_sensevoice_chunked, 200),
        ("Parakeet TDT 0.6B (chunked)", test_parakeet_chunked, 600),
        ("faster-whisper small.en (native)", test_faster_whisper, 150),
    ]

    for name, fn, size_mb in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            transcript, elapsed = fn()
            words = len(transcript.split())
            rtf = elapsed / duration
            print(f"\n  Time: {elapsed:.1f}s | RTF: {rtf:.3f} | Words: {words}")
            print(f"  First 250 chars: {transcript[:250]}...")
            results.append((name, size_mb, elapsed, words, rtf))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {duration/60:.1f}min audio, CPU, 30s chunks")
    print(f"{'='*70}")
    print(f"  {'Model':<40} {'Size':>6} {'Time':>8} {'RTF':>7} {'Words':>7}")
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*7} {'-'*7}")
    for name, size, t, words, rtf in sorted(results, key=lambda x: x[2]):
        print(f"  {name:<40} {size:>5}M {t:>7.1f}s {rtf:>7.3f} {words:>7}")
