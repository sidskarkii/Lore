"""Benchmark ASR models on real tutorial audio (2min Blender tutorial clip).

Models tested:
1. Parakeet TDT 0.6B (onnx-asr) — NVIDIA, best benchmarks, ONNX
2. SenseVoice Small (sherpa-onnx) — Alibaba, 15x faster than Whisper
3. faster-whisper large-v2 (int8) — the baseline (old TutorialVault)
4. faster-whisper small.en (int8) — smaller variant
5. Distil-Whisper small.en (onnx-asr) — lightest Whisper-quality
"""

import time
import os
import sys
import numpy as np

TEST_AUDIO = os.path.join(os.path.dirname(__file__), "test_audio.wav")

GROUND_TRUTH = (
    "Hello everyone today I'm really excited to start our tutorials on modeling "
    "before we dive into that let's first discuss the difficulties we usually encounter during "
    "this process so that we have a clear direction in our minds and then solve them one by one "
    "through this course first the lack of basic knowledge "
    "I believe it's challenging to learn modeling in the beginning facing so many types of brushes"
).lower()


def word_overlap(transcript: str, reference: str) -> float:
    trans_words = set(transcript.lower().split())
    ref_words = set(reference.lower().split())
    if not ref_words:
        return 0.0
    return len(trans_words & ref_words) / len(ref_words) * 100


def print_result(name, transcript, n_segments, elapsed, model_size_mb=None):
    accuracy = word_overlap(transcript, GROUND_TRUTH)
    rtf = elapsed / 120.0  # 2 min audio

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Time: {elapsed:.1f}s | RTF: {rtf:.3f} | Segments: {n_segments}")
    if model_size_mb:
        print(f"  Model: ~{model_size_mb}MB")
    print(f"  Word overlap: {accuracy:.0f}%")
    print(f"  Transcript: {transcript[:250]}...")

    return {"name": name, "time": elapsed, "rtf": rtf, "accuracy": accuracy,
            "segments": n_segments, "model_size_mb": model_size_mb}


# ── 1. Parakeet TDT 0.6B ────────────────────────────────────────────────

def test_parakeet():
    from onnx_asr import load_model

    print("\n--- Loading Parakeet TDT 0.6B ---")
    t0 = time.time()
    model = load_model("istupakov/parakeet-tdt-0.6b-v3-onnx")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    result = model.recognize(TEST_AUDIO)
    elapsed = time.time() - t0

    transcript = result.text if hasattr(result, 'text') else str(result)
    n_segs = len(result.segments) if hasattr(result, 'segments') else 1
    return print_result("Parakeet TDT 0.6B (onnx-asr)", transcript, n_segs, elapsed, 600)


# ── 2. SenseVoice Small ─────────────────────────────────────────────────

def test_sensevoice():
    import sherpa_onnx
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    print("\n--- Loading SenseVoice Small ---")
    model_dir = Path.home() / ".cache" / "sherpa-onnx" / "sensevoice"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / "model.int8.onnx"
    tokens_file = model_dir / "tokens.txt"

    if not model_file.exists():
        print("  Downloading model...")
        repo = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
        hf_hub_download(repo, "model.int8.onnx", local_dir=str(model_dir))
        hf_hub_download(repo, "tokens.txt", local_dir=str(model_dir))

    t0 = time.time()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_file),
        tokens=str(tokens_file),
        use_itn=True,
        num_threads=4,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    import wave
    wf = wave.open(TEST_AUDIO, "rb")
    sample_rate = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    wf.close()
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    t0 = time.time()
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    elapsed = time.time() - t0

    transcript = stream.result.text.strip()
    return print_result("SenseVoice Small (sherpa-onnx)", transcript, 1, elapsed, 200)


# ── 3 & 4. faster-whisper ───────────────────────────────────────────────

def test_faster_whisper(model_size, label, size_mb):
    from faster_whisper import WhisperModel

    print(f"\n--- Loading {label} ---")
    t0 = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    segs, info = model.transcribe(TEST_AUDIO, language="en", word_timestamps=False)
    segments = list(segs)
    elapsed = time.time() - t0

    transcript = " ".join(s.text.strip() for s in segments)
    return print_result(label, transcript, len(segments), elapsed, size_mb)


# ── 5. Distil-Whisper ────────────────────────────────────────────────────

def test_distil_whisper():
    from onnx_asr import load_model

    print("\n--- Loading Distil-Whisper small.en ---")
    t0 = time.time()
    try:
        model = load_model("distil-whisper/distil-small.en")
    except Exception:
        try:
            model = load_model("distil-small.en")
        except Exception as e:
            # List what whisper models are available
            print(f"  Could not load distil-whisper: {e}")
            print("  Trying whisper small.en via onnx-asr instead...")
            model = load_model("whisper-small.en")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    result = model.recognize(TEST_AUDIO)
    elapsed = time.time() - t0

    transcript = result.text if hasattr(result, 'text') else str(result)
    n_segs = len(result.segments) if hasattr(result, 'segments') else 1
    return print_result("Whisper small.en (onnx-asr)", transcript, n_segs, elapsed, 150)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(TEST_AUDIO):
        print(f"ERROR: {TEST_AUDIO} not found")
        sys.exit(1)

    print("=" * 60)
    print("  ASR Benchmark — 2min Blender Tutorial")
    print("=" * 60)

    results = []
    tests = [
        ("Parakeet TDT 0.6B", test_parakeet),
        ("SenseVoice Small", test_sensevoice),
        ("faster-whisper large-v2", lambda: test_faster_whisper("large-v2", "faster-whisper large-v2 (int8)", 1500)),
        ("faster-whisper small.en", lambda: test_faster_whisper("small.en", "faster-whisper small.en (int8)", 150)),
        ("Whisper small.en ONNX", test_distil_whisper),
    ]

    for name, fn in tests:
        try:
            results.append(fn())
        except Exception as e:
            print(f"\n  {name}: FAILED — {e}")
            import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY — 2min audio, CPU only")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'Size':>7} {'Time':>7} {'RTF':>7} {'Accuracy':>9}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
    for r in sorted(results, key=lambda x: x["time"]):
        size = f"{r['model_size_mb']}M" if r['model_size_mb'] else "?"
        print(f"  {r['name']:<35} {size:>7} {r['time']:>6.1f}s {r['rtf']:>7.3f} {r['accuracy']:>8.0f}%")
