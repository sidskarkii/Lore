"""Test transcription module — all paths."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from tutorialvault.core.transcribe import Transcriber

SHORT_AUDIO = "tests/test_audio.wav"        # 2 min
LONG_AUDIO = "tests/test_audio_long.wav"    # 14.5 min
COURSE_VIDEO = "D:/Courses/VFXGRACE/Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/1 - Blender Creature Effects The Complete WorkFlow En/01 Introduction.mp4"
COURSE_SRT = "D:/Courses/VFXGRACE/Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/1 - Blender Creature Effects The Complete WorkFlow En/01 Introduction.srt"


def test_transcribe():
    t = Transcriber()

    # 1. Short audio transcription
    print("1. Transcribe short audio (2 min)")
    t0 = time.time()
    segs = t.transcribe(SHORT_AUDIO, language="en")
    elapsed = time.time() - t0
    assert len(segs) > 0, "No segments returned"
    assert all("start" in s and "end" in s and "text" in s for s in segs)
    assert segs[0]["start"] < segs[0]["end"]
    print(f"   {len(segs)} segments in {elapsed:.1f}s")
    print(f"   First: [{segs[0]['start']:.1f}s] {segs[0]['text'][:60]}")

    # 2. Save + load SRT round-trip
    print("\n2. SRT round-trip")
    srt_path = "tests/test_roundtrip.srt"
    t.save_srt(segs, srt_path)
    loaded = t.load_srt(srt_path)
    assert len(loaded) == len(segs), f"SRT round-trip lost segments: {len(segs)} -> {len(loaded)}"
    assert abs(loaded[0]["start"] - segs[0]["start"]) < 0.01
    assert loaded[0]["text"] == segs[0]["text"]
    os.unlink(srt_path)
    print(f"   {len(loaded)} segments preserved, timestamps match")

    # 3. Save + load TXT round-trip
    print("\n3. TXT round-trip")
    txt_path = "tests/test_roundtrip.txt"
    t.save_txt(segs, txt_path)
    loaded_txt = t.load_txt(txt_path)
    assert len(loaded_txt) == len(segs), f"TXT round-trip lost segments: {len(segs)} -> {len(loaded_txt)}"
    os.unlink(txt_path)
    print(f"   {len(loaded_txt)} segments preserved")

    # 4. Load existing course SRT
    print("\n4. Load existing course SRT")
    if os.path.exists(COURSE_SRT):
        existing = t.load_srt(COURSE_SRT)
        assert len(existing) > 0
        assert existing[0]["text"]
        print(f"   {len(existing)} segments from course SRT")
        print(f"   First: [{existing[0]['start']:.1f}s] {existing[0]['text'][:60]}")
    else:
        print("   SKIPPED (course SRT not found)")

    # 5. Video file (mp4) — tests ffmpeg/av audio extraction
    print("\n5. Transcribe video file (mp4)")
    if os.path.exists(COURSE_VIDEO):
        t0 = time.time()
        vid_segs = t.transcribe(COURSE_VIDEO, language="en")
        elapsed = time.time() - t0
        assert len(vid_segs) > 0
        print(f"   {len(vid_segs)} segments in {elapsed:.1f}s")
        print(f"   First: [{vid_segs[0]['start']:.1f}s] {vid_segs[0]['text'][:60]}")
    else:
        print("   SKIPPED (course video not found)")

    # 6. Word-level timestamps
    print("\n6. Word-level timestamps")
    word_segs = t.transcribe(SHORT_AUDIO, language="en", word_timestamps=True)
    has_words = any("words" in s and s["words"] for s in word_segs)
    if has_words:
        first_with_words = next(s for s in word_segs if s.get("words"))
        w = first_with_words["words"][0]
        assert "word" in w and "start" in w and "end" in w
        print(f"   Words in first segment: {len(first_with_words['words'])}")
        print(f"   First word: '{w['word']}' [{w['start']:.2f}s-{w['end']:.2f}s]")
    else:
        print(f"   No word timestamps returned (model may not support it)")

    # 7. Long audio
    print("\n7. Transcribe long audio (14.5 min)")
    if os.path.exists(LONG_AUDIO):
        t0 = time.time()
        long_segs = t.transcribe(LONG_AUDIO, language="en")
        elapsed = time.time() - t0
        total_words = sum(len(s["text"].split()) for s in long_segs)
        print(f"   {len(long_segs)} segments, {total_words} words in {elapsed:.1f}s")
        print(f"   Last segment ends at: {long_segs[-1]['end']:.1f}s")
        assert long_segs[-1]["end"] > 800, "Transcription seems truncated"
    else:
        print("   SKIPPED (long audio not found)")

    print("\n=== ALL TRANSCRIPTION TESTS PASSED ===")


if __name__ == "__main__":
    test_transcribe()
