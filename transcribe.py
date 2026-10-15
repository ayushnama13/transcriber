#!/usr/bin/env python3
import argparse
import json
import os
import re
from array import array
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from moonshine_voice import ModelArch, Transcriber, get_model_for_language


def get_audio_duration(file_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(res.stdout.strip())
    except Exception as e:
        print(f"Error getting audio duration via ffprobe: {e}", file=sys.stderr)
        sys.exit(1)


def convert_to_wav(input_path, output_wav_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(output_wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: ffmpeg conversion failed:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not on PATH.", file=sys.stderr)
        sys.exit(1)


def find_silences(wav_path):
    cmd = [
        "ffmpeg",
        "-i",
        str(wav_path),
        "-af",
        "silencedetect=noise=-35dB:d=0.4",
        "-f",
        "null",
        "-",
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stderr = res.stderr
    except subprocess.CalledProcessError as e:
        stderr = e.stderr

    silences = []
    current_start = None
    for line in stderr.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([\d.]+)", line)
            if m:
                current_start = float(m.group(1))
        elif "silence_end:" in line and current_start is not None:
            m = re.search(r"silence_end:\s*([\d.]+)", line)
            if m:
                end = float(m.group(1))
                silences.append((current_start, end))
                current_start = None
    return silences


def compute_cut_points(duration, chunk_min, silences):
    chunk_sec = chunk_min * 60
    target_times = []
    t = chunk_sec
    while t < duration:
        target_times.append(t)
        t += chunk_sec

    cut_points = [0.0]
    for target in target_times:
        best_cut = target
        best_diff = 20.0 + 1e-9
        for start, end in silences:
            mid = (start + end) / 2.0
            diff = abs(mid - target)
            if diff <= 20.0 and diff < best_diff:
                best_diff = diff
                best_cut = mid
        cut_points.append(best_cut)
    cut_points.append(duration)

    # Merge final chunk if shorter than 30s
    if len(cut_points) > 2 and (cut_points[-1] - cut_points[-2] < 30.0):
        cut_points.pop(-2)

    return cut_points


def seconds_to_mmss(sec):
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m:02d}:{s:02d}"


def seconds_to_hms(sec):
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def seconds_to_srt_time(sec):
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_and_link_segments(segments):
    cleaned = []
    exceptions = {"I", "I'm", "I'll", "I've", "I'd"}
    for i, seg in enumerate(segments):
        text = " ".join(seg["text"].split())
        text = re.sub(r"([,;:])\1+", r"\1", text)
        if i > 0:
            prev_text = cleaned[-1]["text"]
            # Moonshine capitalises every line; undo it mid-sentence.
            if prev_text.endswith((",", ";")):
                words = text.split()
                if words:
                    first_word = words[0]
                    if first_word[0].isupper() and first_word not in exceptions:
                        text = text[0].lower() + text[1:]
        cleaned.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "chunk": seg["chunk"],
            }
        )
    # Moonshine lines can overlap slightly; keep subtitle cues sequential.
    for prev, nxt in zip(cleaned, cleaned[1:]):
        prev["end"] = max(prev["start"], min(prev["end"], nxt["start"]))
    return cleaned


def build_paragraphs(segments):
    if not segments:
        return []

    paragraphs = []
    curr_para = [segments[0]]
    curr_words = len(segments[0]["text"].split())

    for i in range(1, len(segments)):
        seg = segments[i]
        prev_seg = segments[i - 1]
        gap = seg["start"] - prev_seg["end"]
        prev_ends_term = prev_seg["text"].endswith((".", "?", "!"))

        cond1 = (gap > 2.0) and prev_ends_term
        cond2 = (curr_words > 120) and prev_ends_term

        if cond1 or cond2:
            paragraphs.append(curr_para)
            curr_para = [seg]
            curr_words = len(seg["text"].split())
        else:
            curr_para.append(seg)
            curr_words += len(seg["text"].split())

    if curr_para:
        paragraphs.append(curr_para)

    return paragraphs


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe long audio files with Moonshine in chunks."
    )
    parser.add_argument("input_audio", help="Path to input audio file")
    parser.add_argument(
        "-o", "--outdir", help="Output directory (default: ./out/<stem>/)"
    )
    parser.add_argument(
        "--chunk-min",
        type=int,
        default=5,
        help="Target chunk duration in minutes (default: 5)",
    )
    parser.add_argument(
        "--model",
        choices=["medium", "base", "tiny"],
        default="medium",
        help="Model size (default: medium, most accurate)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_audio)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    stem = input_path.stem
    out_dir = Path(args.outdir) if args.outdir else Path("out") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_wav = Path(tmpdir) / "audio_16k.wav"
        print("Converting audio to 16 kHz mono PCM WAV...", file=sys.stderr)
        convert_to_wav(input_path, temp_wav)

        duration = get_audio_duration(temp_wav)
        print(f"Audio duration: {duration:.2f}s", file=sys.stderr)

        print("Detecting silence intervals...", file=sys.stderr)
        silences = find_silences(temp_wav)

        cut_points = compute_cut_points(duration, args.chunk_min, silences)
        print(
            f"Cut points ({len(cut_points) - 1} chunks): {cut_points}",
            file=sys.stderr,
        )

        print("Loading WAV audio samples...", file=sys.stderr)
        with wave.open(str(temp_wav), "rb") as w:
            audio_int16 = array("h")
            audio_int16.frombytes(w.readframes(w.getnframes()))
        if sys.byteorder == "big":
            audio_int16.byteswap()

        chunks_data = []
        for i in range(len(cut_points) - 1):
            start_t = cut_points[i]
            end_t = cut_points[i + 1]
            start_idx = int(start_t * 16000)
            end_idx = int(end_t * 16000)
            chunk_samples = audio_int16[start_idx:end_idx]
            chunks_data.append((start_t, end_t, chunk_samples))

        arch_map = {
            "medium": ModelArch.MEDIUM_STREAMING,
            "base": ModelArch.BASE,
            "tiny": ModelArch.TINY,
        }
        model_arch_enum = arch_map[args.model]

        print(f"Loading Moonshine model ({args.model})...", file=sys.stderr)
        model_path, model_arch_val = get_model_for_language("en", model_arch_enum)
        transcriber = Transcriber(model_path=model_path, model_arch=model_arch_val)

        all_raw_segments = []
        N = len(chunks_data)
        for i, (start_t, end_t, chunk_samples) in enumerate(chunks_data, 1):
            t0 = time.monotonic()
            floats = [s / 32768.0 for s in chunk_samples]
            tr = transcriber.transcribe_without_streaming(floats, 16000)
            elapsed = time.monotonic() - t0

            start_mmss = seconds_to_mmss(start_t)
            end_mmss = seconds_to_mmss(end_t)
            print(
                f"chunk {i}/{N} [{start_mmss}-{end_mmss}] done in {elapsed:.1f}s",
                file=sys.stderr,
            )

            for line in tr.lines:
                text = (line.text or "").strip()
                if not text:
                    continue
                abs_start = start_t + line.start_time
                abs_end = abs_start + line.duration
                all_raw_segments.append(
                    {
                        "start": abs_start,
                        "end": abs_end,
                        "text": text,
                        "chunk": i,
                    }
                )

        transcriber.close()

    print("Post-processing and merging transcript...", file=sys.stderr)
    segments = clean_and_link_segments(all_raw_segments)
    paragraphs = build_paragraphs(segments)

    txt_path = out_dir / f"{stem}.txt"
    srt_path = out_dir / f"{stem}.srt"
    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"

    # Write .txt
    txt_content = []
    for para in paragraphs:
        para_text = " ".join(seg["text"] for seg in para)
        txt_content.append(para_text)
    txt_path.write_text("\n\n".join(txt_content) + "\n", encoding="utf-8")

    # Write .srt
    srt_lines = []
    for idx, seg in enumerate(segments, 1):
        start_str = seconds_to_srt_time(seg["start"])
        end_str = seconds_to_srt_time(seg["end"])
        srt_lines.append(
            f"{idx}\n{start_str} --> {end_str}\n{seg['text']}\n"
        )
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    # Write .md
    md_content = []
    for para in paragraphs:
        first_start = para[0]["start"]
        timestamp = seconds_to_hms(first_start)
        para_text = " ".join(seg["text"] for seg in para)
        md_content.append(f"**[{timestamp}]** {para_text}")
    md_path.write_text("\n\n".join(md_content) + "\n", encoding="utf-8")

    # Write .json
    json_data = [
        {
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"],
            "chunk": seg["chunk"],
        }
        for seg in segments
    ]
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    print(f"Output files written to {out_dir}:")
    print(f"  - {txt_path}")
    print(f"  - {srt_path}")
    print(f"  - {md_path}")
    print(f"  - {json_path}")


if __name__ == "__main__":
    main()
