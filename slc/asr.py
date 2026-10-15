"""Moonshine ASR per diarized turn, reading each turn from the best recorder.

Channel policy (DESIGN.md "Overlap policy"):
- clean turn: the recorder where this turn is loudest (turn["best_channel"]).
- overlapped turn: the recorder where this *speaker's* clean speech is
  loudest overall, so each voice in a crosstalk region is read from the mic
  nearest to that person rather than whichever mic the loudest talker hit.
"""
import os

import numpy as np

from slc.audio import duration_s, read_segment

SR = 16000
MAX_PIECE_S = 240.0  # Moonshine is run on <= 4 min pieces, as in transcribe.py
_FLUSH = [0.0] * SR


def speaker_channels(turns):
    """cluster -> recorder index with the highest median level over clean turns."""
    levels = {}
    for t in turns:
        if t.get("overlap") or not t.get("channel_rms_db"):
            continue
        levels.setdefault(t["cluster"], []).append(t["channel_rms_db"])
    out = {}
    for c, rows in levels.items():
        out[c] = int(np.argmax(np.median(np.asarray(rows, np.float32), axis=0)))
    return out


def pick_channel(turn, spk_channels):
    if turn.get("overlap") and turn["cluster"] in spk_channels:
        return spk_channels[turn["cluster"]]
    return int(turn.get("best_channel", 0))


def load_transcriber(model="medium", keyterms=None, context=None):
    from moonshine_voice import ModelArch, Transcriber, get_model_for_language

    arch = {"medium": ModelArch.MEDIUM_STREAMING, "base": ModelArch.BASE, "tiny": ModelArch.TINY}[model]
    path, arch = get_model_for_language("en", arch)
    tr = Transcriber(model_path=path, model_arch=arch)
    if keyterms and model != "medium":
        # Moonshine only supports key terms on the medium streaming model.
        print(f"note: key terms ignored for the {model} model (medium only)", flush=True)
    elif keyterms:
        tr.set_keyterms([k.replace(",", " ").strip() for k in keyterms if k.strip()])
    elif context:
        tr.set_context(context)
    return tr


def _pieces(start, end):
    n = max(1, int(np.ceil((end - start) / MAX_PIECE_S)))
    edges = np.linspace(start, end, n + 1)
    return list(zip(edges[:-1], edges[1:]))


def transcribe_lines(transcriber, audio):
    """[(start_s, end_s, text)] relative to the start of audio."""
    if len(audio) < SR * 0.2:
        return []
    res = transcriber.transcribe_without_streaming(audio.astype(np.float32).tolist(), SR)
    # Moonshine carries decoder state across calls and the next clip then often
    # starts with a phantom "Yeah"; decoding a second of silence resets it.
    transcriber.transcribe_without_streaming(_FLUSH, SR)
    out = []
    for ln in res.lines:
        text = (ln.text or "").strip()
        if text:
            out.append((float(ln.start_time), float(ln.start_time + ln.duration), text))
    return out


def transcribe_audio(transcriber, audio):
    return " ".join(t for _, _, t in transcribe_lines(transcriber, audio))


def transcribe_turns(transcriber, turns, aligned_dir, pad=0.15, progress=None):
    """Returns [{"text", "channel", "lines": [{"start", "end", "text"}]}] aligned with turns.

    Line times are absolute (meeting timeline), clamped to the turn.
    """
    spk = speaker_channels(turns)
    out = []
    for i, t in enumerate(turns):
        ch = pick_channel(t, spk)
        wav = os.path.join(aligned_dir, f"rec{ch}.wav")
        lines = []
        for a, b in _pieces(t["start"], t["end"]):
            a0 = max(0.0, a - pad)
            for s, e, text in transcribe_lines(transcriber, read_segment(wav, a0, b + pad)):
                # A line lying wholly in the pad is a neighbour's speech or a
                # decoder artifact (Moonshine leaks a "Yeah" from the previous
                # call into the leading pad), not this turn.
                if a0 + e <= t["start"] + 0.02 or a0 + s >= t["end"] - 0.02:
                    continue
                lines.append({"start": round(float(min(max(a0 + s, t["start"]), t["end"])), 3),
                              "end": round(float(min(max(a0 + e, t["start"]), t["end"])), 3), "text": text})
        out.append({"text": " ".join(ln["text"] for ln in lines), "channel": ch, "lines": lines})
        if progress:
            progress(i + 1, len(turns))
    return out


# ---- Whisper (faster-whisper, GPU) ----------------------------------------
# Whole-recording ASR: the speech is transcribed in long VAD chunks (up to 30 s,
# split at pauses) so the model hears full sentences, then each word is given to
# the diarized turn it falls in. One stretch of audio therefore yields one line,
# never the same words under two speakers.

WHISPER_MODELS = {"turbo": "large-v3-turbo"}
BLOCK_S = 600.0  # read the recording 10 min at a time, cut at a pause
_HALLUCINATIONS = {"thank you", "thanks for watching", "thank you for watching", "you",
                   "subtitles by the amara org community", "please subscribe", "bye"}


def _preload_cuda_libs():
    """CTranslate2 dlopens libcublas.so.12/libcudnn*.so.9 by name; the pip nvidia-* wheels
    (installed for torch) aren't on the loader path, so load them globally first."""
    import ctypes
    import glob
    import site

    for sp in site.getsitepackages():
        for pat in ("nvidia/cublas/lib/libcublasLt.so.*", "nvidia/cublas/lib/libcublas.so.*",
                    "nvidia/cudnn/lib/libcudnn*.so.*"):
            for lib in sorted(glob.glob(os.path.join(sp, pat))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def load_whisper(name="turbo", gpu=True):
    _preload_cuda_libs()
    import ctranslate2
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device = "cuda" if gpu and ctranslate2.get_cuda_device_count() > 0 else "cpu"
    model = WhisperModel(WHISPER_MODELS.get(name, name), device=device, compute_type="int8")
    return BatchedInferencePipeline(model), device


def hotword_string(terms, max_chars=500):
    """Comma list of names/terms for Whisper's per-window hint (kept short: it eats context)."""
    out, n = [], 0
    for t in terms or []:
        if n + len(t) + 2 > max_chars:
            break
        out.append(t)
        n += len(t) + 2
    return ", ".join(out) or None


def _cut_point(wav, approx, search_s=20.0, frame_s=0.25):
    """Quietest frame in the search_s before approx, so a block never splits a word."""
    a = max(0.0, approx - search_s)
    x = read_segment(wav, a, approx)
    n = int(frame_s * SR)
    if len(x) < 2 * n:
        return approx
    k = len(x) // n
    rms = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1))
    i = int(np.argmin(rms))
    return a + (i + 0.5) * frame_s


def _norm(text):
    return " ".join("".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").split())


def is_junk_segment(text, avg_logprob, no_speech_prob):
    """Whisper's typical inventions in murmur/silence."""
    if avg_logprob < -1.2:
        return True
    if no_speech_prob > 0.6 and avg_logprob < -0.7:
        return True
    return _norm(text) in _HALLUCINATIONS and (no_speech_prob > 0.2 or avg_logprob < -0.5)


def collapse_repeats(words, max_rep=2, max_n=6):
    """Cut loops like 'Hush Hush Hush ...' or a phrase said 4x: keep at most max_rep copies."""
    toks = [_norm(w["word"]) for w in words]
    keep = [True] * len(words)
    i = 0
    while i < len(words):
        jumped = False
        for n in range(1, max_n + 1):
            unit = toks[i:i + n]
            if len(unit) < n or not any(unit):
                break
            reps = 1
            while toks[i + reps * n:i + (reps + 1) * n] == unit:
                reps += 1
            if reps > max_rep:
                for j in range(i + max_rep * n, i + reps * n):
                    keep[j] = False
                i += reps * n
                jumped = True
                break
        if not jumped:
            i += 1
    return [w for w, k in zip(words, keep) if k]


def transcribe_words(pipe, wav, hotwords=None, batch_size=8, block_s=BLOCK_S, progress=None):
    """[{"start", "end", "word", "prob"}] for the whole recording, absolute times."""
    dur = duration_s(wav)
    words, a = [], 0.0
    while a < dur - 0.1:
        b = dur if dur - a <= block_s * 1.2 else _cut_point(wav, a + block_s)
        audio = read_segment(wav, a, b)
        segs, _ = pipe.transcribe(audio, language="en", batch_size=batch_size, word_timestamps=True,
                                  hotwords=hotwords, condition_on_previous_text=False)
        for s in segs:
            if is_junk_segment(s.text, s.avg_logprob, s.no_speech_prob):
                continue
            for w in s.words or []:
                if w.word.strip():
                    words.append({"start": round(a + w.start, 3), "end": round(a + w.end, 3),
                                  "word": w.word.strip(), "prob": round(float(w.probability), 3)})
        a = b
        if progress:
            progress(a, dur)
    return collapse_repeats(words)


def assign_words(words, turns, tol=0.25):
    """turn index for each word: the turn containing its midpoint; while the previous
    word's turn still covers it, stay with that speaker (crosstalk goes to whoever was
    already talking); else the longest covering turn; gaps go to the nearest turn."""
    if not turns:
        return []
    starts = np.array([t["start"] for t in turns])
    ends = np.array([t["end"] for t in turns])
    out, prev = [], None
    for w in words:
        m = 0.5 * (w["start"] + w["end"])
        cover = np.nonzero((starts - tol <= m) & (m <= ends + tol))[0]
        if len(cover):
            if prev is not None and prev in cover:
                k = prev
            else:
                k = int(cover[np.argmax(ends[cover] - starts[cover])])
        else:
            dist = np.maximum(starts - m, m - ends)
            k = int(np.argmin(dist))
        out.append(k)
        prev = k
    return out


def join_words(words):
    """Whisper splits "6,500" -> "6" ",500" and "full-fledged" -> "full" "-fledged"; glue those."""
    out = ""
    for w in words:
        if out and not (w[:1] in ",.?!%')-" or out[-1:] == "-"):
            out += " "
        out += w
    return out


def words_to_lines(words, gap_s=1.0, max_line_s=15.0):
    lines, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > gap_s or
                    (w["end"] - cur[0]["start"] > max_line_s and cur[-1]["word"][-1:] in ".?!")):
            lines.append(cur)
            cur = []
        cur.append(w)
    lines.append(cur) if cur else None
    return [{"start": ln[0]["start"], "end": ln[-1]["end"], "text": join_words(w["word"] for w in ln)}
            for ln in lines]


def whisper_turns(words, turns, channel=0):
    """asr.json rows (same shape as transcribe_turns) from whole-recording words."""
    owner = assign_words(words, turns)
    per = [[] for _ in turns]
    for w, k in zip(words, owner):
        per[k].append(w)
    out = []
    for ws in per:
        lines = words_to_lines(ws)
        out.append({"text": " ".join(ln["text"] for ln in lines), "channel": channel, "lines": lines})
    return out


def whisper_meeting(pipe, turns, aligned_dir, cache_dir, hotwords=None, log=print):
    """asr.json rows for all turns. Each recorder a turn reads from (pick_channel) is
    levelled and transcribed whole; its words go only to the turns that read it, so in
    crosstalk each voice comes from its own nearest mic. Per-channel words are cached."""
    import json

    from slc.audio import level_wav

    spk = speaker_channels(turns)
    chans = [pick_channel(t, spk) for t in turns]
    out = [None] * len(turns)
    for ch in sorted(set(chans)):
        cache = os.path.join(cache_dir, f"words_rec{ch}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                words = json.load(f)
        else:
            src = os.path.join(aligned_dir, f"rec{ch}.wav")
            lev = os.path.join(cache_dir, f"asr_rec{ch}.wav")
            log(f"  rec{ch}: levelling loudness")
            level_wav(src, lev)
            words = transcribe_words(pipe, lev, hotwords=hotwords,
                                     progress=lambda d, n: log(f"  rec{ch}: ASR {d / 60:.0f}/{n / 60:.0f} min"))
            with open(cache + ".tmp", "w") as f:
                json.dump(words, f)
            os.replace(cache + ".tmp", cache)
            os.remove(lev)
        idx = [i for i, c in enumerate(chans) if c == ch]
        rows = whisper_turns(words, [turns[i] for i in idx], channel=ch)
        for i, r in zip(idx, rows):
            out[i] = r
    return out
