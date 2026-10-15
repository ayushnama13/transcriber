"""Moonshine ASR per diarized turn, reading each turn from the best recorder.

Channel policy (DESIGN.md "Overlap policy"):
- clean turn: the recorder where this turn is loudest (turn["best_channel"]).
- overlapped turn: the recorder where this *speaker's* clean speech is
  loudest overall, so each voice in a crosstalk region is read from the mic
  nearest to that person rather than whichever mic the loudest talker hit.
"""
import os

import numpy as np

from slc.audio import read_segment

SR = 16000
MAX_PIECE_S = 240.0  # Moonshine is run on <= 4 min pieces, as in transcribe.py


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
    if keyterms:
        tr.set_keyterms([k.replace(",", " ").strip() for k in keyterms if k.strip()])
    elif context:
        tr.set_context(context)
    return tr


def _pieces(start, end):
    n = max(1, int(np.ceil((end - start) / MAX_PIECE_S)))
    edges = np.linspace(start, end, n + 1)
    return list(zip(edges[:-1], edges[1:]))


def transcribe_audio(transcriber, audio):
    if len(audio) < SR * 0.2:
        return ""
    res = transcriber.transcribe_without_streaming(audio.astype(np.float32).tolist(), SR)
    return " ".join((ln.text or "").strip() for ln in res.lines if (ln.text or "").strip())


def transcribe_turns(transcriber, turns, aligned_dir, pad=0.15, progress=None):
    """Returns [{"text", "channel"}] aligned with turns."""
    spk = speaker_channels(turns)
    out = []
    for i, t in enumerate(turns):
        ch = pick_channel(t, spk)
        wav = os.path.join(aligned_dir, f"rec{ch}.wav")
        texts = []
        for a, b in _pieces(t["start"], t["end"]):
            audio = read_segment(wav, max(0.0, a - pad), b + pad)
            texts.append(transcribe_audio(transcriber, audio))
        out.append({"text": " ".join(x for x in texts if x), "channel": ch})
        if progress:
            progress(i + 1, len(turns))
    return out
