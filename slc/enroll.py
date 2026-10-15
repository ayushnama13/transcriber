"""Fill the voice library: from sample clips, notes-matched segments, or reviews."""
import json
import os
import tempfile

import numpy as np

from slc.audio import duration_s, read_segment, to_wav16k

SR = 16000


def speech_pieces(wav, piece_s=8.0, min_s=3.0):
    """Cut the speech parts of a clip into ~piece_s chunks (simple energy VAD)."""
    dur = duration_s(wav)
    x = read_segment(wav, 0, dur)
    hop = int(0.03 * SR)
    n = len(x) // hop
    if n == 0:
        return []
    e = 10 * np.log10((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10)
    thr = max(np.percentile(e, 90) - 25, np.percentile(e, 10) + 6)
    speech = x[:n * hop].reshape(n, hop)[e > thr].reshape(-1)
    k = int(piece_s * SR)
    pieces = [speech[i:i + k] for i in range(0, len(speech), k)]
    return [p for p in pieces if len(p) >= min_s * SR]


def enroll_files(lib, name, files, role="member", embedder=None):
    from slc.diarize import Embedder
    embedder = embedder or Embedder()
    vecs = []
    with tempfile.TemporaryDirectory() as d:
        for i, src in enumerate(files):
            wav = os.path.join(d, f"{i}.wav")
            to_wav16k(src, wav)
            vecs += [embedder(p) for p in speech_pieces(wav)]
    if not vecs:
        raise ValueError(f"no usable speech (>= 3 s) found for {name}")
    lib.add(name, np.stack(vecs), role=role, source="enroll")
    return len(vecs)


def enroll_from_calibration(lib, workdir, embedder=None):
    """Embed the notes-confirmed clips listed in calibration.json["harvest"]."""
    from slc.diarize import Embedder
    with open(os.path.join(workdir, "calibration.json")) as f:
        harvest = json.load(f).get("harvest", [])
    with open(os.path.join(workdir, "transcript.json")) as f:
        by_id = {s["id"]: s for s in json.load(f)["segments"]}
    embedder = embedder or Embedder()
    per = {}
    for h in harvest:
        ch = by_id.get(h["segment_id"], {}).get("channel", 0)
        audio = read_segment(os.path.join(workdir, "aligned", f"rec{ch}.wav"), h["start"], h["end"])
        per.setdefault(h["name"], []).append(embedder(audio))
    for name, vecs in per.items():
        role = lib.people.get(name, {}).get("role", "invitee")
        lib.add(name, np.stack(vecs), role=role, source="notes", max_per_person=200)
    return {k: len(v) for k, v in per.items()}


def learn_from_transcript(lib, workdir):
    """Reviewed transcript lines -> voice library (see voicelib.learn_from_review)."""
    from slc.voicelib import learn_from_review
    with open(os.path.join(workdir, "turns.json")) as f:
        turns = json.load(f)
    emb = np.load(os.path.join(workdir, "embeddings.npy"))
    with open(os.path.join(workdir, "transcript.json")) as f:
        segs = json.load(f)["segments"]
    final = {}
    for s in segs:
        if not s.get("reviewed"):
            continue
        # A turn counts only when every reviewed line of it agrees on the name.
        prev = final.get(s["turn_id"])
        final[s["turn_id"]] = s["speaker"] if prev in (None, s["speaker"]) else ""
    before = {n: lib.people[n].get("n", 0) for n in lib.names()}
    learn_from_review(lib, turns, emb, final)
    return {n: lib.people[n]["n"] - before.get(n, 0) for n in lib.names()
            if lib.people[n]["n"] != before.get(n, 0)}
