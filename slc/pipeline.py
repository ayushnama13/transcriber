"""One meeting end to end: align -> diarize -> name -> ASR -> transcript -> export.

Every stage caches its output in the work dir and is skipped when that output
exists, so a crashed 6 h run resumes where it stopped and re-naming speakers
(after enrolling more voices) does not redo ASR.
"""
import json
import os
import time

import numpy as np

from slc import voicelib


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load(path):
    with open(path) as f:
        return json.load(f)


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def prepare_alignment(recordings, workdir, reference=0):
    from slc.align import align_recordings
    from slc.audio import to_wav16k

    os.makedirs(os.path.join(workdir, "norm"), exist_ok=True)
    norm = []
    for i, src in enumerate(recordings):
        dst = os.path.join(workdir, "norm", f"rec{i}.wav")
        if not os.path.exists(dst):
            _log(f"converting {os.path.basename(src)} -> 16 kHz")
            to_wav16k(src, dst)
        norm.append(dst)
    _save(os.path.join(workdir, "inputs.json"),
          {"recordings": [os.path.abspath(p) for p in recordings], "reference": reference})
    return align_recordings(norm, workdir, reference=reference)


def keyterms(lib=None, terms_file=None, limit=150):
    """Names of enrolled people + campus terms, for Moonshine's decoder bias."""
    terms = []
    if terms_file:
        with open(terms_file, encoding="utf-8") as f:
            terms += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if lib is not None:
        for name in lib.names():
            terms.append(name)
            terms += [w for w in name.split() if len(w) > 2]
    seen, out = set(), []
    for t in terms:
        t = t.replace(",", " ").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:limit]


def build_segments(turns, asr, speakers, conf, max_seg_s=30.0, part_s=1800.0, low_conf=0.5):
    segs = []
    turn_segs = {}
    for t, a, c in zip(turns, asr, conf):
        spk = speakers[t["cluster"]]
        # Split long turns at Moonshine line boundaries for reviewable clips.
        groups, cur = [], []
        for ln in a.get("lines") or []:
            if cur and ln["end"] - cur[0]["start"] > max_seg_s:
                groups.append(cur)
                cur = []
            cur.append(ln)
        if cur:
            groups.append(cur)
        if not groups:
            continue  # no speech recognised in this turn
        for gi, g in enumerate(groups):
            start = t["start"] if gi == 0 else g[0]["start"]
            end = t["end"] if gi == len(groups) - 1 else groups[gi + 1][0]["start"]
            flags = []
            if t.get("overlap"):
                flags.append("crosstalk")
            if spk["source"] != "voicelib" or c < low_conf:
                flags.append("low_speaker_conf")
            seg = {"id": len(segs), "turn_id": t["id"], "start": round(start, 3), "end": round(end, 3),
                   "speaker": spk["name"], "speaker_confidence": round(float(c), 3),
                   "cluster": t["cluster"], "text": " ".join(ln["text"] for ln in g),
                   "channel": a["channel"], "flags": flags, "overlaps_with": [],
                   "reviewed": False, "part": int(start // part_s)}
            segs.append(seg)
            turn_segs.setdefault(t["id"], []).append(seg)
    # Turn-level overlaps -> segment ids that actually intersect in time.
    by_turn = {t["id"]: t for t in turns}
    for s in segs:
        for ot in by_turn[s["turn_id"]].get("overlaps_with", []):
            for o in turn_segs.get(ot, []):
                if min(s["end"], o["end"]) > max(s["start"], o["start"]):
                    s["overlaps_with"].append(o["id"])
    return segs


def name_speakers(workdir, lib):
    turns = _load(os.path.join(workdir, "turns.json"))
    emb = np.load(os.path.join(workdir, "embeddings.npy"))
    speakers = voicelib.name_clusters(lib, turns, emb)
    conf = voicelib.turn_confidence(lib, turns, emb, speakers)
    _save(os.path.join(workdir, "speakers.json"), speakers)
    return turns, speakers, conf


def run_meeting(recordings, workdir, library=None, terms_file=None, model="medium",
                part_min=30.0, gpu=True, reference=0, force=False, meeting=None):
    os.makedirs(workdir, exist_ok=True)
    t0 = time.time()
    p = lambda name: os.path.join(workdir, name)

    tpath = p("transcript.json")
    if os.path.exists(tpath) and not force:
        old = _load(tpath)
        if any(s.get("reviewed") for s in old["segments"]):
            raise RuntimeError(f"{tpath} has reviewed lines; refusing to overwrite (use --force)")

    if not os.path.exists(p("alignment.json")):
        if not recordings:
            raise RuntimeError("no recordings given and no alignment.json in work dir")
        _log(f"aligning {len(recordings)} recorders")
        al = prepare_alignment(recordings, workdir, reference)
        for r in al["recorders"]:
            _log(f"  rec{r['index']}: offset {r['offset_s']:+.3f}s drift {r['drift_ppm']:+.1f}ppm "
                 f"quality {r['quality']:.2f} windows {r['n_windows_used']}")

    if not os.path.exists(p("turns.json")):
        from slc.diarize import diarize_meeting
        diarize_meeting(workdir, gpu=gpu, log=_log)

    lib = voicelib.VoiceLibrary(library) if library else None
    turns, speakers, conf = name_speakers(workdir, lib or voicelib.VoiceLibrary(p("empty_lib")))
    named = sum(1 for s in speakers.values() if s["source"] == "voicelib")
    _log(f"speakers: {len(speakers)} clusters, {named} matched to the voice library")

    if not os.path.exists(p("asr.json")):
        from slc.asr import load_transcriber, transcribe_turns
        terms = keyterms(lib, terms_file)
        _log(f"loading Moonshine {model} ({len(terms)} keyterms)")
        tr = load_transcriber(model, keyterms=terms)
        t_asr = time.time()

        def prog(i, n):
            if i % 25 == 0 or i == n:
                _log(f"  ASR {i}/{n} turns ({time.time() - t_asr:.0f}s)")
        asr = transcribe_turns(tr, turns, os.path.join(workdir, "aligned"), progress=prog)
        _save(p("asr.json"), asr)
    asr = _load(p("asr.json"))

    segs = build_segments(turns, asr, speakers, conf, part_s=part_min * 60)
    transcript = {"meeting": meeting or os.path.basename(os.path.abspath(workdir)), "segments": segs}
    _save(tpath, transcript)

    from slc.export import export_all
    paths = export_all(transcript, p("out"))
    n_x = sum("crosstalk" in s["flags"] for s in segs)
    n_low = sum("low_speaker_conf" in s["flags"] for s in segs)
    _log(f"done in {time.time() - t0:.0f}s: {len(segs)} lines, {n_x} crosstalk, "
         f"{n_low} uncertain speaker -> {paths['md']}")
    return transcript
