"""Use the council's handwritten "Name: what they said" notes to calibrate.

parse_notes -> match_notes (fuzzy-locate each remark in the transcript)
-> calibrate (speaker accuracy, confusions) -> apply_flags / harvest_enrollment.
"""
import re

from rapidfuzz import fuzz, process

_TIME = re.compile(r"^\s*[\[(]\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*[\])]\s*")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# A speaker label: 1-6 name-ish tokens (letters, dots, apostrophes, hyphens).
_LABEL = re.compile(r"^([A-Za-z][A-Za-z.'\- ]{0,60}?)\s*:\s+(.*)$")
_TITLES = {"dr", "prof", "mr", "ms", "mrs", "sir", "madam", "shri", "smt"}


def _parse_time(m):
    a, b, c = m.group(1), m.group(2), m.group(3)
    if c is not None:
        return int(a) * 3600 + int(b) * 60 + int(c)
    return int(a) * 60 + int(b)  # (mm:ss) when only two fields


def parse_notes(path):
    notes = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):  # blank, or a "# Agenda item" heading
                continue
            line = _BULLET.sub("", line)
            t = None
            m = _TIME.match(line)
            if m:
                t = float(_parse_time(m))
                line = line[m.end():]
            m = _LABEL.match(line)
            if m and len(m.group(1).split()) <= 6:
                text = m.group(2).strip().strip('"“”\'')
                notes.append({"idx": len(notes), "name": m.group(1).strip(), "text": text, "time": t})
            elif notes:  # continuation of the previous remark
                notes[-1]["text"] = (notes[-1]["text"] + " " + line.strip('"“”\'')).strip()
    return notes


def normalize_name(s):
    toks = re.sub(r"[^a-z ]+", " ", s.lower().replace(".", " ")).split()
    return " ".join(t for t in toks if t not in _TITLES)


def _canon(name, aliases):
    if not name:
        return ""
    if aliases:
        for k, v in aliases.items():
            if normalize_name(k) == normalize_name(name):
                name = v
                break
    return normalize_name(name)


def _norm_text(s):
    return " ".join(re.sub(r"[^\w\s']", " ", s.lower()).split())


def match_notes(notes, segments, min_score=70):
    segs = sorted(segments, key=lambda s: s["start"])
    texts = [_norm_text(s.get("text", "")) for s in segs]
    results = []
    last_pos = -1
    for n in notes:
        q = _norm_text(n["text"])
        best = {"note_idx": n["idx"], "segment_ids": [], "score": 0.0, "start": None, "end": None}
        if not q:
            results.append(best)
            continue
        allowed = range(len(segs))
        if n.get("time") is not None:
            allowed = [i for i in allowed if abs(segs[i]["start"] - n["time"]) <= 600]
        choices = {i: texts[i] for i in allowed if texts[i]}
        cands = process.extract(q, choices, scorer=fuzz.partial_ratio, limit=30)
        seen = set()
        scored = []
        for _, _, i in cands:
            for lo in range(max(0, i - 2), i + 1):
                for hi in range(i, min(len(segs), lo + 3)):
                    if (lo, hi) in seen or hi < lo:
                        continue
                    seen.add((lo, hi))
                    win = " ".join(texts[lo:hi + 1])
                    sc = fuzz.partial_ratio(q, win) if len(q) < len(win) else fuzz.ratio(q, win)
                    # Prefer tight windows: penalise extra segments slightly.
                    scored.append((sc - 1.5 * (hi - lo), lo, hi, sc))
        if scored:
            top = max(s[0] for s in scored)
            ties = [s for s in scored if s[0] >= top - 5]
            # Notes are written in order: among near-ties prefer the first after the last match.
            after = [s for s in ties if s[1] > last_pos]
            # Within that, the width-penalised score decides; earliest start breaks exact ties.
            pick = max(after or ties, key=lambda s: (s[0], -s[1]))
            _, lo, hi, sc = pick
            if sc >= min_score:
                ids = [segs[k]["id"] for k in range(lo, hi + 1)]
                best.update(segment_ids=ids, score=float(sc),
                            start=segs[lo]["start"], end=segs[hi]["end"])
                last_pos = lo
        results.append(best)
    return results


def _segment_speaker(seg_ids, by_id):
    """Majority speaker (by duration) across the matched window."""
    dur = {}
    for sid in seg_ids:
        s = by_id[sid]
        dur[s.get("speaker", "")] = dur.get(s.get("speaker", ""), 0) + (s["end"] - s["start"])
    return max(dur, key=dur.get) if dur else ""


def calibrate(notes, segments, matches, aliases=None):
    by_id = {s["id"]: s for s in segments}
    n_matched = correct = 0
    per_person, confusion, mismatched = {}, {}, []
    for n, m in zip(notes, matches):
        if not m["segment_ids"]:
            continue
        n_matched += 1
        want = _canon(n["name"], aliases)
        got_raw = _segment_speaker(m["segment_ids"], by_id)
        got = _canon(got_raw, aliases)
        pp = per_person.setdefault(n["name"], {"notes": 0, "correct": 0})
        pp["notes"] += 1
        if want == got:
            correct += 1
            pp["correct"] += 1
        else:
            key = (n["name"], got_raw)
            confusion[key] = confusion.get(key, 0) + 1
            mismatched.append({"note_idx": n["idx"], "note_name": n["name"], "predicted": got_raw,
                               "segment_ids": m["segment_ids"], "start": m["start"]})
    return {
        "n_notes": len(notes),
        "n_matched": n_matched,
        "coverage": n_matched / len(notes) if notes else 0.0,
        "speaker_accuracy": correct / n_matched if n_matched else 0.0,
        "confusion": [{"note_name": a, "predicted": b, "count": c}
                      for (a, b), c in sorted(confusion.items(), key=lambda kv: -kv[1])],
        "per_person": per_person,
        "mismatched": mismatched,
    }


def apply_flags(transcript, report, notes=None, matches=None):
    by_id = {s["id"]: s for s in transcript["segments"]}
    for mm in report["mismatched"]:
        for sid in mm["segment_ids"]:
            flags = by_id[sid].setdefault("flags", [])
            if "notes_mismatch" not in flags:
                flags.append("notes_mismatch")
            by_id[sid]["notes_name"] = mm["note_name"]
    if notes and matches:
        for n, m in zip(notes, matches):
            for sid in m["segment_ids"]:
                by_id[sid].setdefault("notes_name", n["name"])
    return transcript


def harvest_enrollment(matches, notes, segments, min_dur=2.0, min_score=85):
    by_id = {s["id"]: s for s in segments}
    out = []
    for n, m in zip(notes, matches):
        if m["score"] < min_score:
            continue
        for sid in m["segment_ids"]:
            s = by_id[sid]
            if "crosstalk" in s.get("flags", []) or s["end"] - s["start"] < min_dur:
                continue
            out.append({"name": n["name"], "start": s["start"], "end": s["end"], "segment_id": sid})
    return out
