"""Offline review kits: hand each checker one self-contained HTML page (audio embedded) for
their pieces of the meeting, then merge the edits files they send back.

    python -m slc kit WORKDIR [WORKDIR ...] --out kits/2026-08-22 --people "Asha,Ravi,..."   (or --per 5)
    python -m slc import-edits kits/2026-08-22 edits_*.json

kits/<x>/manifest.json stays with the coordinator: it maps piece ids back to workdirs and
segment ids. A reviewer's page only knows its kit_id, so an edits file can only touch the
segments that were in that reviewer's kit. Merging goes through review.Store, so every
applied change lands in transcript.json and edits.jsonl exactly like the web review app.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid

from slc.audio import duration_s

FORMAT = "slc-kit-edits/1"
PIECE_S = 240.0
MP3_KBPS = 32
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "kit_template.html")
GUIDE_DIR = os.path.join(os.path.dirname(HERE), "docs")


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "reviewer"


def _load_transcript(wd):
    with open(os.path.join(wd, "transcript.json")) as f:
        return json.load(f)


def make_pieces(workdirs, piece_s=PIECE_S, include_reviewed=False, log=print):
    """Cut each workdir's timeline into ~piece_s windows; a segment belongs to the window
    its start falls in. A workdir that is already one 4-min clip gives exactly one piece."""
    pieces = []
    for wd in workdirs:
        wd = os.path.abspath(wd)
        t = _load_transcript(wd)
        segs = sorted(t["segments"], key=lambda s: (s["start"], s["id"]))
        dur = duration_s(os.path.join(wd, "aligned", "mix.wav"))
        n = max(1, round(dur / piece_s))
        w = dur / n
        for k in range(n):
            a, b = k * w, (k + 1) * w
            mine = [s for s in segs if a <= s["start"] < b or (k == n - 1 and s["start"] >= b)]
            if not mine:
                continue
            label = f"{os.path.basename(wd)} {hms(a)}-{hms(b)}"
            if not include_reviewed and all(s.get("reviewed") for s in mine):
                log(f"  skip {label}: all {len(mine)} lines already reviewed")
                continue
            pieces.append({"workdir": wd, "meeting": t.get("meeting", ""), "label": label,
                           "start": max(0.0, min(a, mine[0]["start"])),
                           "end": min(dur, max(b, max(s["end"] for s in mine))),
                           "segments": mine})
    return pieces


def assign(pieces, people=None, per=5):
    """Consecutive pieces per person (context carries over between neighbouring pieces)."""
    if not people:
        n = max(1, -(-len(pieces) // per))
        people = [f"Reviewer {i + 1:02d}" for i in range(n)]
    people = people[:len(pieces)] or people[:1]
    q, r = divmod(len(pieces), len(people))
    out, i = [], 0
    for j, name in enumerate(people):
        k = q + (j < r)
        out.append((name, list(range(i, i + k))))
        i += k
    return out


def encode_mp3(wav, start, end):
    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, "p.mp3")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                        "-i", wav, "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", f"{MP3_KBPS}k",
                        dst], check=True)
        with open(dst, "rb") as f:
            return f.read()


def _page_segment(s):
    return {"id": s["id"], "start": round(s["start"], 2), "end": round(s["end"], 2),
            "speaker": s.get("speaker", ""), "text": s.get("text", ""), "flags": s.get("flags", []),
            "overlaps_with": s.get("overlaps_with", []), "reviewed": bool(s.get("reviewed")),
            "note": s.get("review_note", "")}


def render_page(kit, pieces_data, audio, speakers):
    with open(TEMPLATE, encoding="utf-8") as f:
        tpl = f.read()
    data = dict(kit, speakers=speakers, pieces=pieces_data)
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    blobs = "\n".join(f'<script type="text/plain" id="audio-{pid}">{base64.b64encode(b).decode()}</script>'
                      for pid, b in audio.items())
    return tpl.replace("__KIT_DATA__", js).replace("<!--__AUDIO__-->", blobs).replace(
        "__TITLE__", f"SLC review - {kit['reviewer']}")


def make_kits(workdirs, out, people=None, per=5, piece_s=PIECE_S, names=(), include_reviewed=False,
              log=print):
    if os.path.exists(os.path.join(out, "manifest.json")):
        raise FileExistsError(f"{out} already has kits; use a new --out so returned edits still match")
    pieces = make_pieces(workdirs, piece_s, include_reviewed, log)
    if not pieces:
        raise ValueError("nothing to review: every piece is already reviewed (use --include-reviewed)")
    os.makedirs(out, exist_ok=True)
    speakers = sorted({s["speaker"] for p in pieces for s in p["segments"] if s.get("speaker")} |
                      {n.strip() for n in names if n.strip()}, key=str.lower)
    manifest = {"format": "slc-kit/1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "kits": [], "pieces": {}}
    for i, p in enumerate(pieces):
        pid = f"p{i + 1:03d}"
        p["pid"] = pid
        manifest["pieces"][pid] = {"workdir": p["workdir"], "label": p["label"], "start": p["start"],
                                   "end": p["end"], "segment_ids": [s["id"] for s in p["segments"]]}
    used = set()
    for name, idx in assign(pieces, people, per):
        kit_id = uuid.uuid4().hex[:12]
        fname = f"review_{slug(name)}.html"
        while fname in used:
            fname = fname.replace(".html", "_.html")
        used.add(fname)
        audio, pdata = {}, []
        for i in idx:
            p = pieces[i]
            audio[p["pid"]] = encode_mp3(os.path.join(p["workdir"], "aligned", "mix.wav"), p["start"], p["end"])
            pdata.append({"pid": p["pid"], "label": p["label"], "meeting": p["meeting"],
                          "offset": round(p["start"], 3), "segments": [_page_segment(s) for s in p["segments"]]})
        kit = {"format": FORMAT, "kit_id": kit_id, "reviewer": name, "created": manifest["created"]}
        html = render_page(kit, pdata, audio, speakers)
        with open(os.path.join(out, fname), "w", encoding="utf-8") as f:
            f.write(html)
        manifest["kits"].append({"kit_id": kit_id, "reviewer": name, "file": fname,
                                 "pieces": [pieces[i]["pid"] for i in idx]})
        n_lines = sum(len(pieces[i]["segments"]) for i in idx)
        log(f"  {fname}: {name}, {len(idx)} pieces, {n_lines} lines, {len(html) / 1e6:.1f} MB")
    for g in ("reviewer-guide.pdf", "reviewer-guide.html"):
        if os.path.isfile(os.path.join(GUIDE_DIR, g)):
            shutil.copy(os.path.join(GUIDE_DIR, g), out)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def import_edits(kits_dir, files, export=True, log=print):
    """Apply returned edits files. Per segment the newest edit (by its own timestamp) wins;
    applied.json remembers what was applied so re-importing an old file never undoes newer work."""
    from slc.export import export_all
    from slc.review import Store

    man = _load_json(os.path.join(kits_dir, "manifest.json"))
    kits = {k["kit_id"]: k for k in man["kits"]}
    applied_path = os.path.join(kits_dir, "applied.json")
    applied = _load_json(applied_path) if os.path.exists(applied_path) else {}
    best, problems, conflicts = {}, [], 0
    for path in files:
        try:
            d = _load_json(path)
        except (OSError, ValueError) as e:
            problems.append(f"{path}: not a readable edits file ({e})")
            continue
        kit = kits.get(d.get("kit_id")) if isinstance(d, dict) else None
        if not isinstance(d, dict) or d.get("format") != FORMAT:
            problems.append(f"{path}: not an SLC edits file")
            continue
        if kit is None:
            problems.append(f"{path}: belongs to a different set of kits (kit_id {d.get('kit_id')})")
            continue
        for key, e in (d.get("edits") or {}).items():
            pid, _, sid = key.partition(":")
            if pid not in kit["pieces"] or not sid.isdigit() or \
                    int(sid) not in man["pieces"][pid]["segment_ids"] or not isinstance(e, dict):
                problems.append(f"{path}: ignored {key!r} (not in {kit['reviewer']}'s kit)")
                continue
            t = float(e.get("t") or 0)
            if key in best:
                conflicts += best[key][1] != kit["reviewer"]
                if best[key][0] >= t:
                    continue
            best[key] = (t, kit["reviewer"], e)

    stores, stats = {}, {}
    for key, (t, reviewer, e) in sorted(best.items()):
        if applied.get(key, -1) >= t:
            continue
        pid, _, sid = key.partition(":")
        sid = int(sid)
        wd = man["pieces"][pid]["workdir"]
        if wd not in stores:
            stores[wd] = Store(wd)
        st = stores[wd]
        seg = st.by_id[sid]
        st_ = stats.setdefault(reviewer, {"confirmed": 0, "text": 0, "speaker": 0, "notes": 0})
        changes = {}
        if isinstance(e.get("text"), str) and e["text"].strip() and e["text"].strip() != seg.get("text", ""):
            changes["text"] = e["text"].strip()
            st_["text"] += 1
        if isinstance(e.get("speaker"), str) and e["speaker"].strip() and e["speaker"].strip() != seg.get("speaker"):
            changes["speaker"] = e["speaker"].strip()
            st_["speaker"] += 1
        if isinstance(e.get("reviewed"), bool) and e["reviewed"] != bool(seg.get("reviewed")):
            changes["reviewed"] = e["reviewed"]
        if changes:
            st.update(sid, changes, reviewer)
        st_["confirmed"] += bool(e.get("reviewed"))
        note = e.get("note")
        if isinstance(note, str) and note.strip() != seg.get("review_note", ""):
            with st.lock:
                if note.strip():
                    seg["review_note"] = note.strip()
                else:
                    seg.pop("review_note", None)
                st._save_json(st.path, st.t)
                st._log({"reviewer": reviewer, "segment": sid, "after": {"review_note": note.strip()}})
            st_["notes"] += bool(note.strip())
        applied[key] = t

    for wd, st in stores.items():
        if export:
            export_all(st.t, os.path.join(wd, "out"))
    tmp = applied_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(applied, f, indent=1)
    os.replace(tmp, applied_path)
    return {"stats": stats, "problems": problems, "conflicts": conflicts,
            "workdirs": sorted(stores), "status": kit_status(man)}


def kit_status(man):
    """Per reviewer: confirmed lines / lines in their kit, read from the live transcripts."""
    cache, rows = {}, []
    for k in man["kits"]:
        done = total = 0
        for pid in k["pieces"]:
            p = man["pieces"][pid]
            if p["workdir"] not in cache:
                cache[p["workdir"]] = {s["id"]: s for s in _load_transcript(p["workdir"])["segments"]}
            segs = cache[p["workdir"]]
            total += len(p["segment_ids"])
            done += sum(bool(segs.get(i, {}).get("reviewed")) for i in p["segment_ids"])
        rows.append({"reviewer": k["reviewer"], "file": k["file"], "pieces": len(k["pieces"]),
                     "done": done, "total": total})
    return rows
