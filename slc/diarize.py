"""Speaker diarization of an aligned meeting -> turns.json + embeddings.npy.

Backend: pyannote community-1 via the cpp-annote C++ port (bin/slc-diarize,
built by tools/build_diarizer.sh). No Hugging Face token needed: Moonshine
distributes the two community-1 ONNX models ungated.

1. mix all aligned recorders -> aligned/mix.wav (diarize sees every voice).
2. cut the mix into ~30 min blocks at quiet points; diarize each block
   (full VBx clustering within a block, bounded memory/time).
3. per turn: 256-d WeSpeaker embedding from the turn's loudest recorder,
   using only the part not overlapped by other speakers.
4. link block-local speakers into meeting-wide clusters (average-linkage
   cosine clustering, never merging two speakers of the same block).
5. per-recorder level of every turn -> channel_rms_db / best_channel.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

from slc.audio import duration_s, read_segment

SR = 16000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin", "slc-diarize")


# ---- models ------------------------------------------------------------
def model_dir():
    from moonshine_voice import get_diarization_model
    return get_diarization_model()


def _nvidia_lib_path():
    import site
    dirs = []
    for sp in site.getsitepackages():
        nv = os.path.join(sp, "nvidia")
        if os.path.isdir(nv):
            dirs += [os.path.join(nv, d, "lib") for d in sorted(os.listdir(nv))
                     if os.path.isdir(os.path.join(nv, d, "lib"))]
    return ":".join(dirs)


def _gpu_available():
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


class Embedder:
    """WeSpeaker ResNet34 (community-1) on 16 kHz audio -> unit 256-d vectors."""

    def __init__(self, mdir=None, gpu=True):
        import onnxruntime as ort
        mdir = mdir or model_dir()
        providers = ["CPUExecutionProvider"]
        if gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            providers.insert(0, "CUDAExecutionProvider")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(os.path.join(mdir, "embedding.ort"), so, providers=providers)

    @staticmethod
    def fbank(audio):
        import torch
        import torchaudio.compliance.kaldi as kaldi
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None] * 32768.0
        fb = kaldi.fbank(wav, num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
                         dither=0.0, window_type="hamming", sample_frequency=SR)
        return (fb - fb.mean(dim=0, keepdim=True)).numpy()

    def __call__(self, audio):
        if len(audio) < int(0.5 * SR):  # model needs a little context
            audio = np.pad(audio, (0, int(0.5 * SR) - len(audio)))
        fb = self.fbank(audio)[None]
        w = np.ones(fb.shape[:2], np.float32)
        e = self.sess.run(None, {"fbank": fb, "weights": w})[0][0]
        return e / max(np.linalg.norm(e), 1e-12)


# ---- audio prep --------------------------------------------------------
def make_mix(aligned_paths, out_path, block_s=60):
    """Average of all aligned recorders, written blockwise."""
    dur = min(duration_s(p) for p in aligned_paths)
    with sf.SoundFile(out_path, "w", samplerate=SR, channels=1, subtype="PCM_16") as out:
        t = 0.0
        while t < dur:
            e = min(dur, t + block_s)
            chans = [read_segment(p, t, e) for p in aligned_paths]
            # Normalise each recorder's level so one hot mic doesn't dominate.
            chans = [c / max(1e-4, float(np.sqrt(np.mean(c ** 2)))) * 0.05 for c in chans]
            out.write(np.clip(np.mean(chans, axis=0), -1, 1))
            t = e
    return out_path


def block_bounds(wav, block_s=1800.0, search_s=30.0):
    """[(start, end)] blocks of ~block_s, each cut at the quietest 0.5 s nearby."""
    dur = duration_s(wav)
    cuts = [0.0]
    target = block_s
    while target < dur - block_s / 4:
        a, b = max(cuts[-1] + 1, target - search_s), min(dur, target + search_s)
        x = read_segment(wav, a, b)
        hop = SR // 10
        n = len(x) // hop
        e = np.convolve((x[:n * hop].reshape(n, hop) ** 2).mean(1), np.ones(5), "same")
        cuts.append(a + (int(np.argmin(e)) + 0.5) * 0.1)
        target = cuts[-1] + block_s
    cuts.append(dur)
    return list(zip(cuts[:-1], cuts[1:]))


def run_cpp_annote(audio, mdir, gpu=True):
    if not os.path.exists(BIN):
        raise RuntimeError(f"{BIN} missing; run tools/build_diarizer.sh")
    env = dict(os.environ)
    if gpu and _gpu_available():
        env["SLC_ORT_CUDA"] = "1"
        env["LD_LIBRARY_PATH"] = _nvidia_lib_path() + ":" + env.get("LD_LIBRARY_PATH", "")
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as f:
        audio.astype("<f4").tofile(f)
        tmp = f.name
    try:
        r = subprocess.run([BIN, os.path.join(mdir, "segmentation.ort"),
                            os.path.join(mdir, "embedding.ort"), tmp],
                           capture_output=True, text=True, env=env)
    finally:
        os.remove(tmp)
    if r.returncode != 0:
        raise RuntimeError(f"slc-diarize failed ({r.returncode}): {r.stderr.strip()[-500:]}")
    return json.loads(r.stdout)["turns"]


# ---- turn post-processing ---------------------------------------------
def clean_turns(raw, min_dur=0.25, merge_gap=0.5):
    """Merge same-speaker turns split by short gaps; drop slivers."""
    by_spk = {}
    for t in sorted(raw, key=lambda t: t["start"]):
        lst = by_spk.setdefault(t["speaker"], [])
        if lst and t["start"] - lst[-1]["end"] <= merge_gap:
            lst[-1]["end"] = max(lst[-1]["end"], t["end"])
        else:
            lst.append(dict(t))
    out = [t for lst in by_spk.values() for t in lst if t["end"] - t["start"] >= min_dur]
    return sorted(out, key=lambda t: (t["start"], t["end"]))


def mark_overlaps(turns, min_overlap=0.2):
    for t in turns:
        t["overlap"], t["overlaps_with"] = False, []
    for i, a in enumerate(turns):
        for j in range(i + 1, len(turns)):
            b = turns[j]
            if b["start"] >= a["end"]:
                break
            if a["cluster"] == b["cluster"]:
                continue
            if min(a["end"], b["end"]) - max(a["start"], b["start"]) >= min_overlap:
                a["overlap"] = b["overlap"] = True
                a["overlaps_with"].append(b["id"])
                b["overlaps_with"].append(a["id"])
    return turns


def clean_intervals(turn, turns_by_id):
    """Parts of the turn not covered by any overlapping turn."""
    segs = [(turn["start"], turn["end"])]
    for oid in turn["overlaps_with"]:
        o = turns_by_id[oid]
        nxt = []
        for s, e in segs:
            if o["end"] <= s or o["start"] >= e:
                nxt.append((s, e))
                continue
            if o["start"] > s:
                nxt.append((s, o["start"]))
            if o["end"] < e:
                nxt.append((o["end"], e))
        segs = nxt
    return [(s, e) for s, e in segs if e - s >= 0.3]


def channel_levels(turn, aligned_paths):
    db = []
    for p in aligned_paths:
        x = read_segment(p, turn["start"], turn["end"])
        db.append(round(20 * np.log10(max(1e-6, float(np.sqrt(np.mean(x ** 2))) if len(x) else 1e-6)), 1))
    return db


def link_speakers(keys, cents, blocks, threshold=0.5):
    """Average-linkage agglomerative clustering with a same-block cannot-link.

    keys[i] is a block-local speaker, cents[i] its unit centroid, blocks[i] its
    block. Merges while the best pair's mean cosine >= threshold.
    """
    groups = [[i] for i in range(len(keys))]
    S = cents @ cents.T
    while True:
        best, pair = threshold, None
        for a in range(len(groups)):
            ba = {blocks[i] for i in groups[a]}
            for b in range(a + 1, len(groups)):
                if ba & {blocks[i] for i in groups[b]}:
                    continue
                sim = float(S[np.ix_(groups[a], groups[b])].mean())
                if sim >= best:
                    best, pair = sim, (a, b)
        if pair is None:
            break
        a, b = pair
        groups[a] += groups.pop(b)
    label = {}
    groups.sort(key=lambda g: min(g))
    for gi, g in enumerate(groups):
        for i in g:
            label[keys[i]] = f"SPK_{gi:02d}"
    return label


# ---- main entry --------------------------------------------------------
def diarize_meeting(workdir, block_s=1800.0, gpu=True, link_threshold=0.5, log=print):
    with open(os.path.join(workdir, "alignment.json")) as f:
        alignment = json.load(f)
    aligned = [os.path.join(workdir, "aligned", f"rec{r['index']}.wav") for r in alignment["recorders"]]
    mix = os.path.join(workdir, "aligned", "mix.wav")
    if not os.path.exists(mix):
        log("mixing recorders ...")
        make_mix(aligned, mix)

    mdir = model_dir()
    turns = []
    for bi, (bs, be) in enumerate(block_bounds(mix, block_s)):
        log(f"diarizing block {bi} [{bs:.0f}s - {be:.0f}s] ...")
        raw = run_cpp_annote(read_segment(mix, bs, be), mdir, gpu=gpu)
        for t in clean_turns(raw):
            turns.append({"start": round(bs + t["start"], 3), "end": round(bs + t["end"], 3),
                          "block": bi, "local": f"b{bi}_s{t['speaker']}"})
    turns.sort(key=lambda t: (t["start"], t["end"]))
    for i, t in enumerate(turns):
        t["id"] = i
        t["cluster"] = t["local"]
    mark_overlaps(turns)

    log(f"{len(turns)} turns; levels + embeddings ...")
    emb_model = Embedder(mdir, gpu=gpu)
    by_id = {t["id"]: t for t in turns}
    embs = np.zeros((len(turns), 256), np.float32)
    for t in turns:
        t["channel_rms_db"] = channel_levels(t, aligned)
        t["best_channel"] = int(np.argmax(t["channel_rms_db"]))
        parts = clean_intervals(t, by_id) or [(t["start"], t["end"])]
        src = aligned[t["best_channel"]]
        audio, budget = [], 30.0  # cap per-turn embedding audio
        for s, e in parts:
            take = min(e - s, budget)
            if take <= 0:
                break
            audio.append(read_segment(src, s, s + take))
            budget -= take
        embs[t["id"]] = emb_model(np.concatenate(audio))

    # Link block-local speakers into meeting-wide clusters.
    keys = sorted({t["local"] for t in turns})
    cents, blocks = [], []
    for k in keys:
        idx = [t["id"] for t in turns if t["local"] == k]
        clean = [i for i in idx if not by_id[i]["overlap"]] or idx
        w = np.array([by_id[i]["end"] - by_id[i]["start"] for i in clean], np.float32)
        c = (embs[clean] * w[:, None]).sum(0)
        cents.append(c / max(np.linalg.norm(c), 1e-12))
        blocks.append(by_id[idx[0]]["block"])
    label = link_speakers(keys, np.array(cents, np.float32), blocks, link_threshold) if keys else {}

    out = []
    for t in turns:
        out.append({"id": t["id"], "start": t["start"], "end": t["end"], "cluster": label[t["local"]],
                    "overlap": t["overlap"], "overlaps_with": t["overlaps_with"],
                    "channel_rms_db": t["channel_rms_db"], "best_channel": t["best_channel"]})
    with open(os.path.join(workdir, "turns.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.save(os.path.join(workdir, "embeddings.npy"), embs)
    n_spk = len(set(label.values()))
    log(f"wrote {len(out)} turns, {n_spk} speakers, "
        f"{sum(t['overlap'] for t in out)} overlapped turns")
    return out, embs


if __name__ == "__main__":
    diarize_meeting(sys.argv[1])
