"""Voice library: enrolled speaker embeddings, identification, cluster naming.

Embeddings are plain float32 vectors; this module never touches audio.
Layout on disk: <path>/library.json + <path>/emb/<slug>.npy (one row per sample).
"""
import json
import os
import re
import tempfile
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _atomic_write(path, write_fn, mode):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as f:
            write_fn(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def unknown_label(i):
    """0 -> 'Unknown A', 25 -> 'Unknown Z', 26 -> 'Unknown AA', ..."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return f"Unknown {s}"


class VoiceLibrary:
    def __init__(self, path):
        self.path = str(path)
        self.emb_dir = os.path.join(self.path, "emb")
        self.json_path = os.path.join(self.path, "library.json")
        self.dim = None
        self.people = {}
        self._cache = {}
        if os.path.exists(self.json_path):
            # Deliberately not swallowing errors: a silently empty library
            # would be saved over the real one on the next mutation.
            with open(self.json_path) as f:
                data = json.load(f)
            self.dim = data.get("dim")
            self.people = data.get("people", {})

    # ---- persistence -------------------------------------------------
    def _emb_path(self, name):
        return os.path.join(self.emb_dir, f"{_slug(name)}.npy")

    def _save(self):
        data = {"dim": self.dim, "people": self.people}
        _atomic_write(self.json_path, lambda f: json.dump(data, f, indent=2), "w")

    def _write_vectors(self, name, vecs):
        _atomic_write(self._emb_path(name), lambda f: np.save(f, vecs), "wb")
        self._cache[name] = vecs

    def vectors(self, name):
        if name not in self._cache:
            p = self._emb_path(name)
            v = np.load(p) if os.path.exists(p) else np.zeros((0, self.dim or 0), np.float32)
            self._cache[name] = v.reshape(-1, v.shape[-1]) if v.size else v
        return self._cache[name]

    # ---- mutation ----------------------------------------------------
    def add(self, name, vectors, role="member", source="", max_per_person=None):
        vecs = _unit(np.atleast_2d(np.asarray(vectors, dtype=np.float32)))
        if vecs.size == 0:
            return
        if self.dim is None:
            self.dim = int(vecs.shape[1])
        elif vecs.shape[1] != self.dim:
            raise ValueError(f"embedding dim {vecs.shape[1]} != library dim {self.dim}")
        clash = [p for p in self.people if p != name and _slug(p) == _slug(name)]
        if clash:
            raise ValueError(f"name {name!r} collides with existing {clash[0]!r}")

        existing = self.vectors(name) if name in self.people else np.zeros((0, self.dim), np.float32)
        combined = np.vstack([existing, vecs]) if len(existing) else vecs
        if max_per_person and len(combined) > max_per_person:
            combined = combined[-max_per_person:]  # keep newest
        self._write_vectors(name, combined)

        info = self.people.setdefault(name, {"role": role, "sources": []})
        info["role"] = info.get("role", role) if source == "review" else role
        info["n"] = int(len(combined))
        info["updated"] = datetime.now().isoformat(timespec="seconds")
        if source and source not in info["sources"]:
            info["sources"].append(source)
        self._save()

    def remove(self, name):
        if name not in self.people:
            return
        p = self._emb_path(name)
        if os.path.exists(p):
            os.remove(p)
        self._cache.pop(name, None)
        del self.people[name]
        if not self.people:
            self.dim = None
        self._save()

    def rename(self, old, new):
        if old not in self.people or old == new:
            return
        old_vecs, old_info = self.vectors(old), self.people[old]
        if new in self.people:  # merge into the existing person
            self.add(new, old_vecs, role=self.people[new]["role"], source="merge")
            for s in old_info.get("sources", []):
                if s not in self.people[new]["sources"]:
                    self.people[new]["sources"].append(s)
        else:
            self.people[new] = dict(old_info)
            self._write_vectors(new, old_vecs)
        if _slug(old) != _slug(new) and os.path.exists(self._emb_path(old)):
            os.remove(self._emb_path(old))
        self._cache.pop(old, None)
        del self.people[old]
        self._save()

    # ---- queries -----------------------------------------------------
    def names(self):
        return sorted(self.people)

    def centroid(self, name):
        v = self.vectors(name)
        if not len(v):
            return np.zeros(self.dim or 0, np.float32)
        return _unit(v.mean(axis=0))

    def score_matrix(self, emb):
        """Cosine score of each row of emb against each person: [n, P].

        Person score = max(centroid cosine, mean of top-3 individual cosines),
        which tolerates people whose enrollment spans different mics/rooms.
        """
        emb = _unit(np.atleast_2d(emb))
        names = self.names()
        out = np.zeros((len(emb), len(names)), np.float32)
        for j, name in enumerate(names):
            v = self.vectors(name)
            if not len(v):
                continue
            c = emb @ self.centroid(name)
            sims = emb @ v.T
            k = min(3, sims.shape[1])
            top = -np.sort(-sims, axis=1)[:, :k].mean(axis=1)
            out[:, j] = np.maximum(c, top)
        return out, names


def identify(lib, emb, threshold=0.55, margin=0.05):
    S, names = lib.score_matrix(emb)
    results = []
    for row in S:
        if not names:
            results.append({"name": None, "score": 0.0, "second": None, "second_score": 0.0})
            continue
        order = np.argsort(-row)
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else None
        b, s2 = float(row[best]), float(row[second]) if second is not None else 0.0
        ok = b >= threshold and (b - s2) >= margin
        results.append({"name": names[best] if ok else None, "score": b,
                        "second": names[second] if second is not None else None,
                        "second_score": s2})
    return results


def _cluster_centroids(turns, emb):
    order, members = [], {}
    for i, t in enumerate(turns):
        c = t["cluster"]
        if c not in members:
            order.append(c)
            members[c] = []
        members[c].append(i)
    cents = {}
    for c in order:
        idx = [i for i in members[c] if not turns[i].get("overlap")] or members[c]
        w = np.array([max(1e-3, turns[i]["end"] - turns[i]["start"]) for i in idx], np.float32)
        cents[c] = _unit((emb[idx] * w[:, None]).sum(axis=0))
    return order, cents


def name_clusters(lib, turns, emb, threshold=0.55, margin=0.05):
    """Map diarization clusters to library people, one-to-one; rest -> Unknown X."""
    emb = np.atleast_2d(np.asarray(emb, np.float32))
    order, cents = _cluster_centroids(turns, emb)
    speakers = {}
    assigned = {}
    if order and lib.names():
        S, names = lib.score_matrix(np.stack([cents[c] for c in order]))
        cost = np.where(S >= threshold, -S, 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if S[i, j] >= threshold:
                assigned[order[i]] = (names[j], float(S[i, j]))
        best_any = {c: float(S[i].max()) for i, c in enumerate(order)}
    else:
        best_any = {c: 0.0 for c in order}
    n_unknown = 0
    for c in order:
        if c in assigned:
            name, score = assigned[c]
            speakers[c] = {"name": name, "score": score, "source": "voicelib"}
        else:
            speakers[c] = {"name": unknown_label(n_unknown), "score": best_any[c], "source": "unknown"}
            n_unknown += 1
    return speakers


def turn_confidence(lib, turns, emb, speakers):
    emb = _unit(np.atleast_2d(emb))
    cents = {}
    out = []
    for i, t in enumerate(turns):
        spk = speakers.get(t["cluster"], {})
        name = spk.get("name", "")
        if spk.get("source") != "voicelib" or name not in lib.people:
            out.append(0.0)
            continue
        if name not in cents:
            cents[name] = lib.centroid(name)
        score = float(emb[i] @ cents[name])
        if t["end"] - t["start"] < 1.5:
            score *= 0.7
        out.append(max(0.0, score))
    return out


def learn_from_review(lib, turns, emb, final_names, min_dur=2.0, max_per_person=200):
    """Add embeddings of human-confirmed, clean, long-enough turns to the library.

    final_names: list aligned with turns, or dict keyed by turn id.
    """
    emb = np.atleast_2d(np.asarray(emb, np.float32))
    per_person = {}
    for i, t in enumerate(turns):
        name = final_names[i] if isinstance(final_names, (list, tuple)) else final_names.get(t.get("id", i))
        if (not name or name.startswith("Unknown") or t.get("overlap")
                or t["end"] - t["start"] < min_dur):
            continue
        per_person.setdefault(name, []).append(emb[i])
    for name, vecs in per_person.items():
        role = lib.people.get(name, {}).get("role", "invitee")
        lib.add(name, np.stack(vecs), role=role, source="review", max_per_person=max_per_person)
