import json
import os
import re
import shutil
import tempfile
import unittest

from slc.kit import assign, import_edits, make_kits, make_pieces
from tests.test_review import make_workdir


def edits_file(path, kit, edits):
    with open(path, "w") as f:
        json.dump({"format": "slc-kit-edits/1", "kit_id": kit["kit_id"], "reviewer": kit["reviewer"],
                   "edits": edits}, f)
    return path


class KitTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.w = [os.path.join(self.d, f"clip_{i}") for i in range(2)]
        for w in self.w:
            make_workdir(w)
        self.out = os.path.join(self.d, "kits")

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_assign(self):
        self.assertEqual([len(ix) for _, ix in assign(list(range(50)), per=5)], [5] * 10)
        self.assertEqual([len(ix) for _, ix in assign(list(range(51)), per=5)], [5] * 7 + [4] * 4)  # evened out
        got = assign(list(range(7)), people=["A", "B", "C"])
        self.assertEqual([ix for _, ix in got], [[0, 1, 2], [3, 4], [5, 6]])

    def test_pieces_split_and_skip_reviewed(self):
        p = make_pieces(self.w[:1], piece_s=10, log=lambda *a: None)  # 20 s file -> 2 pieces
        self.assertEqual([[s["id"] for s in x["segments"]] for x in p], [[0, 1], [2]])
        t = json.load(open(os.path.join(self.w[1], "transcript.json")))
        for s in t["segments"]:
            s["reviewed"] = True
        json.dump(t, open(os.path.join(self.w[1], "transcript.json"), "w"))
        self.assertEqual(len(make_pieces(self.w, log=lambda *a: None)), 1)

    def test_kit_roundtrip(self):
        man = make_kits(self.w, self.out, people=["Asha", "Ravi"], names=["Sindhu HL"], log=lambda *a: None)
        a, r = man["kits"]
        html = open(os.path.join(self.out, a["file"]), encoding="utf-8").read()
        data = json.loads(re.search(r'<script id="kit" type="application/json">(.*?)</script>', html, re.S).group(1))
        self.assertEqual(data["reviewer"], "Asha")
        self.assertIn("Sindhu HL", data["speakers"])
        self.assertIn(f'id="audio-{a["pieces"][0]}"', html)
        with self.assertRaises(FileExistsError):
            make_kits(self.w, self.out, log=lambda *a: None)

        pa, pr = a["pieces"][0], r["pieces"][0]
        f1 = edits_file(os.path.join(self.d, "e1.json"), a, {
            f"{pa}:0": {"text": "hello there, all", "speaker": "Sindhu HL", "reviewed": True, "t": 10},
            f"{pa}:1": {"note": "can't hear", "t": 10},
            f"{pr}:0": {"text": "NOT MINE", "t": 10}})          # Ravi's segment: must be refused
        res = import_edits(self.out, [f1], export=False)
        self.assertEqual(len(res["problems"]), 1)
        segs = {s["id"]: s for s in json.load(open(os.path.join(self.w[0], "transcript.json")))["segments"]}
        self.assertEqual((segs[0]["text"], segs[0]["speaker"], segs[0]["reviewed"], segs[0]["reviewed_by"]),
                         ("hello there, all", "Sindhu HL", True, "Asha"))
        self.assertEqual(segs[1]["review_note"], "can't hear")
        other = json.load(open(os.path.join(self.w[1], "transcript.json")))["segments"]
        self.assertNotIn("NOT MINE", [s["text"] for s in other])
        self.assertEqual(res["status"][0]["done"], 1)

        # Older file imported later must not undo the newer edit; newer one wins.
        f0 = edits_file(os.path.join(self.d, "e0.json"), a, {f"{pa}:0": {"text": "old", "t": 5}})
        f2 = edits_file(os.path.join(self.d, "e2.json"), a, {f"{pa}:0": {"text": "newest", "t": 20}})
        import_edits(self.out, [f0], export=False)
        segs = {s["id"]: s for s in json.load(open(os.path.join(self.w[0], "transcript.json")))["segments"]}
        self.assertEqual(segs[0]["text"], "hello there, all")
        import_edits(self.out, [f2, f0], export=True)
        segs = {s["id"]: s for s in json.load(open(os.path.join(self.w[0], "transcript.json")))["segments"]}
        self.assertEqual(segs[0]["text"], "newest")
        self.assertIn("newest", open(os.path.join(self.w[0], "out", "transcript.txt")).read())


if __name__ == "__main__":
    unittest.main()
