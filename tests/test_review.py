import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

from slc.review import create_app


def make_workdir(d):
    os.makedirs(os.path.join(d, "aligned"))
    x = (0.1 * np.sin(np.arange(16000 * 20) * 0.05)).astype(np.float32)
    for name in ("mix", "rec0", "rec1"):
        sf.write(os.path.join(d, "aligned", f"{name}.wav"), x, 16000)
    segs = [
        {"id": 0, "turn_id": 0, "start": 1.0, "end": 4.0, "speaker": "Unknown A", "speaker_confidence": 0.0,
         "cluster": "SPK_00", "text": "hello there", "channel": 0, "flags": ["low_speaker_conf"],
         "overlaps_with": [], "reviewed": False, "part": 0},
        {"id": 1, "turn_id": 1, "start": 5.0, "end": 8.0, "speaker": "Unknown A", "speaker_confidence": 0.0,
         "cluster": "SPK_00", "text": "second line", "channel": 1, "flags": [],
         "overlaps_with": [], "reviewed": False, "part": 0},
        {"id": 2, "turn_id": 2, "start": 12.0, "end": 14.0, "speaker": "Rahul", "speaker_confidence": 0.9,
         "cluster": "SPK_01", "text": "third", "channel": 0, "flags": [],
         "overlaps_with": [], "reviewed": False, "part": 1},
    ]
    with open(os.path.join(d, "transcript.json"), "w") as f:
        json.dump({"meeting": "T", "segments": segs}, f)


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        make_workdir(self.d)
        self.c = create_app(self.d).test_client()

    def tearDown(self):
        shutil.rmtree(self.d)

    def saved(self):
        with open(os.path.join(self.d, "transcript.json")) as f:
            return {s["id"]: s for s in json.load(f)["segments"]}

    def test_meta_and_part(self):
        m = self.c.get("/api/meta").get_json()
        self.assertEqual([p["part"] for p in m["parts"]], [0, 1])
        self.assertEqual(m["parts"][0]["flagged"], 1)
        self.assertEqual(m["channels"], [0, 1])
        self.assertIn("Rahul", m["speakers"])
        self.assertEqual(len(self.c.get("/api/part/0").get_json()["segments"]), 2)
        self.assertEqual(self.c.get("/api/part/9").status_code, 404)

    def test_edit_is_saved_and_logged(self):
        r = self.c.post("/api/segment/0", json={"reviewer": "asha",
                        "changes": {"text": "Hello there.", "speaker": "Priya", "reviewed": True}})
        self.assertEqual(r.status_code, 200)
        s = self.saved()[0]
        self.assertEqual((s["text"], s["speaker"], s["reviewed"], s["reviewed_by"]),
                         ("Hello there.", "Priya", True, "asha"))
        with open(os.path.join(self.d, "edits.jsonl")) as f:
            log = [json.loads(ln) for ln in f]
        self.assertEqual(log[0]["before"]["speaker"], "Unknown A")
        self.assertEqual(log[0]["reviewer"], "asha")

    def test_bad_edits_rejected(self):
        self.assertEqual(self.c.post("/api/segment/0", json={"changes": {"text": "x"}}).status_code, 400)
        for ch in ({"flags": []}, {"reviewed": "yes"}, {"speaker": "  "}):
            r = self.c.post("/api/segment/0", json={"reviewer": "a", "changes": ch})
            self.assertEqual(r.status_code, 400, ch)
        self.assertEqual(self.c.post("/api/segment/99", json={"reviewer": "a", "changes": {}}).status_code, 404)
        self.assertEqual(self.saved()[0]["text"], "hello there")

    def test_cluster_rename_skips_reviewed(self):
        self.c.post("/api/segment/1", json={"reviewer": "a", "changes": {"speaker": "Kiran", "reviewed": True}})
        r = self.c.post("/api/cluster/SPK_00", json={"reviewer": "a", "name": "Meera"}).get_json()
        self.assertEqual(r["updated"], [0])
        s = self.saved()
        self.assertEqual((s[0]["speaker"], s[1]["speaker"]), ("Meera", "Kiran"))

    def test_claim(self):
        self.c.post("/api/part/1/claim", json={"reviewer": "asha"})
        m = self.c.get("/api/meta").get_json()
        self.assertEqual(m["parts"][1]["claimed_by"], "asha")

    def test_audio(self):
        r = self.c.get("/audio/0?ch=1")
        self.assertEqual(r.mimetype, "audio/wav")
        info = sf.info(__import__("io").BytesIO(r.data))
        self.assertAlmostEqual(info.duration, 3.8, places=2)  # 3 s + 0.4 s pad each side
        self.assertEqual(self.c.get("/audio/0?ch=7").status_code, 404)
        self.assertEqual(self.c.get("/audio/0?ch=../x").status_code, 404)

    def test_page(self):
        self.assertIn(b"<title>Transcript Review</title>", self.c.get("/").data)


if __name__ == "__main__":
    unittest.main()
