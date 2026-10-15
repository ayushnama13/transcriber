import json
import os
import pathlib
import tempfile
import unittest
import subprocess
import sys

from slc.notes import (apply_flags, calibrate, harvest_enrollment, match_notes,
                       normalize_name, parse_notes)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEGS = [
    {"id": 0, "start": 0.0, "end": 4.0, "speaker": "Chair Person", "text": "Good evening everyone, let us begin the meeting.", "flags": []},
    {"id": 1, "start": 4.5, "end": 9.0, "speaker": "Rahul Sharma", "text": "I think you are doing good work on the hostel issue.", "flags": []},
    {"id": 2, "start": 9.5, "end": 11.0, "speaker": "Unknown A", "text": "I second the motion.", "flags": []},
    {"id": 3, "start": 11.2, "end": 15.0, "speaker": "Chair Person", "text": "All in favour? The motion passes.", "flags": ["crosstalk"]},
    {"id": 4, "start": 16.0, "end": 22.0, "speaker": "Rahul Sharma", "text": "Next, the mess committee budget needs to be revised", "flags": []},
    {"id": 5, "start": 22.0, "end": 26.0, "speaker": "Rahul Sharma", "text": "because the vendor raised prices by twenty percent.", "flags": []},
]

NOTES_TXT = """# Agenda: hostel issues
- Rahul: "you are doing good wrok"
* [00:00:09] Priya: I second the motion
Chair: Motion passes
Dr. Rahul Sharma: the mess committee budget needs to be revised because
the vendor raised prices
Meera: something that was never said in the meeting at all really
"""


class TestNotes(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._d.name)

    def tearDown(self):
        self._d.cleanup()

    @staticmethod
    def _notes(tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text(NOTES_TXT)
        return parse_notes(p)


    def test_parse(self):
        tmp_path = self.tmp
        n = self._notes(tmp_path)
        assert [x["name"] for x in n] == ["Rahul", "Priya", "Chair", "Dr. Rahul Sharma", "Meera"]
        assert n[0]["text"] == "you are doing good wrok"
        assert n[1]["time"] == 9.0
        assert n[3]["text"].endswith("the vendor raised prices")  # continuation line joined
        assert normalize_name("Dr. Rahul  Sharma") == "rahul sharma"


    def test_match_and_calibrate(self):
        tmp_path = self.tmp
        n = self._notes(tmp_path)
        m = match_notes(n, SEGS)
        assert m[0]["segment_ids"] == [1]
        assert m[1]["segment_ids"] == [2]
        assert m[2]["segment_ids"] == [3]
        assert m[3]["segment_ids"] == [4, 5]  # spans two segments
        assert m[4]["segment_ids"] == []      # not in transcript

        aliases = {"Chair": "Chair Person", "Rahul": "Rahul Sharma"}
        r = calibrate(n, SEGS, m, aliases)
        assert r["n_matched"] == 4 and abs(r["coverage"] - 0.8) < 1e-9
        assert abs(r["speaker_accuracy"] - 0.75) < 1e-9  # Priya mislabelled as Unknown A
        assert r["confusion"] == [{"note_name": "Priya", "predicted": "Unknown A", "count": 1}]
        # Without aliases, "Rahul" vs "Rahul Sharma" and "Chair" vs "Chair Person" mismatch too.
        assert calibrate(n, SEGS, m)["speaker_accuracy"] < 0.75

        tr = apply_flags({"segments": [dict(s, flags=list(s["flags"])) for s in SEGS]}, r, n, m)
        s2 = tr["segments"][2]
        assert "notes_mismatch" in s2["flags"] and s2["notes_name"] == "Priya"
        assert "notes_mismatch" not in tr["segments"][1]["flags"]
        assert tr["segments"][1]["notes_name"] == "Rahul"


    def test_harvest(self):
        tmp_path = self.tmp
        n = self._notes(tmp_path)
        m = match_notes(n, SEGS)
        h = harvest_enrollment(m, n, SEGS)
        ids = {x["segment_id"] for x in h}
        assert 3 not in ids  # crosstalk excluded
        assert 2 not in ids  # 1.5 s, too short
        assert {4, 5} <= ids


    def test_cli(self):
        tmp_path = self.tmp
        work = tmp_path / "w"
        work.mkdir()
        (work / "transcript.json").write_text(json.dumps({"meeting": "t", "segments": SEGS}))
        notes = tmp_path / "notes.txt"
        notes.write_text(NOTES_TXT)
        aliases = tmp_path / "aliases.json"
        aliases.write_text(json.dumps({"Chair": "Chair Person", "Rahul": "Rahul Sharma"}))
        r = subprocess.run([sys.executable, "-m", "slc", "calibrate", str(work), "--notes", str(notes),
                            "--aliases", str(aliases)], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "75.0%" in r.stdout
        rep = json.loads((work / "calibration.json").read_text())
        assert rep["n_matched"] == 4
        tr = json.loads((work / "transcript.json").read_text())
        assert "notes_mismatch" in tr["segments"][2]["flags"]

        bad = subprocess.run([sys.executable, "-m", "slc", "calibrate", str(tmp_path / "nope"),
                              "--notes", str(notes)], cwd=ROOT, capture_output=True, text=True)
        assert bad.returncode == 1


if __name__ == "__main__":
    unittest.main()
