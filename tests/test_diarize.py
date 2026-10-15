import unittest

import numpy as np

from slc.diarize import clean_intervals, clean_turns, link_speakers, mark_overlaps


class TestDiarizeLogic(unittest.TestCase):
    def test_clean_turns_merges_and_drops(self):
        raw = [{"start": 0.0, "end": 1.0, "speaker": 0}, {"start": 1.3, "end": 2.0, "speaker": 0},
               {"start": 2.0, "end": 2.1, "speaker": 1},  # sliver
               {"start": 3.0, "end": 4.0, "speaker": 0}]
        out = clean_turns(raw)
        self.assertEqual([(t["start"], t["end"], t["speaker"]) for t in out],
                         [(0.0, 2.0, 0), (3.0, 4.0, 0)])

    def test_overlaps_and_clean_intervals(self):
        turns = [{"id": 0, "start": 0.0, "end": 10.0, "cluster": "A"},
                 {"id": 1, "start": 4.0, "end": 6.0, "cluster": "B"},
                 {"id": 2, "start": 9.95, "end": 12.0, "cluster": "B"},  # 50 ms: not overlap
                 {"id": 3, "start": 11.0, "end": 13.0, "cluster": "B"}]  # same speaker
        mark_overlaps(turns)
        self.assertEqual(turns[0]["overlaps_with"], [1])
        self.assertTrue(turns[1]["overlap"])
        self.assertFalse(turns[2]["overlap"] or turns[3]["overlap"])
        by_id = {t["id"]: t for t in turns}
        self.assertEqual(clean_intervals(turns[0], by_id), [(0.0, 4.0), (6.0, 10.0)])
        self.assertEqual(clean_intervals(turns[1], by_id), [])

    def test_link_speakers_cannot_link(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((3, 64))

        def v(k):
            x = base[k] + 0.3 * rng.standard_normal(64)
            return x / np.linalg.norm(x)

        # blocks: 0 has speakers p,q ; 1 has p,q ; 2 has p,r
        keys = ["b0_s0", "b0_s1", "b1_s0", "b1_s1", "b2_s0", "b2_s1"]
        who = [0, 1, 1, 0, 0, 2]
        blocks = [0, 0, 1, 1, 2, 2]
        lab = link_speakers(keys, np.array([v(k) for k in who]), blocks, threshold=0.5)
        groups = {}
        for k, w in zip(keys, who):
            groups.setdefault(w, set()).add(lab[k])
        self.assertTrue(all(len(g) == 1 for g in groups.values()))  # each person one label
        self.assertEqual(len(set(lab.values())), 3)
        # Two near-identical speakers in the same block must stay apart.
        same = np.array([v(0), v(0)])
        lab2 = link_speakers(["b0_s0", "b0_s1"], same, [0, 0], threshold=0.5)
        self.assertNotEqual(lab2["b0_s0"], lab2["b0_s1"])


if __name__ == "__main__":
    unittest.main()
