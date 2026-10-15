import os
import unittest
import tempfile
import numpy as np
from slc.voicelib import (
    VoiceLibrary,
    identify,
    name_clusters,
    turn_confidence,
    learn_from_review,
    unknown_label,
)

class TestVoiceLibrary(unittest.TestCase):
    def test_voicelib_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            D = 192
            np.random.seed(42)

            # 1. 60 random person "true" directions
            true_dirs = np.random.randn(60, D)
            true_dirs /= np.linalg.norm(true_dirs, axis=1, keepdims=True)

            # Pick 8 enrolled persons
            enrolled_indices = list(range(8))
            enrolled_names = [f"Person_{i}" for i in enrolled_indices]

            lib = VoiceLibrary(tmpdir)

            # Enroll 8 persons: enrollment = true + noise(0.3)
            for idx, name in zip(enrolled_indices, enrolled_names):
                true_vec = true_dirs[idx]
                noise = np.random.randn(5, D) * (0.6 / np.sqrt(D))  # |noise| ~0.6
                enrol_vecs = true_vec + noise
                enrol_vecs /= np.linalg.norm(enrol_vecs, axis=1, keepdims=True)
                lib.add(name, enrol_vecs, role="member", source="test")

            self.assertEqual(lib.names(), sorted(enrolled_names))
            self.assertEqual(lib.dim, D)

            # 2. Clusters of turns = person + noise(0.5) for 8 enrolled persons and 2 non-enrolled (total 10)
            non_enrolled_indices = [8, 9]
            all_cluster_indices = enrolled_indices + non_enrolled_indices

            turns = []
            turn_embs = []
            turn_id = 0

            for idx in all_cluster_indices:
                true_vec = true_dirs[idx]
                cluster_label = f"SPEAKER_{idx:02d}"
                for t_idx in range(3):
                    noise = np.random.randn(D) * (1.0 / np.sqrt(D))  # turn cos ~0.7, like real ECAPA
                    t_emb = true_vec + noise
                    t_emb /= np.linalg.norm(t_emb)
                    turn_embs.append(t_emb)
                    turns.append({
                        "id": turn_id,
                        "start": t_idx * 3.0,
                        "end": (t_idx + 1) * 3.0,
                        "cluster": cluster_label,
                        "overlap": False
                    })
                    turn_id += 1

            turn_embs = np.array(turn_embs, dtype=np.float32)

            # 3. name_clusters test
            speakers = name_clusters(lib, turns, turn_embs, threshold=0.55, margin=0.05)

            # Assert all 8 enrolled clusters are named correctly
            for idx in enrolled_indices:
                cluster_label = f"SPEAKER_{idx:02d}"
                expected_name = f"Person_{idx}"
                self.assertIn(cluster_label, speakers)
                self.assertEqual(speakers[cluster_label]["name"], expected_name)
                self.assertEqual(speakers[cluster_label]["source"], "voicelib")

            # Assert the 2 others get "Unknown A" / "Unknown B"
            unknown_names = set()
            for idx in non_enrolled_indices:
                cluster_label = f"SPEAKER_{idx:02d}"
                self.assertIn(cluster_label, speakers)
                spk = speakers[cluster_label]
                self.assertEqual(spk["source"], "unknown")
                self.assertTrue(spk["name"].startswith("Unknown"))
                unknown_names.add(spk["name"])

            self.assertEqual(unknown_names, {"Unknown A", "Unknown B"})

            # Assert one-to-one holds (all assigned names are unique across clusters)
            assigned_names = [info["name"] for info in speakers.values()]
            self.assertEqual(len(assigned_names), len(set(assigned_names)))

            # 4. Turn confidence test
            confidences = turn_confidence(lib, turns, turn_embs, speakers)
            self.assertEqual(len(confidences), len(turns))
            for conf in confidences:
                self.assertGreaterEqual(conf, 0.0)
                self.assertLessEqual(conf, 1.0)

            # 5. Rename & Remove & Persistence tests
            lib.rename("Person_0", "Renamed_Person")
            self.assertIn("Renamed_Person", lib.names())
            self.assertNotIn("Person_0", lib.names())

            lib.remove("Person_1")
            self.assertNotIn("Person_1", lib.names())

            # Test reload across new VoiceLibrary instance
            lib2 = VoiceLibrary(tmpdir)
            self.assertIn("Renamed_Person", lib2.names())
            self.assertNotIn("Person_1", lib2.names())
            self.assertEqual(lib2.dim, D)
            c_renamed = lib2.centroid("Renamed_Person")
            self.assertEqual(c_renamed.shape, (D,))
            self.assertGreater(np.linalg.norm(c_renamed), 0.0)

            # 6. learn_from_review & vector capping test
            many_vecs = np.random.randn(25, D).astype(np.float32)
            many_vecs /= np.linalg.norm(many_vecs, axis=1, keepdims=True)
            
            learn_turns = []
            learn_embs = []
            final_names = []
            for k in range(25):
                learn_turns.append({"id": 100+k, "start": 0.0, "end": 3.0, "cluster": "SPEAKER_X", "overlap": False})
                learn_embs.append(many_vecs[k])
                final_names.append("Person_2")

            learn_embs = np.array(learn_embs, dtype=np.float32)
            learn_from_review(lib2, learn_turns, learn_embs, final_names, min_dur=2.0, max_per_person=10)

            p2_emb_path = os.path.join(tmpdir, "emb", "person_2.npy")
            p2_loaded = np.load(p2_emb_path)
            self.assertLessEqual(len(p2_loaded), 10)
            self.assertEqual(lib2.people["Person_2"]["n"], len(p2_loaded))

class TestVoiceLibraryEdges(unittest.TestCase):
    def test_unknown_labels_past_z(self):
        self.assertEqual(unknown_label(0), "Unknown A")
        self.assertEqual(unknown_label(25), "Unknown Z")
        self.assertEqual(unknown_label(26), "Unknown AA")
        self.assertEqual(unknown_label(27), "Unknown AB")

    def test_corrupt_library_raises_instead_of_wiping(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "library.json"), "w") as f:
                f.write("{not json")
            with self.assertRaises(ValueError):
                VoiceLibrary(d)

    def test_slug_collision_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            lib = VoiceLibrary(d)
            lib.add("A. Kumar", np.ones((1, 8), np.float32))
            with self.assertRaises(ValueError):
                lib.add("A Kumar", np.ones((1, 8), np.float32))

    def test_learn_skips_short_overlap_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            lib = VoiceLibrary(d)
            emb = np.eye(4, 8, dtype=np.float32)
            turns = [{"id": 0, "start": 0, "end": 1.0, "cluster": "S"},              # too short
                     {"id": 1, "start": 0, "end": 5.0, "cluster": "S", "overlap": True},
                     {"id": 2, "start": 0, "end": 5.0, "cluster": "S"},              # unknown
                     {"id": 3, "start": 0, "end": 5.0, "cluster": "S"}]              # kept
            learn_from_review(lib, turns, emb, ["Asha", "Asha", "Unknown A", "Asha"])
            self.assertEqual(lib.people["Asha"]["n"], 1)
            self.assertEqual(lib.people["Asha"]["role"], "invitee")


if __name__ == "__main__":
    unittest.main()
