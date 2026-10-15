import os
import unittest
import tempfile
import json
import subprocess
import sys
import soundfile as sf
from moonshine_voice import get_assets_path
from slc.align import align_recordings
from slc.audio import duration_s

class TestAlign(unittest.TestCase):
    def test_alignment(self):
        assets = get_assets_path()
        speech_files = [
            os.path.join(assets, "two_cities.wav"),
            os.path.join(assets, "beckett.wav"),
            os.path.join(assets, "endgame_nagg_nell.wav"),
            os.path.join(assets, "clone-test.wav")
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Generate synth
            cmd = [
                sys.executable,
                "tools/make_synth.py",
                "--speech"
            ] + speech_files + [
                "--out", tmpdir,
                "--recorders", "4",
                "--seed", "0"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"make_synth failed:\n{res.stderr}\n{res.stdout}")

            # Load truth.json
            with open(os.path.join(tmpdir, "truth.json")) as f:
                truth = json.load(f)

            # 2. Run align_recordings
            wav_paths = [os.path.join(tmpdir, f"rec{i}.wav") for i in range(4)]
            alignment = align_recordings(wav_paths, tmpdir, reference=0)

            # 3. Assertions
            ref_dur = alignment["duration_s"]
            ref_len_samples = round(ref_dur * 16000)

            for rec_info in alignment["recorders"]:
                i = rec_info["index"]
                # Check aligned file length
                aligned_wav = os.path.join(tmpdir, "aligned", f"rec{i}.wav")
                self.assertTrue(os.path.exists(aligned_wav))
                aligned_dur = duration_s(aligned_wav)
                aligned_len_samples = round(aligned_dur * 16000)
                self.assertEqual(aligned_len_samples, ref_len_samples)

                # Check offset error < 5 ms (0.005 s)
                est_offset = rec_info["offset_s"]
                true_offset = truth["offsets_s"][i]
                offset_error = abs(est_offset - true_offset)
                self.assertLess(
                    offset_error, 0.005,
                    f"Recorder {i}: offset error {offset_error:.4f}s exceeds 5ms (est={est_offset}, true={true_offset})"
                )

                # Check drift error < 20 ppm
                est_drift = rec_info["drift_ppm"]
                true_drift = truth["drift_ppm"][i]
                drift_error = abs(est_drift - true_drift)
                self.assertLess(
                    drift_error, 20.0,
                    f"Recorder {i}: drift error {drift_error:.2f}ppm exceeds 20ppm (est={est_drift}, true={true_drift})"
                )

if __name__ == "__main__":
    unittest.main()
