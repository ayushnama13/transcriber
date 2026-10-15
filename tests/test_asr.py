import os
import pathlib
import tempfile
import unittest

import numpy as np
import soundfile as sf

from slc.asr import _pieces, pick_channel, speaker_channels


def _asset(name):
    import moonshine_voice
    p = pathlib.Path(moonshine_voice.__file__).parent / "assets" / name
    return p if p.exists() else None


class TestChannels(unittest.TestCase):
    def test_speaker_channels_and_pick(self):
        turns = [
            {"cluster": "A", "channel_rms_db": [-20, -35], "best_channel": 0},
            {"cluster": "A", "channel_rms_db": [-22, -30], "best_channel": 0},
            {"cluster": "B", "channel_rms_db": [-40, -21], "best_channel": 1},
            # overlapped: B is loudest on mic 0 here, but A's own mic is 0 and B's is 1
            {"cluster": "B", "channel_rms_db": [-18, -25], "best_channel": 0, "overlap": True},
        ]
        spk = speaker_channels(turns)
        self.assertEqual(spk, {"A": 0, "B": 1})
        self.assertEqual(pick_channel(turns[3], spk), 1)
        self.assertEqual(pick_channel(turns[0], spk), 0)
        # overlapped turn of a speaker with no clean speech falls back to best_channel
        self.assertEqual(pick_channel({"cluster": "C", "overlap": True, "best_channel": 1}, spk), 1)

    def test_pieces(self):
        self.assertEqual(len(_pieces(0, 100)), 1)
        p = _pieces(0, 500)
        self.assertEqual(len(p), 3)
        self.assertAlmostEqual(p[-1][1], 500)


@unittest.skipUnless(_asset("endgame_nagg_nell.wav"), "sample audio missing")
class TestMoonshine(unittest.TestCase):
    def test_transcribe_turns_reads_chosen_channel(self):
        from slc.asr import load_transcriber, transcribe_turns
        audio, sr = sf.read(str(_asset("endgame_nagg_nell.wav")), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import scipy.signal
            audio = scipy.signal.resample_poly(audio, 16000, sr).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            sf.write(os.path.join(d, "rec0.wav"), audio, 16000, subtype="PCM_16")
            junk = 0.3 * rng.standard_normal(len(audio)).astype(np.float32)  # unusable mic
            sf.write(os.path.join(d, "rec1.wav"), np.clip(junk, -1, 1), 16000, subtype="PCM_16")
            dur = len(audio) / 16000
            turns = [
                {"id": 0, "start": 0.0, "end": dur / 2, "cluster": "A",
                 "channel_rms_db": [-20, -40], "best_channel": 0},
                {"id": 1, "start": dur / 2, "end": dur, "cluster": "A", "overlap": True,
                 "channel_rms_db": [-40, -20], "best_channel": 1},
            ]
            tr = load_transcriber("medium")
            out = transcribe_turns(tr, turns, d)
        print("\n".join(f"  [{o['channel']}] {o['text']}" for o in out))
        self.assertEqual([o["channel"] for o in out], [0, 0])
        self.assertTrue(all(len(o["text"].split()) >= 3 for o in out))


if __name__ == "__main__":
    unittest.main()
