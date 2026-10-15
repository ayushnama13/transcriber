import os
import pathlib
import tempfile
import unittest

import numpy as np
import soundfile as sf

from slc.asr import (join_words, _pieces, assign_words, collapse_repeats, is_junk_segment, pick_channel,
                     speaker_channels, whisper_turns, words_to_lines)


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


def _w(start, end, word):
    return {"start": start, "end": end, "word": word, "prob": 0.9}


class TestWhisperWords(unittest.TestCase):
    def test_join_words(self):
        self.assertEqual(join_words(["costs", "6", ",500", "rupees."]), "costs 6,500 rupees.")
        self.assertEqual(join_words(["full", "-fledgedly,", "good"]), "full-fledgedly, good")

    TURNS = [
        {"id": 0, "start": 0.0, "end": 5.0, "cluster": "A"},
        {"id": 1, "start": 3.0, "end": 4.0, "cluster": "B", "overlap": True},  # short interjection inside A
        {"id": 2, "start": 6.0, "end": 9.0, "cluster": "B"},
    ]

    def test_crosstalk_words_go_to_one_speaker(self):
        words = [_w(0.5, 0.9, "So"), _w(2.8, 3.2, "the"), _w(3.3, 3.6, "budget"), _w(6.2, 6.6, "Yes"),
                 _w(5.4, 5.6, "um")]  # gap word -> nearest turn
        self.assertEqual(assign_words(words, self.TURNS), [0, 0, 0, 2, 0])
        rows = whisper_turns(words, self.TURNS)
        self.assertEqual(rows[1]["text"], "")  # no duplicate line for the interjection
        self.assertEqual(rows[0]["text"], "So the budget um")
        # a word that starts a fresh run inside the overlap goes to the longest covering turn
        self.assertEqual(assign_words([_w(3.4, 3.6, "x")], self.TURNS), [0])

    def test_collapse_repeats(self):
        ws = [_w(i, i + 0.5, t) for i, t in enumerate("hush hush hush hush hush I am".split())]
        self.assertEqual([w["word"] for w in collapse_repeats(ws)], ["hush", "hush", "I", "am"])
        phrase = ("I am using the market " * 4 + "okay").split()
        ws = [_w(i, i + 0.5, t) for i, t in enumerate(phrase)]
        out = " ".join(w["word"] for w in collapse_repeats(ws))
        self.assertEqual(out, "I am using the market I am using the market okay")
        ws = [_w(0, 1, "No,"), _w(1, 2, "no."), _w(2, 3, "Fine")]
        self.assertEqual(len(collapse_repeats(ws)), 3)  # a natural double is kept

    def test_junk_and_lines(self):
        self.assertTrue(is_junk_segment(" Thank you.", -0.8, 0.1))
        self.assertFalse(is_junk_segment(" Thank you.", -0.2, 0.05))
        self.assertTrue(is_junk_segment("anything", -1.5, 0.0))
        self.assertTrue(is_junk_segment("I'm going to go", -0.9, 0.8))
        self.assertFalse(is_junk_segment("The budget is 49 lakhs", -0.3, 0.1))
        lines = words_to_lines([_w(0, 1, "a"), _w(1.1, 2, "b."), _w(4, 5, "c")])
        self.assertEqual([ln["text"] for ln in lines], ["a b.", "c"])


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
