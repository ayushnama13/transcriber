# SLC meeting pipeline — design

Goal: word-for-word, speaker-named transcript of 3–6 h SLC council meetings
recorded on 3–4 unsynchronised recorders. ~50 permanent members (enrolled
voice library) + 10–20 invitees per meeting. Overlapping speech is flagged and,
where possible, transcribed per speaker. 7–10 humans verify in parts.

Hardware target: GTX 1050 4 GB (CUDA, Pascal sm_61), 8 cores, 14 GB RAM.
**Never hold a full 6 h recording as float arrays in RAM** (6 h @16 kHz =
1.4 GB float32 per recorder). Work on 16 kHz mono WAVs on disk and read blocks
with `soundfile`.

## Package layout (`slc/`)

| module | role |
|---|---|
| `slc/audio.py` | ffmpeg → 16 kHz mono PCM16 WAV; block readers; `read_segment(path, start_s, end_s) -> np.float32` |
| `slc/align.py` | align N recorders to a reference timeline (offset + linear clock drift) |
| `slc/diarize.py` | pyannote diarization (with overlap) on the reference mix → turns + embeddings |
| `slc/voicelib.py` | voice library: enroll, identify (cosine), cluster unknowns |
| `slc/asr.py` | Moonshine ASR backend, keyterms; transcribe one turn |
| `slc/notes.py` | parse handwritten notes, match to transcript, calibration metrics, harvest enrollment clips |
| `slc/pipeline.py` | orchestration of one meeting |
| `slc/export.py` | md / docx / srt / txt |
| `slc/review/` | Flask review app for verifiers |
| `slc/__main__.py` | CLI: `python -m slc <cmd>` |

## Meeting work dir: `work/<meeting>/`

```
inputs.json            {"recordings": [abs paths], "reference": 0}
norm/rec{i}.wav        16 kHz mono PCM16 conversion of each input
aligned/rec{i}.wav     resampled onto the reference timeline; all same length
alignment.json         see below
turns.json             diarization output (see below)
embeddings.npy         float32 [n_turns, D]; row i <-> turns[i]
speakers.json          {cluster_label: {"name", "score", "source"}}
transcript.json        final segments (see below); edited in place by review app
edits.jsonl            append-only log of reviewer edits
```

### alignment.json
```json
{"reference": 0, "sample_rate": 16000, "duration_s": 12345.6,
 "recorders": [{"index": 0, "path": "...", "offset_s": 0.0, "drift_ppm": 0.0,
                "quality": 1.0, "n_windows_used": 0}]}
```
Model: a sample at reference time `t` is found in recorder *i* at its own time
`t_i = offset_s + t * (1 + drift_ppm * 1e-6)`. Reference has offset 0, drift 0.
`quality` = median normalised cross-correlation peak of the windows used (0–1).

### turns.json
```json
[{"id": 0, "start": 12.34, "end": 18.9, "cluster": "SPEAKER_03",
  "overlap": false, "overlaps_with": [],
  "channel_rms_db": [-23.1, -30.4, -35.0, -28.2], "best_channel": 0}]
```

### transcript.json
```json
{"meeting": "...", "segments": [
 {"id": 0, "start": 12.34, "end": 18.9, "speaker": "Rahul Sharma",
  "speaker_confidence": 0.82, "cluster": "SPEAKER_03",
  "text": "...", "channel": 0,
  "flags": ["crosstalk", "low_speaker_conf", "notes_mismatch"],
  "overlaps_with": [5], "reviewed": false, "part": 3}]}
```

## Overlap policy
1. Detect and flag every overlap (`crosstalk` flag, list of overlapping ids).
2. For each speaker in an overlap, transcribe from the recorder where that
   speaker's own (non-overlapped) speech is loudest — spatial separation.
3. Later/optional: neural separation on flagged regions only.
