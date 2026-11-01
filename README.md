# Transcriber & SLC Meeting Pipeline

A robust, local-first speech transcription and speaker diarization pipeline designed for multi-recorder council meetings, featuring automatic time alignment, speaker identification via voice libraries, spatial overlap separation, and offline verification kits. Also includes a legacy single-file audio transcriber.

---

## Overview / Purpose

Transcribing multi-hour meetings recorded across multiple unsynchronized audio recorders presents significant challenges regarding clock drift, overlapping speech (crosstalk), speaker identification among dozens of members, and collaborative human verification. 

This project solves these challenges by providing:
1. **Multi-recorder synchronization**: Automatically aligns timebases and corrects clock drift across up to 4 independent recorders.
2. **Advanced Diarization & Spatial Separation**: Combines pyannote diarization and WeSpeaker embeddings with channel-aware acoustic selection to cleanly separate overlapping speech.
3. **Voice Library Matching**: Identifies permanent members and invitees using cosine similarity, centroid scoring, and Hungarian matching against an enrolled voice library.
4. **Offline & Online Verification**: Supports collaborative human review either via a local Flask web application or self-contained HTML review kits with embedded audio that checkers can use completely offline.
5. **Flexible Exports**: Outputs clean plaintext, markdown with timestamps, SRT subtitles, and formatted Microsoft Word (`.docx`) documents.

---

## Features

- **Multi-Recorder Time Alignment**: Aligns multiple audio streams to a reference timeline using cross-correlation for offset and linear clock drift.
- **Robust Speaker Diarization**: Uses a C++ port (`slc-diarize`) of pyannote community-1 segmentation and WeSpeaker embeddings with block-based processing and agglomerative cross-block speaker clustering.
- **Spatial Overlap Handling**: Detects crosstalk and routes transcription for each overlapping speaker to the recorder microphone closest to that individual.
- **High-Performance ASR**: Supports Whisper large-v3-turbo (`faster-whisper` int8 on GPU with hotword biasing) and Moonshine streaming models.
- **Directory-Backed Voice Library**: Enrolls speaker voice samples (`library.json` and `.npy` embeddings), performs similarity scoring, and learns from reviewer-confirmed lines.
- **Handnotes Calibration**: Parses handwritten meeting notes, matches entries against transcript segments, computes calibration metrics, and harvests new enrollment clips.
- **Offline Review Kits**: Generates self-contained HTML review pages with embedded 32 kbps MP3 audio blocks and reviewer guides. Returned edits (`edits_*.json`) safely merge back with conflict resolution.
- **Local Review Web App**: Interactive Flask web application for verifiers to correct transcript text, speaker labels, and review notes.
- **Multi-Format Export**: Generates `.txt`, `.srt`, `.md`, and `.docx` transcript exports.
- **Legacy Single-File Transcriber**: Command-line tool (`transcribe.py`) for transcribing long single-channel audio files using Moonshine with silence-based chunking.

---

## How It Works / Architecture

The pipeline processes meeting audio through sequential, cacheable stages where each step stores its output in a work directory (`work/<meeting>/`). If interrupted, execution resumes from the last completed stage.

```
 [ Input Recorders ] 
        │
        ▼ (ffmpeg 16kHz conversion)
 [ norm/rec{i}.wav ] ──► [ alignment.json ] ──► [ Aligned Recorders ]
                                                       │
                                                       ▼
 [ Notes / Library ] ──► [ Diarization & Mix ] ──► [ turns.json & embeddings.npy ]
                                                       │
                                                       ▼
                       [ ASR & Speaker Naming ] ──► [ asr.json & speakers.json ]
                                                       │
                                                       ▼
                       [ Segment Assembly ] ──────► [ transcript.json ]
                                                       │
                                                       ▼
                                             [ Export (TXT/MD/SRT/DOCX) ]
```

---

## Project Structure

```
transcriber/
├── transcribe.py                # Legacy single-file long audio transcriber using Moonshine
├── DESIGN.md                    # Detailed pipeline design notes and specifications
├── requirements.freeze.txt      # Pinned dependency versions
├── slc/                         # Core meeting pipeline package
│   ├── __init__.py              # Package marker
│   ├── __main__.py              # CLI entry point (python -m slc)
│   ├── pipeline.py              # End-to-end meeting orchestration
│   ├── audio.py                 # Audio conversion, chunking, reading, and dynaudnorm leveling
│   ├── align.py                 # Multi-recorder time alignment and clock drift correction
│   ├── diarize.py               # Speaker diarization, mix generation, and WeSpeaker embeddings
│   ├── asr.py                   # Whisper / Moonshine ASR integration, word assignment, and filtering
│   ├── voicelib.py              # Directory-backed voice library and similarity matching
│   ├── notes.py                 # Handwritten notes parser, calibration, and clip harvesting
│   ├── enroll.py                # Audio enrollment and transcript learning utilities
│   ├── review.py                # Flask web application for interactive verification
│   ├── kit.py                   # Offline review kit generator and edit merger
│   ├── export.py                # Exporters for TXT, Markdown, SRT, and DOCX formats
│   └── kit_template.html        # HTML template for offline reviewer kits
├── tools/                       # Build scripts and utilities
│   ├── build_diarizer.sh        # Compiles the C++ diarizer binary
│   ├── make_synth.py            # Synthetic audio generator for testing
│   └── cpp/
│       └── slc_diarize.cpp      # C++ port of pyannote segmentation inference
└── docs/                        # Documentation assets
    ├── reviewer-guide.html      # Reviewer instructions (HTML)
    └── reviewer-guide.pdf       # Reviewer instructions (PDF)
```

---

## Tech Stack

- **Language**: Python 3.10+
- **Speech & Audio**: `faster-whisper`, `moonshine-voice`, `torchaudio`, `soundfile`, `av`, `julius`, `asteroid-filterbanks`
- **Machine Learning & Inference**: PyTorch, ONNX Runtime (`onnxruntime-gpu`), CTranslate2, `pyannote-audio`
- **Web & API**: Flask, Werkzeug, Jinja2
- **Document Processing**: `python-docx`, `lxml`
- **System Utilities**: FFmpeg / ffprobe

---

## Requirements

- **Python**: Version 3.10 or higher.
- **System Tools**: `ffmpeg` and `ffprobe` must be installed and available on system `PATH`.
- **Hardware**: An NVIDIA GPU (e.g., GTX 1050 4 GB or higher with CUDA support) is strongly recommended for Whisper large-v3-turbo and diarization models. CPU execution is supported via `--cpu`.
- **Dependencies**: All required Python packages are listed in `requirements.freeze.txt`.

---

## Installation / Setup

1. **Clone the repository and navigate to the project directory**:
   ```bash
   cd transcriber
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.freeze.txt
   ```

4. **Build the C++ diarizer binary** (required for meeting diarization):
   ```bash
   bash tools/build_diarizer.sh
   ```

---

## Configuration

The pipeline supports the following environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SLC_ORT_CUDA` | Set to `1` by the pipeline itself when it launches the diarizer on GPU (`slc/diarize.py`); you normally do not set it. | Set automatically |
| `LD_LIBRARY_PATH` | Path used to load native CUDA libraries (such as `libcublas` and `libcudnn`) for ONNX Runtime. | System default |

---

## Usage

### 1. Legacy Single-File Transcriber (`transcribe.py`)
Transcribes a single long audio file using Moonshine with silence-based chunking.

```bash
python transcribe.py recordings/Voice.m4a --chunk-min 5 --model medium -o out/meeting
```

**Options**:
- `-o OUTDIR`: Custom output directory (default: `./out/<stem>/`)
- `--chunk-min N`: Target chunk duration in minutes (default: `5`)
- `--model {medium,base,tiny}`: Moonshine model size (default: `medium`)

### 2. Full SLC Meeting Pipeline (`python -m slc`)

#### Run Whole Pipeline (Align -> Diarize -> Name -> ASR -> Export)
```bash
python -m slc run recordings/rec0.wav recordings/rec1.wav --out work/meeting --library voice_lib --terms terms.txt --name "Council Meeting"
```
*(Options: `--model {turbo,medium,base,tiny}`, `--part-min N`, `--cpu`, `--force` (overwrite a transcript that already has reviewed lines), `--reference INDEX`)*

#### Step-by-Step Pipeline Commands
- **Align recorders**:
  ```bash
  python -m slc align recordings/rec0.wav recordings/rec1.wav --out work/meeting
  ```
- **Diarize**:
  ```bash
  python -m slc diarize work/meeting --block-min 30
  ```
- **Voice Library Management**:
  ```bash
  # List enrolled people
  python -m slc library --library voice_lib
  
  # Enroll voice samples for a person
  python -m slc enroll clip1.wav clip2.wav --library voice_lib --name "Rahul Sharma" --role member
  ```
- **Calibrate against notes**:
  ```bash
  python -m slc calibrate work/meeting --notes recordings/notes.txt --aliases aliases.json
  ```
- **Interactive Review (Web App)**:
  ```bash
  python -m slc review work/meeting --library voice_lib --port 8000
  ```
- **Generate Offline Review Kits**:
  ```bash
  python -m slc kit work/meeting --out kits/meeting --people "Asha,Ravi,Meena" --notes recordings/notes.txt
  ```
- **Import Reviewer Edits**:
  ```bash
  python -m slc import-edits kits/meeting Downloads/edits_*.json
  ```
- **Feed reviewed lines back into the voice library**:
  ```bash
  python -m slc learn work/meeting --library voice_lib
  ```
- **Re-export Transcript**:
  ```bash
  python -m slc export work/meeting --out work/meeting/out
  ```

---

## Data Model / Database Schema

Meeting work directories (`work/<meeting>/`) maintain a file-based schema:
- `inputs.json`: Records input file paths and reference recorder index.
- `alignment.json`: Recorder offsets (seconds), drift (ppm), and alignment quality scores.
- `turns.json`: Diarization turns (`id`, `start`, `end`, `cluster`, `overlap`, `overlaps_with`, `channel_rms_db`, `best_channel`).
- `embeddings.npy`: Float32 array of shape `[n_turns, 256]` for speaker identification.
- `speakers.json`: Cluster label mappings to speaker names, confidence scores, and source types.
- `asr.json`: Per-turn recognized text and word-level timestamp arrays (`start`, `end`, `word`, `prob`).
- `transcript.json`: Final merged meeting segments (`id`, `turn_id`, `start`, `end`, `speaker`, `speaker_confidence`, `cluster`, `text`, `channel`, `flags`, `overlaps_with`, `reviewed`, `part`).
- `edits.jsonl`: Append-only log of reviewer modifications.

---

## Testing

Tests live in `tests/` (`test_align.py`, `test_asr.py`, `test_diarize.py`, `test_kit.py`, `test_notes.py`, `test_review.py`, `test_voicelib.py`). Run them with `python -m pytest tests` inside the virtual environment (pytest must be installed).

---

## Troubleshooting / FAQ & Known Limitations

- **FFmpeg Not Found**: Ensure `ffmpeg` and `ffprobe` are installed and available in your system `PATH`.
- **CUDA Errors**: Ensure compatible NVIDIA driver and CUDA toolkit libraries are installed when running on GPU. If CUDA libraries fail to load, use `--cpu` to run on CPU.
- **Memory Usage**: The pipeline processes meeting recordings in blockwise chunks to prevent high RAM consumption on long audio files.

---

## Notes on Safety, Rate Limits & Privacy

- **Local Execution**: All transcription, alignment, diarization, voice identification, and review tasks execute entirely locally on your machine. No cloud API keys, external accounts, or internet connections are required.
- **Data Privacy**: Meeting recordings, work directories (`work/`), and offline review kits (`kits/`) contain sensitive audio data and should be kept secure and excluded from public version control (gitignored).
