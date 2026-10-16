# Transcriber

SLC meeting pipeline: `python -m slc run AUDIO --out WORKDIR --terms terms.txt` (Whisper large-v3-turbo on GPU by default, `--model medium` for Moonshine). See DESIGN.md.

Legacy single-file tool:

CLI tool to transcribe long English audio files using Moonshine Voice with smart chunking and silence-based cutting.

## Usage

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Run transcription:
   ```bash
   python transcribe.py path/to/audio.mp3
   ```

### Options
- `-o OUTDIR`: Custom output directory (default: `./out/<stem>/`)
- `--chunk-min N`: Target chunk duration in minutes (default: `5`)
- `--model {medium,base,tiny}`: Model architecture (default: `medium`, most accurate)

### Outputs
Produces 4 files in the output directory:
- `<stem>.txt`: Clean plaintext transcript (paragraphs separated by blank lines).
- `<stem>.srt`: Subtitle file with precise timestamps.
- `<stem>.md`: Markdown transcript with paragraph timestamps.
- `<stem>.json`: Full segment JSON data with start, end, text, and chunk number.

## Team review with offline kits (coordinator)

Each checker gets one self-contained HTML page (their ~4-min pieces with the audio inside, ~1.3 MB per piece) plus `docs/reviewer-guide.pdf`. They review in any browser with no install or login, then send back an `edits_*.json` file.

```bash
# 1. Make kits (one workdir = whole meeting, or several clip workdirs in time order)
python -m slc kit work/2026-08-22-turbo/clip_* --out kits/2026-08-22 \
    --people "Asha,Ravi,Meena,..." --notes recordings/notes_2026-08-22.txt
#    or --per 5 for "Reviewer 01..NN"; fully reviewed pieces are skipped (--include-reviewed to keep them)

# 2. WhatsApp each person their kits/2026-08-22/review_<name>.html + reviewer-guide.pdf.
#    Keep kits/2026-08-22/manifest.json: it is how returned edits are matched back.

# 3. Put the returned files anywhere and merge (safe to re-run; newest edit per line wins)
python -m slc import-edits kits/2026-08-22 ~/Downloads/edits_*.json

# Progress only:
python -m slc import-edits kits/2026-08-22
```

Edits go through the same path as the review web app (`transcript.json` + `edits.jsonl`) and
`out/transcript.*` is re-exported. An edits file can only change lines that were in that
person's kit. `kits/` holds meeting audio and is gitignored: never commit or share it.
