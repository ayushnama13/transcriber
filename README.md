# Transcriber

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
