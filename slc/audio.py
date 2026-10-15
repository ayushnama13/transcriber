import subprocess
import numpy as np
import soundfile as sf

def to_wav16k(src, dst):
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{res.stderr}")

def duration_s(path):
    info = sf.info(path)
    return info.duration

def read_segment(path, start_s, end_s):
    fs = 16000
    start_sample = round(start_s * fs)
    end_sample = round(end_s * fs)
    n_samples = max(0, end_sample - start_sample)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float32)

    info = sf.info(path)
    total_frames = info.frames

    file_start = max(0, start_sample)
    file_end = min(total_frames, end_sample)
    n_read = max(0, file_end - file_start)

    data = np.zeros(n_samples, dtype=np.float32)
    if n_read > 0:
        raw, _ = sf.read(path, start=file_start, frames=n_read, dtype='float32', always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        dest_start = file_start - start_sample
        data[dest_start:dest_start + len(raw)] = raw

    return data

def iter_blocks(path, block_s=60):
    dur = duration_s(path)
    start_s = 0.0
    while start_s < dur:
        end_s = min(dur, start_s + block_s)
        block = read_segment(path, start_s, end_s)
        yield start_s, block
        start_s += block_s

def level_wav(src, dst):
    """Even out loudness (ffmpeg dynaudnorm, streamed) so far-from-mic speakers are not
    drowned by near ones; gain is capped at ~24 dB so room noise is not blown up."""
    cmd = ["ffmpeg", "-y", "-i", str(src), "-af", "dynaudnorm=f=500:g=31:m=15:p=0.9",
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{res.stderr}")
