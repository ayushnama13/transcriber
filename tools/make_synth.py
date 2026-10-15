import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import numpy as np
import scipy.signal
import soundfile as sf
from slc.audio import to_wav16k, duration_s, read_segment

def main():
    parser = argparse.ArgumentParser(description="Make synthetic multi-recorder meeting audio")
    parser.add_argument("--speech", nargs="+", required=True, help="Input speech wav files")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--recorders", type=int, default=4, help="Number of recorders")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # 1. Load and convert speech clips to 16k mono
    clip_paths = []
    for idx, src in enumerate(args.speech):
        dst = os.path.join(args.out, f"clip_{idx}.wav")
        to_wav16k(src, dst)
        clip_paths.append(dst)

    clips = []
    for p in clip_paths:
        dur = duration_s(p)
        audio = read_segment(p, 0, dur)
        clips.append(audio)

    # 2. Build ~3 min meeting timeline
    target_dur = 180.0
    current_t = 1.0 # start after 1s gap
    turns = []
    overlaps = []

    # Insert an overlap region near 30s
    overlap_inserted = False
    clip_idx = 0

    while current_t < target_dur or not overlap_inserted:
        c_audio = clips[clip_idx % len(clips)]
        c_dur = len(c_audio) / 16000.0
        start = current_t
        end = start + c_dur
        speaker = f"SPEAKER_{clip_idx % len(clips):02d}"

        turns.append({"speaker": speaker, "start": float(start), "end": float(end), "audio": c_audio})

        # Check if we should insert overlap around 30s
        if not overlap_inserted and start >= 25.0:
            ov_start = start
            ov_dur = 4.0
            ov_end = ov_start + ov_dur
            other_clip = clips[(clip_idx + 1) % len(clips)]
            # truncate or use other_clip for 4s
            other_audio = other_clip[:int(ov_dur * 16000)]
            other_speaker = f"SPEAKER_{(clip_idx + 1) % len(clips):02d}"
            turns.append({"speaker": other_speaker, "start": float(ov_start), "end": float(ov_end), "audio": other_audio})
            overlaps.append([float(ov_start), float(min(end, ov_end))])
            overlap_inserted = True

        gap = np.random.uniform(0.5, 2.0)
        current_t = end + gap
        clip_idx += 1

    total_dur = max(t["end"] for t in turns) + 2.0
    total_samples = round(total_dur * 16000)

    # 3. Speaker positions and recorder positions
    n_rec = args.recorders
    # Recorder positions in 10x10 room
    rec_pos = np.array([
        [2.0, 2.0],
        [8.0, 2.0],
        [2.0, 8.0],
        [8.0, 8.0],
        [5.0, 5.0],
        [5.0, 2.0],
        [2.0, 5.0],
        [8.0, 5.0]
    ][:n_rec])

    # Assign each speaker a position near a distinct recorder
    unique_speakers = sorted(list(set(t["speaker"] for t in turns)))
    speaker_pos = {}
    for idx, spk in enumerate(unique_speakers):
        rec_idx = idx % n_rec
        # place speaker near rec_idx with some offset
        offset = np.random.uniform(-0.5, 0.5, size=2)
        speaker_pos[spk] = rec_pos[rec_idx] + offset

    # Compute gains: gain = 1 / (1 + distance)
    gains = np.zeros((n_rec, len(unique_speakers)))
    spk_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
    for j in range(n_rec):
        for spk, pos in speaker_pos.items():
            dist = np.linalg.norm(rec_pos[j] - pos)
            gains[j, spk_to_idx[spk]] = 1.0 / (1.0 + dist)

    # 4. Per recorder generation
    offsets_s = []
    drift_ppm = []

    noise_rms = 10.0 ** (-45.0 / 20.0)

    for j in range(n_rec):
        if j == 0:
            d_ppm = 0.0
            t_start = 0.0
        else:
            d_ppm = float(np.random.uniform(-150, 150))
            t_start = float(np.random.uniform(0, 20))

        # Recorder time tau <-> reference time t = (tau + t_start) * factor, so in
        # DESIGN.md convention t_i = -t_start + t / factor.
        offset_val = -t_start
        drift_ppm.append((1.0 / (1.0 + d_ppm * 1e-6) - 1.0) * 1e6)
        offsets_s.append(offset_val)

        # Build clean mix for recorder j
        mix = np.zeros(total_samples, dtype=np.float32)
        for t_item in turns:
            spk = t_item["speaker"]
            s_idx = spk_to_idx[spk]
            gain = gains[j, s_idx]
            s_samp = round(t_item["start"] * 16000)
            aud = t_item["audio"]
            end_samp = s_samp + len(aud)
            if end_samp > total_samples:
                aud = aud[:total_samples - s_samp]
                end_samp = total_samples
            mix[s_samp:end_samp] += aud * gain

        # Apply clock drift via resampling
        factor = 1.0 + d_ppm * 1e-6
        # Resample mix
        new_length = round(len(mix) / factor)
        resampled_mix = scipy.signal.resample(mix, new_length)

        # Apply start offset (crop start by t_start seconds)
        start_crop_samples = round(t_start * 16000)
        if start_crop_samples < len(resampled_mix):
            rec_audio = resampled_mix[start_crop_samples:]
        else:
            rec_audio = np.zeros(0, dtype=np.float32)

        # Add white noise at -45 dBFS
        noise = np.random.normal(0, noise_rms, len(rec_audio)).astype(np.float32)
        rec_audio += noise

        # Write rec{j}.wav
        rec_wav_path = os.path.join(args.out, f"rec{j}.wav")
        sf.write(rec_wav_path, rec_audio, 16000, subtype='PCM_16')

    # Clean turns for truth.json (remove 'audio')
    clean_turns = [{"speaker": t["speaker"], "start": t["start"], "end": t["end"]} for t in turns]

    truth = {
        "offsets_s": offsets_s,
        "drift_ppm": drift_ppm,
        "turns": clean_turns,
        "overlaps": overlaps
    }

    truth_path = os.path.join(args.out, "truth.json")
    with open(truth_path, "w") as f:
        json.dump(truth, f, indent=2)

    print(f"Synthetic meeting generated in {args.out} with {n_rec} recorders.")

if __name__ == "__main__":
    main()
