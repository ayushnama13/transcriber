import os
import json
import numpy as np
import scipy.signal
import soundfile as sf
from slc.audio import duration_s, read_segment, iter_blocks

def compute_envelope(path, block_s=60):
    env_blocks = []
    hop = 160
    for start_s, block in iter_blocks(path, block_s=block_s):
        n_frames = len(block) // hop
        if n_frames == 0:
            continue
        trimmed = block[:n_frames * hop].reshape(n_frames, hop)
        energy = np.mean(trimmed**2, axis=1)
        log_e = np.log(np.maximum(energy, 1e-10))
        env_blocks.append(log_e)
    if not env_blocks:
        return np.zeros(0, dtype=np.float32)
    env = np.concatenate(env_blocks)
    env -= np.mean(env)
    return env

def align_recordings(wav_paths, out_dir, reference=0):
    os.makedirs(out_dir, exist_ok=True)
    aligned_dir = os.path.join(out_dir, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    ref_path = wav_paths[reference]
    ref_dur = duration_s(ref_path)
    ref_env = compute_envelope(ref_path)

    recorders_info = []

    for i, path in enumerate(wav_paths):
        if i == reference:
            offset_s = 0.0
            drift_ppm = 0.0
            quality = 1.0
            n_windows_used = 0
        else:
            rec_env = compute_envelope(path)
            if len(ref_env) == 0 or len(rec_env) == 0:
                coarse_offset_s = 0.0
            else:
                corr = scipy.signal.correlate(rec_env, ref_env, mode='full')
                lags = scipy.signal.correlation_lags(len(rec_env), len(ref_env), mode='full')
                lag_frames = lags[np.argmax(corr)]
                coarse_offset_s = lag_frames * 0.01

            # Fine + drift windows: every 30 s
            win_s = min(20.0, max(2.0, ref_dur / 4))
            span = max(0.0, ref_dur - win_s)
            step = min(30.0, span / 10) if span > 0 else 1.0
            w_list = np.arange(0, span + 1e-9, step)
            t_vals = []
            t_i_vals = []
            norm_peaks = []

            for w in w_list:
                ref_win = read_segment(ref_path, w, w + win_s)
                if len(ref_win) == 0:
                    continue
                rms = np.sqrt(np.mean(ref_win**2))
                if rms < 0.005: # speech energy threshold
                    continue

                pred_start = w + coarse_offset_s
                rec_win = read_segment(path, pred_start - 1.0, pred_start + win_s + 1.0)
                if len(rec_win) == 0:
                    continue

                N = len(ref_win)
                M = len(rec_win)
                L = 1 << (N + M - 1).bit_length()

                X = np.fft.fft(ref_win, L)
                Y = np.fft.fft(rec_win, L)
                R = Y * np.conj(X)
                denom = np.abs(R)
                denom[denom == 0] = 1e-15
                R /= denom
                cc = np.fft.ifft(R)
                cc_shifted = np.fft.fftshift(cc)
                lags = np.arange(-L//2, L//2)

                peak_idx = np.argmax(np.abs(cc_shifted))
                if 0 < peak_idx < len(cc_shifted) - 1:
                    y_m1 = np.abs(cc_shifted[peak_idx - 1])
                    y_0 = np.abs(cc_shifted[peak_idx])
                    y_p1 = np.abs(cc_shifted[peak_idx + 1])
                    denom_q = (y_m1 - 2.0 * y_0 + y_p1)
                    delta = 0.5 * (y_m1 - y_p1) / denom_q if abs(denom_q) > 1e-12 else 0.0
                else:
                    delta = 0.0

                lag = lags[peak_idx] + delta
                # Validity: PHAT peak must stand well above the correlation floor.
                mag = np.abs(cc_shifted)
                psr = mag[peak_idx] / (np.median(mag) + 1e-12)
                if psr < 5.0:
                    continue
                # Quality: Pearson correlation of the aligned windows.
                k = int(round(lags[peak_idx]))
                if k < 0 or k + N > M:
                    continue
                seg = rec_win[k:k + N]
                # Informational only: far-apart recorders correlate weakly even when
                # perfectly aligned, so this must not gate the window.
                norm_peak = float(np.corrcoef(ref_win, seg)[0, 1])

                # local recorder time corresponding to reference time w
                # rec_win starts at pred_start - 1.0
                # lag gives shift of ref_win relative to rec_win start
                local_t_i = (pred_start - 1.0) + lag / 16000.0

                t_vals.append(w)
                t_i_vals.append(local_t_i)
                norm_peaks.append(norm_peak)

            t_vals = np.array(t_vals)
            t_i_vals = np.array(t_i_vals)
            norm_peaks = np.array(norm_peaks)

            if len(t_vals) < 3:
                drift_ppm = 0.0
                if len(t_vals) > 0:
                    offset_s = float(np.median(t_i_vals - t_vals))
                else:
                    offset_s = coarse_offset_s
                quality = float(np.median(norm_peaks)) if len(norm_peaks) > 0 else 0.5
                n_windows_used = len(t_vals)
            else:
                b, a = np.polyfit(t_vals, t_i_vals, 1)
                residuals = np.abs(t_i_vals - (a + b * t_vals))
                mad = np.median(np.abs(residuals - np.median(residuals)))
                thresh = max(3 * mad, 0.005)
                mask = residuals <= thresh
                if np.sum(mask) >= 3:
                    b, a = np.polyfit(t_vals[mask], t_i_vals[mask], 1)
                    inliers = mask
                else:
                    inliers = np.ones_like(t_vals, dtype=bool)

                offset_s = float(a)
                drift_ppm = float((b - 1.0) * 1e6)
                quality = float(np.median(norm_peaks[inliers]))
                n_windows_used = int(np.sum(inliers))

        recorders_info.append({
            "index": i,
            "path": path,
            "offset_s": float(offset_s),
            "drift_ppm": float(drift_ppm),
            "quality": float(quality),
            "n_windows_used": int(n_windows_used)
        })

    # Resample to reference timeline blockwise (60 s output blocks)
    ref_total_samples = round(ref_dur * 16000)
    block_s = 60
    block_samples = block_s * 16000

    for i, rec_info in enumerate(recorders_info):
        rec_path = wav_paths[i]
        out_wav_path = os.path.join(aligned_dir, f"rec{i}.wav")
        offset_s = rec_info["offset_s"]
        drift_ppm = rec_info["drift_ppm"]

        with sf.SoundFile(out_wav_path, mode='w', samplerate=16000, channels=1, subtype='PCM_16') as sf_out:
            for start_sample in range(0, ref_total_samples, block_samples):
                end_sample = min(ref_total_samples, start_sample + block_samples)
                n = np.arange(start_sample, end_sample, dtype=np.float64)
                if len(n) == 0:
                    continue

                src_samples = (offset_s + (n / 16000.0) * (1.0 + drift_ppm * 1e-6)) * 16000.0
                min_src = src_samples[0]
                max_src = src_samples[-1]

                read_start = max(0, int(np.floor(min_src)) - 100)
                rec_info_info = sf.info(rec_path)
                read_end = min(rec_info_info.frames, int(np.ceil(max_src)) + 100)

                src_block = read_segment(rec_path, read_start / 16000.0, read_end / 16000.0)
                xp = np.arange(read_start, read_start + len(src_block), dtype=np.float64)

                if len(xp) == 0:
                    interp_block = np.zeros_like(n)
                else:
                    interp_block = np.interp(src_samples, xp, src_block, left=0.0, right=0.0)

                pcm_block = np.clip(interp_block, -1.0, 1.0) * 32767.0
                sf_out.write(pcm_block.astype(np.int16))

    result = {
        "reference": reference,
        "sample_rate": 16000,
        "duration_s": float(ref_dur),
        "recorders": recorders_info
    }

    alignment_json_path = os.path.join(out_dir, "alignment.json")
    with open(alignment_json_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
