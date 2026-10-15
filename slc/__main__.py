import os
import sys
import json
import argparse
from slc.audio import to_wav16k
from slc.align import align_recordings

def main():
    parser = argparse.ArgumentParser(description="SLC meeting pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    align_parser = subparsers.add_parser("align", help="Align multiple recorder audio files to a reference timeline")
    align_parser.add_argument("recordings", nargs="+", help="Input recorder audio files")
    align_parser.add_argument("--out", required=True, help="Working directory output path")
    align_parser.add_argument("--reference", type=int, default=0, help="Reference recorder index")

    args = parser.parse_args()

    if args.command == "align":
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        norm_dir = os.path.join(out_dir, "norm")
        os.makedirs(norm_dir, exist_ok=True)

        norm_paths = []
        abs_inputs = [os.path.abspath(p) for p in args.recordings]

        for i, src in enumerate(args.recordings):
            dst = os.path.join(norm_dir, f"rec{i}.wav")
            to_wav16k(src, dst)
            norm_paths.append(dst)

        inputs_data = {
            "recordings": abs_inputs,
            "reference": args.reference
        }
        with open(os.path.join(out_dir, "inputs.json"), "w") as f:
            json.dump(inputs_data, f, indent=2)

        alignment = align_recordings(norm_paths, out_dir, reference=args.reference)

        print(f"{'Recorder':<10} | {'Offset (s)':<12} | {'Drift (ppm)':<12} | {'Quality':<8}")
        print("-" * 52)
        for rec in alignment["recorders"]:
            print(f"{rec['index']:<10} | {rec['offset_s']:<12.3f} | {rec['drift_ppm']:<12.2f} | {rec['quality']:<8.3f}")

if __name__ == "__main__":
    main()
