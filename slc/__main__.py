import os
import sys
import json
import argparse
from slc.audio import to_wav16k
from slc.align import align_recordings

def cmd_calibrate(args):
    from slc.notes import apply_flags, calibrate, harvest_enrollment, match_notes, parse_notes

    tpath = os.path.join(args.workdir, "transcript.json")
    for p in (tpath, args.notes, args.aliases):
        if p and not os.path.isfile(p):
            print(f"error: not found: {p}", file=sys.stderr)
            return 1
    with open(tpath) as f:
        transcript = json.load(f)
    aliases = None
    if args.aliases:
        with open(args.aliases) as f:
            aliases = json.load(f)
    notes = parse_notes(args.notes)
    if not notes:
        print("error: no 'Name: text' lines found in notes", file=sys.stderr)
        return 1
    segs = transcript["segments"]
    matches = match_notes(notes, segs, min_score=args.min_score)
    report = calibrate(notes, segs, matches, aliases)
    report["harvest"] = harvest_enrollment(matches, notes, segs)
    apply_flags(transcript, report, notes, matches)

    with open(os.path.join(args.workdir, "calibration.json"), "w") as f:
        json.dump(report, f, indent=2)
    tmp = tpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(transcript, f, indent=2)
    os.replace(tmp, tpath)

    print(f"notes: {report['n_notes']}  matched: {report['n_matched']} ({report['coverage']:.1%})")
    print(f"speaker accuracy on matched notes: {report['speaker_accuracy']:.1%}")
    for c in report["confusion"][:10]:
        print(f"  noted {c['note_name']!r} but labelled {c['predicted']!r}: {c['count']}x")
    print(f"enrollment clips harvestable: {len(report['harvest'])}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="SLC meeting pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    align_parser = subparsers.add_parser("align", help="Align multiple recorder audio files to a reference timeline")
    align_parser.add_argument("recordings", nargs="+", help="Input recorder audio files")
    align_parser.add_argument("--out", required=True, help="Working directory output path")
    align_parser.add_argument("--reference", type=int, default=0, help="Reference recorder index")

    cal_parser = subparsers.add_parser("calibrate", help="Score speaker labels against handwritten notes")
    cal_parser.add_argument("workdir", help="Meeting work dir containing transcript.json")
    cal_parser.add_argument("--notes", required=True, help='Notes file, one "Name: what they said" per line')
    cal_parser.add_argument("--aliases", help='JSON map, e.g. {"Chair": "Rahul Sharma"}')
    cal_parser.add_argument("--min-score", type=float, default=70)

    dia_parser = subparsers.add_parser("diarize", help="Who spoke when (after align) -> turns.json")
    dia_parser.add_argument("workdir", help="Meeting work dir containing alignment.json")
    dia_parser.add_argument("--block-min", type=float, default=30, help="Diarize in blocks of this many minutes")
    dia_parser.add_argument("--link-threshold", type=float, default=0.5,
                            help="Cosine needed to join speakers across blocks")
    dia_parser.add_argument("--cpu", action="store_true", help="Do not use the GPU")

    args = parser.parse_args()

    if args.command == "diarize":
        if not os.path.isfile(os.path.join(args.workdir, "alignment.json")):
            print(f"error: {args.workdir}/alignment.json not found; run align first", file=sys.stderr)
            sys.exit(1)
        from slc.diarize import diarize_meeting
        diarize_meeting(args.workdir, block_s=args.block_min * 60, gpu=not args.cpu,
                        link_threshold=args.link_threshold)
        sys.exit(0)

    if args.command == "calibrate":
        sys.exit(cmd_calibrate(args))

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
