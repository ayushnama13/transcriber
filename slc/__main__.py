"""python -m slc <command> ...   (see `python -m slc -h`)"""
import argparse
import json
import os
import sys


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def need_files(*paths):
    for p in paths:
        if p and not os.path.isfile(p):
            die(f"not found: {p}")


def cmd_align(args):
    from slc.pipeline import prepare_alignment
    need_files(*args.recordings)
    al = prepare_alignment(args.recordings, args.out, reference=args.reference)
    print(f"{'Recorder':<10} | {'Offset (s)':<12} | {'Drift (ppm)':<12} | {'Quality':<8}")
    print("-" * 52)
    for rec in al["recorders"]:
        print(f"{rec['index']:<10} | {rec['offset_s']:<12.3f} | {rec['drift_ppm']:<12.2f} | {rec['quality']:<8.3f}")


def cmd_diarize(args):
    need_files(os.path.join(args.workdir, "alignment.json"))
    from slc.diarize import diarize_meeting
    diarize_meeting(args.workdir, block_s=args.block_min * 60, gpu=not args.cpu,
                    link_threshold=args.link_threshold)


def cmd_run(args):
    from slc.pipeline import run_meeting
    need_files(*args.recordings, args.terms)
    if not args.recordings and not os.path.isfile(os.path.join(args.out, "alignment.json")):
        die("give the recordings (or a work dir that is already aligned)")
    try:
        run_meeting(args.recordings, args.out, library=args.library, terms_file=args.terms,
                    model=args.model, part_min=args.part_min, gpu=not args.cpu,
                    reference=args.reference, force=args.force, meeting=args.name)
    except RuntimeError as e:
        die(str(e))


def cmd_calibrate(args):
    from slc.notes import apply_flags, calibrate, harvest_enrollment, match_notes, parse_notes

    tpath = os.path.join(args.workdir, "transcript.json")
    need_files(tpath, args.notes, args.aliases)
    with open(tpath) as f:
        transcript = json.load(f)
    aliases = None
    if args.aliases:
        with open(args.aliases) as f:
            aliases = json.load(f)
    notes = parse_notes(args.notes)
    if not notes:
        die("no 'Name: text' lines found in notes")
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


def cmd_enroll(args):
    from slc.enroll import enroll_files, enroll_from_calibration
    from slc.voicelib import VoiceLibrary
    lib = VoiceLibrary(args.library)
    if args.from_calibration:
        need_files(os.path.join(args.from_calibration, "calibration.json"))
        added = enroll_from_calibration(lib, args.from_calibration)
        for name, n in sorted(added.items()):
            print(f"  {name}: +{n} clips from notes")
        if not added:
            print("no harvestable clips in calibration.json")
        return
    if not args.name or not args.files:
        die("give --name and one or more audio files (or --from-calibration WORKDIR)")
    need_files(*args.files)
    try:
        n = enroll_files(lib, args.name, args.files, role=args.role)
    except ValueError as e:
        die(str(e))
    print(f"enrolled {args.name} ({args.role}): {n} voice samples; library has {len(lib.names())} people")


def cmd_learn(args):
    from slc.enroll import learn_from_transcript
    from slc.voicelib import VoiceLibrary
    need_files(os.path.join(args.workdir, "transcript.json"))
    added = learn_from_transcript(VoiceLibrary(args.library), args.workdir)
    for name, n in sorted(added.items()):
        print(f"  {name}: +{n}")
    print(f"{len(added)} people updated from reviewed lines")


def cmd_export(args):
    from slc.export import export_all
    tpath = os.path.join(args.workdir, "transcript.json")
    need_files(tpath)
    with open(tpath) as f:
        paths = export_all(json.load(f), args.out or os.path.join(args.workdir, "out"))
    for k, v in paths.items():
        print(f"{k}: {v}")


def cmd_review(args):
    for f in ("transcript.json", "aligned/mix.wav"):
        need_files(os.path.join(args.workdir, f))
    from slc.review import create_app
    app = create_app(args.workdir, library=args.library)
    print(f"review app: http://127.0.0.1:{args.port}/  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=args.port, threaded=True)  # localhost only: no login yet


def cmd_library(args):
    from slc.voicelib import VoiceLibrary
    lib = VoiceLibrary(args.library)
    if args.remove:
        lib.remove(args.remove)
    if args.rename:
        lib.rename(*args.rename)
    for n in lib.names():
        info = lib.people[n]
        print(f"{n:<30} {info.get('role', ''):<8} {info.get('n', 0):>4} samples  {','.join(info.get('sources', []))}")
    print(f"{len(lib.names())} people")


def main():
    ap = argparse.ArgumentParser(prog="python -m slc", description="SLC meeting transcriber")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("run", help="Whole pipeline: recordings -> named transcript")
    s.add_argument("recordings", nargs="*", help="All recorder files of one meeting")
    s.add_argument("--out", required=True, help="Meeting work dir")
    s.add_argument("--library", help="Voice library dir (enrolled voices)")
    s.add_argument("--terms", help="Text file: one name/acronym/term per line (ASR bias)")
    s.add_argument("--model", default="turbo", choices=["turbo", "medium", "base", "tiny"],
                   help="turbo = Whisper large-v3-turbo on GPU (default); medium/base/tiny = Moonshine on CPU")
    s.add_argument("--part-min", type=float, default=30, help="Review part length (minutes)")
    s.add_argument("--reference", type=int, default=0)
    s.add_argument("--name", help="Meeting title")
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--force", action="store_true", help="Overwrite a transcript with reviewed lines")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("align", help="Align recorders to a reference timeline")
    s.add_argument("recordings", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--reference", type=int, default=0)
    s.set_defaults(fn=cmd_align)

    s = sub.add_parser("diarize", help="Who spoke when (after align) -> turns.json")
    s.add_argument("workdir")
    s.add_argument("--block-min", type=float, default=30)
    s.add_argument("--link-threshold", type=float, default=0.5)
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(fn=cmd_diarize)

    s = sub.add_parser("calibrate", help="Score speaker labels against handwritten notes")
    s.add_argument("workdir")
    s.add_argument("--notes", required=True, help='One "Name: what they said" per line')
    s.add_argument("--aliases", help='JSON, e.g. {"Chair": "Rahul Sharma"}')
    s.add_argument("--min-score", type=float, default=70)
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("enroll", help="Add voice samples of a person to the library")
    s.add_argument("files", nargs="*", help="Audio clips of this one person speaking")
    s.add_argument("--library", required=True)
    s.add_argument("--name")
    s.add_argument("--role", default="member", choices=["member", "invitee"])
    s.add_argument("--from-calibration", metavar="WORKDIR", help="Use clips matched to notes")
    s.set_defaults(fn=cmd_enroll)

    s = sub.add_parser("learn", help="Feed reviewer-confirmed lines back into the library")
    s.add_argument("workdir")
    s.add_argument("--library", required=True)
    s.set_defaults(fn=cmd_learn)

    s = sub.add_parser("export", help="Re-export transcript.json (after review)")
    s.add_argument("workdir")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("review", help="Web app for checkers to correct text and names")
    s.add_argument("workdir")
    s.add_argument("--library", help="Voice library (adds its names to the speaker list)")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("library", help="List / edit the voice library")
    s.add_argument("--library", required=True)
    s.add_argument("--remove", metavar="NAME")
    s.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"))
    s.set_defaults(fn=cmd_library)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
