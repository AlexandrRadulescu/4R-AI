"""Spotted events -> a watchable highlight reel.

    python make_reel.py --video matches/liverpool_arsenal.mp4 \
        --spots spots/liverpool_arsenal.json --max-clips 12

Requires ffmpeg on PATH. Clips are cut with re-encoding rather than stream
copy: stream copy snaps to keyframes, which can shift a cut by several seconds
and cut the goal off.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from config import PRE_ROLL_S, POST_ROLL_S

# Higher = more likely to make the cut when the reel is length-limited.
SALIENCE = {"goal": 3.0, "card": 1.6, "shot": 1.0}


def rank(spots, max_clips):
    scored = sorted(
        spots,
        key=lambda s: -(SALIENCE.get(s["label"], 1.0) * s["confidence"]),
    )
    return sorted(scored[:max_clips], key=lambda s: s["time_s"])


def merge_overlaps(segments):
    """Two events seconds apart shouldn't produce two overlapping clips."""
    merged = []
    for start, end in sorted(segments):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def cut(video, start, duration, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{start:.2f}", "-i", str(video), "-t", f"{duration:.2f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", str(out_path)],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--spots", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reel.mp4"))
    ap.add_argument("--max-clips", type=int, default=12)
    ap.add_argument("--pre", type=float, default=PRE_ROLL_S)
    ap.add_argument("--post", type=float, default=POST_ROLL_S)
    args = ap.parse_args()

    spots = json.loads(args.spots.read_text())
    if not spots:
        raise SystemExit("no spots to clip")

    chosen = rank(spots, args.max_clips)
    segments = merge_overlaps([
        [max(0.0, s["time_s"] - args.pre), s["time_s"] + args.post]
        for s in chosen
    ])

    total = sum(e - s for s, e in segments)
    print(f"{len(chosen)} events -> {len(segments)} clips, {total:.0f}s total")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = []
        for i, (start, end) in enumerate(segments):
            part = tmp / f"clip_{i:03d}.mp4"
            cut(args.video, start, end - start, part)
            parts.append(part)
            print(f"  clip {i + 1}/{len(segments)}  {start:.0f}s-{end:.0f}s")

        manifest = tmp / "parts.txt"
        manifest.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in parts)
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
             "-safe", "0", "-i", str(manifest), "-c", "copy", str(args.out)],
            check=True,
        )

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()