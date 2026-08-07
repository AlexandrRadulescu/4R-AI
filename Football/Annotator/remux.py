"""Convert matches into a container the browser can actually play.

    python remux.py --video-dir matches

Browsers refuse the Matroska (.mkv) container outright, and also reject HEVC
and VP9 inside MP4, so annotate.html cannot open most broadcast rips. This
scans a folder, works out per file whether a lossless remux is enough or a real
re-encode is needed, and skips anything already converted.

    remux     ~1 minute per match, bit-identical video
    re-encode ~real-time, only when the codec itself is unsupported

Needs ffmpeg. If it isn't on PATH, `pip install imageio-ffmpeg` supplies a
bundled binary and this will find it automatically -- no PATH setup required.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from ffmpeg_tool import get_ffmpeg, ffmpeg_source, probe

# Containers a browser will never open, whatever codec is inside.
BAD_CONTAINERS = {".mkv", ".avi", ".ts", ".m2ts", ".flv", ".wmv", ".mpg",
                  ".mpeg", ".vob", ".ogv", ".m2v", ".mts"}
# Codecs a browser can decode inside MP4.
GOOD_VIDEO = {"h264", "avc1"}
GOOD_AUDIO = {"aac", "mp3", "opus", "vorbis", "flac"}


def plan(path: Path):
    """-> ('skip'|'remux'|'encode', reason)"""
    v, a, _ = probe(path)
    if v is None:
        return "error", "ffmpeg could not read this file"

    container_ok = path.suffix.lower() == ".mp4"
    video_ok = v in GOOD_VIDEO
    audio_ok = a is None or a in GOOD_AUDIO

    if container_ok and video_ok and audio_ok:
        return "skip", f"already playable ({v}/{a or 'no audio'})"
    if video_ok:
        # Video stream is fine; only the wrapper (and maybe audio) is wrong.
        return "remux", f"{path.suffix} container, {v} video is fine"
    return "encode", f"{v} video is not browser-decodable"


def convert(src: Path, dst: Path, mode: str, audio_ok: bool):
    if mode == "remux":
        # Copy the video stream untouched; re-encode audio only if needed.
        acodec = ["-c:a", "copy"] if audio_ok else ["-c:a", "aac", "-b:a", "128k"]
        cmd = [get_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src),
               "-c:v", "copy", *acodec, "-movflags", "+faststart", str(dst)]
    else:
        cmd = [get_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
               str(dst)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, default=Path("matches"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to the same folder as the source")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan without converting anything")
    args = ap.parse_args()

    get_ffmpeg()   # raises with install instructions if nothing is available
    print(f"ffmpeg: {ffmpeg_source()}\n")
    if not args.video_dir.is_dir():
        raise SystemExit(f"no such folder: {args.video_dir}")

    out_dir = args.out_dir or args.video_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.video_dir.iterdir()
                   if p.is_file()
                   and (p.suffix.lower() in BAD_CONTAINERS
                        or p.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm"}))
    if not files:
        raise SystemExit(f"no video files in {args.video_dir}")

    todo, skipped, errors = [], [], []
    print(f"inspecting {len(files)} file(s)...\n")

    for src in files:
        dst = out_dir / f"{src.stem}.mp4"
        if dst == src:
            # An .mp4 that still needs converting cannot be written in place --
            # ffmpeg would be reading and writing the same file.
            dst = out_dir / f"{src.stem}_h264.mp4"
        if dst.exists():
            skipped.append((src, "an .mp4 of that name already exists"))
            continue
        mode, reason = plan(src)
        if mode == "error":
            errors.append((src, reason))
        elif mode == "skip":
            skipped.append((src, reason))
        else:
            todo.append((src, dst, mode, reason))

    for src, reason in skipped:
        print(f"  skip    {src.name[:52]:<54} {reason}")
    for src, dst, mode, reason in todo:
        print(f"  {mode:<7} {src.name[:52]:<54} {reason}")
    for src, reason in errors:
        print(f"  ERROR   {src.name[:52]:<54} {reason}")

    n_encode = sum(1 for t in todo if t[2] == "encode")
    print(f"\n{len(todo)} to convert ({n_encode} need a full re-encode), "
          f"{len(skipped)} skipped, {len(errors)} unreadable")

    if args.dry_run or not todo:
        if args.dry_run:
            print("\n(dry run -- nothing was written)")
        return

    done, failed = 0, []
    for i, (src, dst, mode, _) in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] {mode} {src.name}")
        _, a, _ = probe(src)
        try:
            convert(src, dst, mode, a is None or a in GOOD_AUDIO)
            size = dst.stat().st_size / 1e6
            print(f"  wrote {dst.name}  ({size:.0f} MB)")
            done += 1
        except subprocess.CalledProcessError as exc:
            failed.append((src.name, str(exc)))
            print(f"  FAILED: {exc}")

    print(f"\n{done} converted, {len(failed)} failed")
    for name, err in failed:
        print(f"  ! {name}: {err}")

    if done:
        print("\nA remux keeps timestamps bit-identical, so features already")
        print("extracted from the source stay valid. After a re-encode, "
              "re-extract\nto be safe:")
        print("  python extract_features.py --video-dir matches")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()