""" 
    Runs once per match

Ouptut: features/<video_id>.npy, shape (T, 2048), float16

Call: 
# whole folder — extracts what's missing, skips what's cached
python extract_features.py --video-dir matches

# one match
python extract_features.py --video matches/liverpool_arsenal.mkv

# somewhere other than features/
python extract_features.py --video-dir matches --out-dir features

# re-extract everything (after changing FPS or the backbone)
python extract_features.py --video-dir matches --force

# smaller batches if you hit memory limits
python extract_features.py --video-dir matches --batch-size 16

"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision

from config import FPS, FEATURE_DIM

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".ts", ".webm"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def is_valid_cache(path: Path) -> bool:

    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.ndim == 2 and arr.shape[1] == FEATURE_DIM and len(arr) > 0
    except Exception:
        return False

def build_backbone(device):

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    net = torchvision.models.resnet50(weights = weights)

    net.fc = nn.Identity()

    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad = False
    return net

def preprocess(frame_bgr):

    img = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224,224), interpolation=cv.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    return torch.from_numpy(img).permute(2,0,1)

def extract(video_path: Path, out_path: Path, device, backbone, batch_size: int = 64):
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    native_fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    step = max(1, round(native_fps / FPS))
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    print(
    f"{video_path.name}: {native_fps:.1f} fps native; "
    f"keeping every {step}th frame -> ~{total // step} timesteps"
)

    features, batch, idx = [], [], 0

    def flush():
        if not batch:
            return
        with torch.no_grad():
            stacked = torch.stack(batch).to(device)
            out = backbone(stacked).cpu().numpy().astype(np.float16)
        features.append(out)
        batch.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            batch.append(preprocess(frame))
            if len(batch) == batch_size:
                flush()
                done = sum(len(f) for f in features)
                print(f"    {done} timesteps", end="\r", flush=True)

        idx += 1

    flush()
    cap.release()

    if not features:
        raise RuntimeError("no frames decoded - Is the video readable?")

    arr = np.concatenate(features, axis=0)
    assert arr.shape[1] == FEATURE_DIM, f"unexpected feature dim {arr.shape[1]}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary name and rename on success, so an interrupted run never leaves half-written .npy
    tmp = out_path.with_suffix(".partial.npy")
    np.save(tmp, arr)
    tmp.replace(out_path)

    print(f"\nsaved {out_path}  shape={arr.shape}  "f"({arr.nbytes / 1e6:.0f} MB)")
    return len(arr)

def find_videos(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )

def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="a single match file")
    group.add_argument("--video-dir", type=Path,
                       help="a folder of matches; already-cached ones are skipped")
    ap.add_argument("--out-dir", type=Path, default=Path("features"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if a valid .npy already exists")
    args = ap.parse_args()
 
    if args.video:
        videos = [args.video]
    else:
        if not args.video_dir.is_dir():
            raise SystemExit(f"not a folder: {args.video_dir}")
        videos = find_videos(args.video_dir)
        if not videos:
            raise SystemExit(
                f"no video files in {args.video_dir} "
                f"(looking for {', '.join(sorted(VIDEO_SUFFIXES))})"
            )
 
    # Decide what to do before loading the backbone -- if everything is
    # already cached there is no reason to spend 30 seconds loading ResNet.
    todo, skipped = [], []
    for path in videos:
        out = args.out_dir / f"{path.stem}.npy"
        if not args.force and is_valid_cache(out):
            skipped.append(path)
        else:
            todo.append((path, out))
 
    print(f"{len(videos)} video(s): {len(todo)} to extract, "
          f"{len(skipped)} already cached")
    for path in skipped:
        print(f"  skip  {path.name}")
    if not todo:
        print("\nnothing to do.")
        return
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\ndevice: {device}")
    backbone = build_backbone(device)   # loaded once, reused for every match
 
    done, failed = 0, []
    for i, (path, out) in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] {path.name}")
        try:
            extract(path, out, device, backbone, args.batch_size)
            done += 1
        except Exception as exc:
            # One unreadable file shouldn't abandon the rest of the queue.
            failed.append((path.name, str(exc)))
            print(f"  FAILED: {exc}")
 
    print(f"\n{done} extracted, {len(skipped)} skipped, {len(failed)} failed")
    for name, err in failed:
        print(f"  ! {name}: {err}")
    if failed:
        raise SystemExit(1)
 
 
if __name__ == "__main__":
    main()