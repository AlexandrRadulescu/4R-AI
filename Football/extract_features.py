""" 
    Runs once per match

Ouptut: features/<video_id>.npy, shape (T, 2048), float16

Call: python extract_features.py --video matches/liverpool_arsenal.mkv

"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision

from config import FPS, FEATURE_DIM

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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

def extract(video_path: Path, out_path: Path, device, batch_size: int =64):
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    native_fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    step = max(1, round(native_fps / FPS))
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"{video_path.name}: {native_fps:.1f} fps native"f"keeping every {step}th frame -> ~{total // step} timesteps")

    backbone = build_backbone(device)
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
    np.save(out_path, arr)

    print(f"\nsaved {out_path}  shape={arr.shape}  "f"({arr.nbytes / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("features"))
    ap.add_argument("--batch-size", type=int, default=64)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device {device}")

    out = args.out_dir / f"{args.video.stem}.npy"
    extract(args.video, out, device, args.batch_size)


if __name__ == "__main__":
    main()