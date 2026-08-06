"""List of spotted events.

    python spot.py --video-id liverpool_arsenal --checkpoint checkpoints/best.pt

Writes spots/<video_id>.json, which is the input to make_reel.py -- and, later,
the seed of the event stream your commentary layer will consume.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import PEAK_THRESHOLD, NMS_GAP_S
from model import Spotter
from postprocess import predict_video, pick_peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--features", type=Path, default=Path("features"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/best.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("spots"))
    ap.add_argument("--threshold", type=float, default=PEAK_THRESHOLD)
    ap.add_argument("--nms-gap", type=float, default=NMS_GAP_S)
    ap.add_argument("--save-curve", action="store_true",
                    help="also dump the raw probability curve for inspection")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Spotter().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    features = np.load(args.features / f"{args.video_id}.npy", mmap_mode="r")
    probs = predict_video(model, features, device)
    spots = pick_peaks(probs, threshold=args.threshold, nms_gap_s=args.nms_gap)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.video_id}.json"
    out.write_text(json.dumps(spots, indent=2))

    if args.save_curve:
        np.save(args.out_dir / f"{args.video_id}_curve.npy", probs)

    print(f"{len(spots)} events -> {out}\n")
    for s in spots:
        m, sec = divmod(s["time_s"], 60)
        print(f"  {int(m):3d}:{sec:05.2f}  {s['label']:<6} "
              f"{s['confidence']:.2f}")


if __name__ == "__main__":
    main()