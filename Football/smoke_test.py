"""Validate the pipeline before you label a single match.

Generates synthetic features with a planted signal at known timestamps, so you
can confirm training and spotting work end-to-end without any video. If this
doesn't reach a high F1, the problem is in the code -- not in your data.

    python smoke_test.py
    python train.py --features _smoke/features --labels _smoke/labels.csv \
        --train-videos m1 m2 m3 --val-videos val1 --epochs 15 \
        --out _smoke/checkpoints

Expect val_f1 above 0.85 within ~10 epochs. Delete _smoke/ when done.
"""

import csv
from pathlib import Path

import numpy as np

from config import FEATURE_DIM, FPS, CLASSES

T = 1500              # ~12 min at 2fps
OUT = Path("_smoke")


def main():
    rng = np.random.default_rng(0)
    features_dir = OUT / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Each class gets its own direction in feature space. Real footage is not
    # this clean -- this only tests the plumbing, not the difficulty.
    signatures = rng.normal(size=(len(CLASSES), FEATURE_DIM)).astype(np.float32) * 2.0

    rows = []
    for vid, n_events in [("m1", 25), ("m2", 25), ("m3", 25), ("val1", 20)]:
        x = rng.normal(size=(T, FEATURE_DIM)).astype(np.float32) * 0.5
        times = sorted(rng.choice(np.arange(60, T - 60), size=n_events,
                                  replace=False))
        for t in times:
            cls = int(rng.integers(0, len(CLASSES)))
            for offset in range(-3, 4):       # smear over ~3 seconds
                x[t + offset] += signatures[cls] * (1.0 - abs(offset) / 5)
            rows.append({
                "video_id": vid,
                "timestamp_s": t / FPS,
                "label": CLASSES[cls],
            })
        np.save(features_dir / f"{vid}.npy", x.astype(np.float16))

    with open(OUT / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "timestamp_s", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} synthetic events across 4 videos -> {OUT}/")
    print("\nnow run:")
    print("  python train.py --features _smoke/features "
          "--labels _smoke/labels.csv \\")
    print("      --train-videos m1 m2 m3 --val-videos val1 "
          "--epochs 15 --out _smoke/checkpoints")


if __name__ == "__main__":
    main()