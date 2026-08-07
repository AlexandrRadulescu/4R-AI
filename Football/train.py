"""
    python train.py --features features --labels labels.csv --train-videos match1 match2 match3 --val-videos match4
"""


import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from history import TrainingHistory
import time
from collections import Counter

from config import BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, FPS, CLASSES
from dataset import SpottingDataset, compute_pos_weight, load_labels
from model import Spotter
from postprocess import predict_video, pick_peaks, evaluate_spots


def run_validation(model, feature_dir, labels_csv, video_ids, device):

    all_events = load_labels(labels_csv)
    idx_to_class = {i: c for i, c in enumerate(CLASSES)}

    agg = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}

    for vid in video_ids:
        features = np.load(Path(feature_dir) / f"{vid}.npy", mmap_mode="r")
        probs = predict_video(model, features, device)
        spots = pick_peaks(probs)
        truth = [(step / FPS, idx_to_class[cls]) for step, cls in all_events.get(vid, [])]

        for label, m in evaluate_spots(spots, truth).items():
            for k in ("tp", "fp", "fn"):
                agg[label][k] += m[k]

        report, f1s = {}, []

        for label, m in agg.items():
            p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
            r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            report[label] = {"precision": round(p, 3), "recall": round(r, 3),
                            "f1": round(f1, 3), **m}
            f1s.append(f1)
 
    return float(np.mean(f1s)), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("features"))
    ap.add_argument("--labels", type=Path, default=Path("labels.csv"))
    ap.add_argument("--train-videos", nargs="+", required=True)
    ap.add_argument("--val-videos", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--run-name", default=None, help="label for this run inside history.json")
    args = ap.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
 
    args.out.mkdir(parents=True, exist_ok=True)
    history = TrainingHistory(args.out / "history.json", args.run_name)
    history.set_environment(device)
 
    train_ds = SpottingDataset(
        args.features, args.labels, args.train_videos, train=True
    )
    print(f"train windows: {len(train_ds)}")
 
    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
 
    model = Spotter().to(device)
    history.set_model(model)
 
    pos_weight = compute_pos_weight(train_ds).to(device)
    print(f"pos_weight per class: "
          f"{dict(zip(CLASSES, pos_weight.cpu().numpy().round(1)))}")
 
    # Dataset composition, recorded so the run is interpretable months later.
    all_events = load_labels(args.labels)
    events_per_class = Counter()
    for vid in args.train_videos:
        for _, cls in all_events.get(vid, []):
            events_per_class[CLASSES[cls]] += 1
    hours = sum(len(train_ds.features[v]) for v in args.train_videos) / FPS / 3600
    history.set_dataset(
        args.train_videos, args.val_videos, len(train_ds),
        {c: events_per_class.get(c, 0) for c in CLASSES},
        pos_weight.cpu().numpy(), hours,
    )
 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs
    )
 
    best_f1 = -1.0
 
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(loss.item())
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
 
        mean_loss = float(np.mean(losses))
        history.log_epoch(epoch, mean_loss, lr_now, time.perf_counter() - tick)
        line = f"epoch {epoch:3d}  loss {mean_loss:.4f}"
 
        if epoch % 5 == 0 or epoch == args.epochs:
            mean_f1, report = run_validation(
                model, args.features, args.labels, args.val_videos, device
            )
            history.log_validation(epoch, mean_f1, report)
            line += f"  val_f1 {mean_f1:.3f}"
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                torch.save(model.state_dict(), args.out / "best.pt")
                (args.out / "best_report.json").write_text(
                    json.dumps(report, indent=2)
                )
                line += "  *saved*"
 
        print(line)
 
    history.finalise(args.out / "best.pt")
 
    print(f"\nbest val F1: {best_f1:.3f}")
    print(f"checkpoint:  {args.out / 'best.pt'}")
    print(f"history:     {args.out / 'history.json'}")
 
 
if __name__ == "__main__":
    main()