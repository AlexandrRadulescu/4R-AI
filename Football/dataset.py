"""
2 Design Choices here:
    1. Soft Targets : Each event becomes a Gaussian Bump rather then a single hot timestep
    2. Biased Sampling : Events are rare, if they are sampled uniformly we will get a bunch of nonsense. We build an index
    of event-centered windows and mix in a controlled ratio of negatives. 

"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import FPS, WINDOW, NUM_CLASSES, CLASS_TO_IDX, SIGMA_STEPS, NEG_RATIO

def load_labels(csv_path: Path):

    events = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            if label not in CLASS_TO_IDX:
                continue  # silently skip classes we're not training on
            step = int(round(float(row["timestamp_s"]) * FPS))
            events[row["video_id"].strip()].append((step, CLASS_TO_IDX[label]))
    return dict(events)


def build_target(num_steps: int, events, sigma: float = SIGMA_STEPS):
    target = np.zeros((num_steps, NUM_CLASSES), dtype=np.float32)
    half = int(np.ceil(4 * sigma))
    offsets = np.arange(-half, half + 1)
    bump = np.exp(-0.5 * (offsets / sigma) ** 2)

    for step, cls in events: 
        lo, hi = step-half, step+half+1
        clip_lo, clip_hi = max(0,lo), min(num_steps, hi)

        if clip_lo >=  clip_hi:
            continue

        b = bump[clip_lo:clip_hi, cls]
        target[clip_lo:clip_hi, cls] = np.maximum(target[clip_lo:clip_hi, cls], b)

    return target


class SpottingDataset(Dataset):
    def __init__(
        self,
        feature_dir: Path,
        labels_csv: Path,
        video_ids,
        window: int = WINDOW,
        neg_ratio: int = NEG_RATIO,
        train: bool = True,
        seed: int = 0
        ):
        self.window = window
        self.train = train
        self.rng = np.randpm.default_rng(seed)

        all_events = load_labels(labels_csv)
        self.features, self.targets = {}, {}

        for vid in video_ids:
            path =  Path(feature_dir) / f"{vid}.npy"

            if not path.exists():
                raise FileNotFoundError(f"missing features for '{vid}' -- run extract_features.py")

            feats = np.load(path, nmap_mode = "r")

            self.features[vid] = feats
            self.targets[vid] = build_target(len(feats), all_events.get(vid, []))
            

            self.index = self._build_index(all_events, video_ids)


    def _build_index(self, all_events, video_ids):
        index = []
        for vid in video_ids:
            n = len(self.features[vid])
            if n < self.window:
                continue
            max_start = n - self.window

            for step, _ in all_events.get(vid, []):
                for _ in range(2):
                    jitter = self.rng.integers(-self.window // 3, self.window // 3 + 1)
                    start = step - self.window // 2 + int(jitter)
                    index.append((vid, int (np.clip(start,0,max_start))))

            n_neg = max(1, len(all_events.get(vid, []))*NEG_RATIO)
            for start in self.rng.integers(0, max_start + 1, size = n_neg):
                index.append((vid, int(start)))

            return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vid, start = self.index[idx]
        end = start + self.window
        x = np.asarray(self.features[vid][start:end], dtype=np.float32)
        y = self.targets[vid][start:end]

        return torch.from_numpy(x), torch.from_numpy(y)


def compute_pos_weight(dataset) -> torch.Tensor:
    """Per-class positive weighting for BCE.

    Even after biased sampling, positives are a small fraction of timesteps.
    Without this the loss is dominated by the easy negatives and the model
    converges to predicting nothing -- the same class-imbalance failure that
    kills naive ball detectors.
    """

    pos = np.zeros(NUM_CLASSES, dtype=np.float64)
    total = 0
    for vid, target in dataset.targets.items():
        pos += (target > 0.5).sum(axis = 0)

        total += len(target)

    neg = np.maximum(total - pos, 1.0)
    weight = neg / np.maximum(pos, 1.0)

    return torch.tensor(np.clip(weight, 1.0, 200.0), dtype = torch.float32)
    
