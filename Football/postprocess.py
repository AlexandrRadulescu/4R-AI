"""Turning a dense probability curve into discrete events.
"""


import numpy as np
import torch

from config import FPS, WINDOW, STRIDE, NUM_CLASSES, CLASSES, PEAK_THRESHOLD, NMS_GAP_S

@torch.no_grad()
def predict_video(model, features, device, window = WINDOW, stride = STRIDE):
    """
    Full match features -> probabilities
    """

    model.eval()
    n = len(features)
    total = np.zeros((n, NUM_CLASSES), dtype = np.float64)

    counts = np.zeros((n,1),dtype=  np.float64)

    starts = list(range(0, max(1, n-window+1), stride))
    if starts[-1] + window < n:
        starts.append(n - window)

    for start in starts:
        end = min(start + window, n)
        chunk = np.asarray(features[start:end], dtype = np.float32)
        if len(chunk) < window: ## pad the tail
            pad = np.zeros((window - len(chunk), chunk.shape[1]), np.float32)
            chunk = np.concatenate([chunk, pad], axis = 0)

        x = torch.from_numpy(chunk).unsqueeze(0).to(device)

        probs = torch.sigmoid(model(x))[0].cpu().numpy()

        total[start:end] += probs[: end - start]
        counts[start:end] += 1 


    return (total / np.maximum(counts, 1)).astype(np.float32)


def pick_peaks(probs, threshold = PEAK_THRESHOLD, nms_gap_s = NMS_GAP_S, fps = FPS):
    # Time-Confidence Probabilities

    gap = int(round(nms_gap_s * fps))
    spots = []

    for cls in range(probs.shape[1]):
        curve = probs[:,cls]
        candidates = np.flatnonzero(curve >= threshold)
        if candidates.size == 0:
            continue

        order = candidates[np.argsort(-curve[candidates])]

        taken = []

        for idx in order:
            if all(abs(idx - t) >= gap for t in taken):
                taken.append(idx)
                spots.append({
                    "time_s": float(idx/fps),
                    "label": CLASSES[cls],
                    "confidence": float(curve[idx]),
                })

    return sorted(spots, key=lambda s: s["time_s"])


def evaluate_spots(predicted, ground_truth, tolerance_s = 5.0):

    results = {}
    for label in CLASSES:
        preds = sorted(
            [p for p in predicted if p["label"] == label],
            key = lambda p: -p["confidence"],
        )
        truths = [t for t, l in ground_truth if l == label]

        matched = set()
        tp = 0

        for p in preds:
            best, best_dist = None, tolerance_s

            for i, t in enumerate(truths):
                if i in matched:
                    continue
                d = abs(p["time_s"] - t)

                if d <= best_dist:
                    best, best_dist = i,d

            if best is not None:
                matched.add(best)
                tp += 1

        fp = len(preds) - tp
        fn = len(truths) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp+fn) else 0.0

        f1 = (2* precision * recall / (precision + recall) if (precision + recall) else 0.0)

        results[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
        }

    return results