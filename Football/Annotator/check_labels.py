"""Sanity-check labels.csv before you spend an evening training on it.

    python check_labels.py --labels labels.csv --features features

Annotation errors are silent. A timestamp past the end of the video, a stray
class name, a match you thought you labelled but didn't -- none of these throw
an error, they just quietly cap your F1 and leave you tuning hyperparameters
against a data problem. Run this after every annotation session.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import FPS, CLASSES, WINDOW

# A shot and a goal seconds apart is normal football. Two goals 3s apart is
# almost always the same event marked twice, or a replay logged as live.
SUSPICIOUS_GAP_S = 5.0


def read_rows(path: Path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"video_id", "timestamp_s", "label"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"FAIL  labels.csv missing column(s): {sorted(missing)}")
        return list(reader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("labels.csv"))
    ap.add_argument("--features", type=Path, default=Path("features"))
    args = ap.parse_args()

    rows = read_rows(args.labels)
    errors, warnings = [], []

    # --- parse -------------------------------------------------------------
    events = defaultdict(list)
    class_counts = Counter()
    unknown = Counter()

    for i, row in enumerate(rows, start=2):  # line 1 is the header
        label = row["label"].strip()
        vid = row["video_id"].strip()
        try:
            t = float(row["timestamp_s"])
        except ValueError:
            errors.append(f"line {i}: timestamp '{row['timestamp_s']}' is not a number")
            continue
        if t < 0:
            errors.append(f"line {i}: negative timestamp {t}")
            continue
        if label not in CLASSES:
            unknown[label] += 1
            continue
        events[vid].append((t, label))
        class_counts[label] += 1

    for label, n in unknown.items():
        errors.append(f"unknown label '{label}' on {n} row(s) -- typo, or missing "
                      f"from CLASSES in config.py")

    # --- cross-check against cached features -------------------------------
    feature_files = {p.stem for p in args.features.glob("*.npy")}
    labelled = set(events)

    for vid in sorted(labelled - feature_files):
        errors.append(f"'{vid}' is labelled but has no features -- run extract_features.py")
    for vid in sorted(feature_files - labelled):
        warnings.append(f"'{vid}' has features but no labels -- unlabelled, or a val match "
                        f"you haven't annotated yet")

    durations = {}
    for vid in sorted(labelled & feature_files):
        n_steps = len(np.load(args.features / f"{vid}.npy", mmap_mode="r"))
        durations[vid] = n_steps / FPS
        if n_steps < WINDOW:
            errors.append(f"'{vid}' is only {n_steps} steps, shorter than the "
                          f"{WINDOW}-step training window")
        for t, label in events[vid]:
            if t > durations[vid]:
                errors.append(f"'{vid}': {label} at {t:.1f}s is past the end of the "
                              f"video ({durations[vid]:.1f}s)")

    # --- near-duplicates ---------------------------------------------------
    for vid, evs in events.items():
        by_class = defaultdict(list)
        for t, label in evs:
            by_class[label].append(t)
        for label, times in by_class.items():
            times.sort()
            for a, b in zip(times, times[1:]):
                if b - a < SUSPICIOUS_GAP_S:
                    warnings.append(f"'{vid}': two {label} marks {b - a:.1f}s apart "
                                    f"({a:.1f}s, {b:.1f}s) -- duplicate, or a replay?")

    # --- report ------------------------------------------------------------
    print(f"{len(rows)} rows across {len(events)} matches\n")

    print("events per class")
    for label in CLASSES:
        n = class_counts[label]
        if n < 40:
            verdict = "too few -- will not generalise"
        elif n < 100:
            verdict = "workable, expect it to be the weak class"
        elif n < 150:
            verdict = "approaching solid"
        else:
            verdict = "solid"
        print(f"  {label:<8} {n:>5}   {verdict}")

    if durations:
        total_h = sum(durations.values()) / 3600
        print(f"\n{total_h:.1f} hours of footage, "
              f"{len(rows) / max(total_h, 1e-9):.1f} events per hour")

    per_match = Counter(vid for vid in events for _ in events[vid])
    if len(per_match) > 1:
        counts = np.array(list(per_match.values()))
        thin = [v for v, c in per_match.items() if c < counts.mean() / 2]
        for vid in sorted(thin):
            warnings.append(f"'{vid}' has only {per_match[vid]} events, well below the "
                            f"average of {counts.mean():.0f} -- partially annotated?")

    if warnings:
        print(f"\n{len(warnings)} warning(s)")
        for w in warnings:
            print(f"  ~ {w}")

    if errors:
        print(f"\n{len(errors)} error(s)")
        for e in errors:
            print(f"  ! {e}")
        raise SystemExit(1)

    print("\nno errors. Missed events are the one thing this cannot detect --")
    print("only a second pass over the footage will catch those.")


if __name__ == "__main__":
    main()
