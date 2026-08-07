"""Reconcile feature files with label CSVs, then write labels.csv + split.json.

    python prepare_split.py --features features --labels-dir validated_shots

Solves three chores at once:

1. Matches each .npy to its label CSV even when the names disagree -- the
   annotator sanitises video_id to word characters, so "Arsenal vs Real
   Betis 05.08.2026.npy" and "Arsenal_vs_Real_Betis_05_08_2026" are the same
   match but don't compare equal.
2. Merges every per-match CSV into one labels.csv with consistent ids.
3. Writes split.json so you never type 22 filenames on a command line.

Run with --rename to also rename the .npy files to their slug, which makes
everything downstream easier to type. Without it, nothing on disk is touched
except the two output files.
"""

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def slugify(name: str) -> str:
    """Filename or video_id -> a canonical, typeable id.

    Must be stable across both sides of the match: "Molde vs Sapsborg (1)"
    and "Molde_vs_Sapsborg_1" both land on molde_vs_sapsborg_1.
    """
    name = re.sub(r"\.npy$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_labels$", "", name, flags=re.IGNORECASE)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode()
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def read_csv_rows(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("features"))
    ap.add_argument("--labels-dir", type=Path, default=Path("validated_shots"),
                    help="folder of per-match CSVs exported from annotate.html")
    ap.add_argument("--out-labels", type=Path, default=Path("labels.csv"))
    ap.add_argument("--out-split", type=Path, default=Path("split.json"))
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rename", action="store_true",
                    help="also rename the .npy files on disk to their slug")
    args = ap.parse_args()

    if not args.features.is_dir():
        raise SystemExit(f"no such folder: {args.features}")
    if not args.labels_dir.is_dir():
        raise SystemExit(f"no such folder: {args.labels_dir}")

    # --- index both sides by slug ------------------------------------------
    npys = {}
    for p in sorted(args.features.glob("*.npy")):
        slug = slugify(p.stem)
        if slug in npys:
            raise SystemExit(
                f"two feature files collapse to the same id '{slug}':\n"
                f"  {npys[slug].name}\n  {p.name}\nRename one of them."
            )
        npys[slug] = p

    csv_rows = defaultdict(list)
    csv_source = {}
    for p in sorted(args.labels_dir.glob("*.csv")):
        for row in read_csv_rows(p):
            vid = (row.get("video_id") or "").strip()
            if not vid:
                continue
            slug = slugify(vid)
            csv_rows[slug].append(row)
            csv_source.setdefault(slug, p.name)

    matched = sorted(set(npys) & set(csv_rows))
    only_npy = sorted(set(npys) - set(csv_rows))
    only_csv = sorted(set(csv_rows) - set(npys))

    print(f"{len(npys)} feature files, {len(csv_rows)} labelled matches, "
          f"{len(matched)} matched\n")

    if only_csv:
        print(f"{len(only_csv)} labelled but NO features (these will be dropped):")
        for s in only_csv:
            print(f"  ! {s}   (from {csv_source[s]})")
        print()
    if only_npy:
        print(f"{len(only_npy)} features but NO labels (unannotated):")
        for s in only_npy:
            print(f"  ~ {s}")
        print()

    if not matched:
        raise SystemExit("nothing matched -- check the two folder paths")

    # --- optional rename ----------------------------------------------------
    if args.rename:
        for slug in matched:
            src = npys[slug]
            dst = src.with_name(f"{slug}.npy")
            if src != dst:
                if dst.exists():
                    raise SystemExit(f"cannot rename, {dst.name} already exists")
                src.rename(dst)
                npys[slug] = dst
        print(f"renamed {len(matched)} feature files to their slug\n")

    # --- merged labels.csv --------------------------------------------------
    total = 0
    with open(args.out_labels, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "timestamp_s", "label"])
        writer.writeheader()
        for slug in matched:
            for row in sorted(csv_rows[slug],
                              key=lambda r: float(r["timestamp_s"])):
                writer.writerow({
                    "video_id": slug,
                    "timestamp_s": row["timestamp_s"],
                    "label": row["label"].strip(),
                })
                total += 1
    print(f"wrote {args.out_labels}  ({total} events across {len(matched)} matches)")

    # --- split --------------------------------------------------------------
    # Split by whole match, never within one: clips from the same match share
    # kit, pitch and camera operator, so a within-match split leaks.
    rng = random.Random(args.seed)
    shuffled = matched[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_fraction))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])

    args.out_split.write_text(json.dumps({"train": train, "val": val}, indent=2))
    print(f"wrote {args.out_split}  ({len(train)} train / {len(val)} val)")
    print(f"\nnow run:\n  python train.py --split {args.out_split}")


if __name__ == "__main__":
    main()