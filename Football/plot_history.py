"""Turn history.json into figures you can put on a slide.

    python plot_history.py --history checkpoints/history.json
    python plot_history.py --history checkpoints/history.json --dark --dpi 200

Produces four panels:
    1. Training loss with validation F1 overlaid
    2. Per-class F1 across epochs
    3. Precision vs recall per class at the best epoch
    4. Hits, misses and false alarms per class

Saved as PNG next to the history file, plus an SVG for slides that need to
scale. Also prints a plain-text summary you can paste into notes.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Same class colours as the annotator, so figures and tool agree.
CLASS_COLOURS = {"goal": "#E09A1E", "shot": "#2E90B4", "card": "#D64C60"}
FALLBACK = ["#6A8E7F", "#8E6A88", "#8E7F6A"]


def theme(dark: bool):
    if dark:
        return {"bg": "#12211F", "fg": "#E8F1EC", "grid": "#2C4A46",
                "loss": "#E8F1EC", "f1": "#E09A1E"}
    return {"bg": "#FFFFFF", "fg": "#1B2B28", "grid": "#D8E2DE",
            "loss": "#1B2B28", "f1": "#C77F14"}


def colour_for(label, i):
    return CLASS_COLOURS.get(label, FALLBACK[i % len(FALLBACK)])


def style_axes(ax, t):
    ax.set_facecolor(t["bg"])
    ax.grid(True, color=t["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["fg"], labelsize=9)
    ax.yaxis.label.set_color(t["fg"])
    ax.xaxis.label.set_color(t["fg"])
    ax.title.set_color(t["fg"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path,
                    default=Path("checkpoints/history.json"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    h = json.loads(args.history.read_text())
    t = theme(args.dark)
    classes = h["config"]["CLASSES"]
    epochs = h["epochs"]
    if not epochs:
        raise SystemExit("history.json has no epochs -- did training run?")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": t["bg"],
        "savefig.facecolor": t["bg"],
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.86,
                        bottom=0.09, left=0.07, right=0.95)

    # ---- 1. loss + val F1 --------------------------------------------------
    ax = axes[0][0]
    style_axes(ax, t)
    xs = [e["epoch"] for e in epochs]
    ax.plot(xs, [e["train_loss"] for e in epochs],
            color=t["loss"], linewidth=2, label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Loss and validation F1", fontsize=11, fontweight="bold",
                 loc="left", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    val = [(e["epoch"], e["validation"]["mean_f1"])
           for e in epochs if e["validation"]]
    if val:
        ax2 = ax.twinx()
        ax2.plot([v[0] for v in val], [v[1] for v in val],
                 color=t["f1"], linewidth=2, marker="o", markersize=5,
                 label="val F1")
        ax2.set_ylabel("mean F1", color=t["f1"])
        ax2.set_ylim(0, 1.02)
        ax2.tick_params(colors=t["f1"], labelsize=9)
        for side in ("top", "left"):
            ax2.spines[side].set_visible(False)
        ax2.spines["right"].set_color(t["grid"])
        if h.get("best"):
            ax2.axvline(h["best"]["epoch"], color=t["f1"],
                        linestyle=":", linewidth=1.2, alpha=0.7)

    # ---- 2. per-class F1 over time ----------------------------------------
    ax = axes[0][1]
    style_axes(ax, t)
    for i, label in enumerate(classes):
        pts = [(e["epoch"], e["validation"]["per_class"][label]["f1"])
               for e in epochs if e["validation"]]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=colour_for(label, i), linewidth=2,
                    marker="o", markersize=4, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("F1 by class", fontsize=11, fontweight="bold",
                 loc="left", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    leg = ax.legend(frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])

    best = h.get("best")

    # ---- 3. precision vs recall at best epoch ------------------------------
    ax = axes[1][0]
    style_axes(ax, t)
    if best:
        width = 0.36
        for i, label in enumerate(classes):
            m = best["per_class"][label]
            c = colour_for(label, i)
            ax.bar(i - width / 2, m["precision"], width, color=c, alpha=0.95)
            ax.bar(i + width / 2, m["recall"], width, color=c, alpha=0.5)
            ax.text(i - width / 2, m["precision"] + 0.03, f"{m['precision']:.2f}",
                    ha="center", fontsize=8, color=t["fg"])
            ax.text(i + width / 2, m["recall"] + 0.03, f"{m['recall']:.2f}",
                    ha="center", fontsize=8, color=t["fg"])
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("score")
        ax.set_title(f"Precision (solid) vs recall (faded) "
                     f"— epoch {best['epoch']}",
                     fontsize=11, fontweight="bold", loc="left", pad=10)

    # ---- 4. hits, misses, false alarms -------------------------------------
    ax = axes[1][1]
    style_axes(ax, t)
    if best:
        for i, label in enumerate(classes):
            m = best["per_class"][label]
            c = colour_for(label, i)
            ax.barh(i, m["tp"], color=c, label="_")
            ax.barh(i, m["fn"], left=m["tp"], color=c, alpha=0.45)
            ax.barh(i, m["fp"], left=m["tp"] + m["fn"], color=t["grid"])
            total = m["tp"] + m["fn"] + m["fp"]
            ax.text(total + 0.4, i,
                    f"{m['tp']} hit  {m['fn']} miss  {m['fp']} false",
                    va="center", fontsize=8.5, color=t["fg"])
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.invert_yaxis()
        ax.set_xlabel("events")
        ax.set_xlim(0, max(
            best["per_class"][c]["tp"] + best["per_class"][c]["fn"]
            + best["per_class"][c]["fp"] for c in classes) * 1.9 + 1)
        ax.set_title("Hits, misses, false alarms", fontsize=11,
                     fontweight="bold", loc="left", pad=10)

    # ---- title block -------------------------------------------------------
    d = h.get("dataset", {})
    mdl = h.get("model", {})
    fig.text(0.07, 0.955, f"Action spotter — {h['run_name']}",
             fontsize=15, fontweight="bold", color=t["fg"])
    subtitle = (
        f"{len(d.get('train_videos', []))} train / "
        f"{len(d.get('val_videos', []))} val matches  ·  "
        f"{d.get('num_train_windows', '?')} windows  ·  "
        f"{mdl.get('parameters', 0):,} params  ·  "
        f"{mdl.get('receptive_field_seconds', '?')}s context"
    )
    if best:
        subtitle += f"  ·  best mean F1 {best['mean_f1']:.3f}"
    fig.text(0.07, 0.915, subtitle, fontsize=9.5, color=t["fg"], alpha=0.75)

    out = args.out or args.history.parent / "training_report.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    # ---- text summary ------------------------------------------------------
    print(f"run           {h['run_name']}")
    print(f"device        {h['environment'].get('device')}")
    print(f"epochs        {len(epochs)}")
    if h.get("total_seconds"):
        print(f"wall time     {h['total_seconds'] / 60:.1f} min")
    print(f"events        {d.get('events_per_class')}")
    if d.get("hours_of_footage"):
        print(f"footage       {d['hours_of_footage']:.1f} h")
    if best:
        print(f"\nbest epoch    {best['epoch']}  (mean F1 {best['mean_f1']:.3f})")
        for label in classes:
            m = best["per_class"][label]
            print(f"  {label:<6} P {m['precision']:.2f}  R {m['recall']:.2f}  "
                  f"F1 {m['f1']:.2f}   {m['tp']} hit / {m['fn']} miss / "
                  f"{m['fp']} false")
    print(f"\nwrote {out}")
    print(f"wrote {out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()