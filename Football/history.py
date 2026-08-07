"""Records everything about a training run to a single JSON file.

Written after every epoch, so a crash or a Ctrl-C still leaves you with a
usable history rather than nothing. The output feeds plot_history.py, but it's
plain JSON -- load it in a notebook, a spreadsheet, or anything else.

The run metadata matters as much as the curves.
"""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import config as cfg


class TrainingHistory:
    def __init__(self, out_path: Path, run_name: str = None):
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()

        self.data = {
            "run_name": run_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_at": None,
            "total_seconds": None,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": None,
            },
            "config": {
                k: getattr(cfg, k) for k in [
                    "FPS", "FEATURE_DIM", "CLASSES", "WINDOW", "HIDDEN",
                    "DILATIONS", "DROPOUT", "SIGMA_STEPS", "BATCH_SIZE", "LR",
                    "WEIGHT_DECAY", "EPOCHS", "NEG_RATIO", "STRIDE",
                    "PEAK_THRESHOLD", "NMS_GAP_S",
                ]
            },
            "model": {},
            "dataset": {},
            "epochs": [],
            "best": None,
        }

    # -- metadata ------------------------------------------------------------
    def set_environment(self, device: str):
        self.data["environment"]["device"] = str(device)

    def set_model(self, model):
        n = sum(p.numel() for p in model.parameters())
        rf = model.receptive_field
        self.data["model"] = {
            "parameters": int(n),
            "size_mb": round(n * 4 / 1e6, 2),
            "receptive_field_steps": int(rf),
            "receptive_field_seconds": round(rf / cfg.FPS, 1),
        }

    def set_dataset(self, train_videos, val_videos, num_windows,
                    events_per_class, pos_weight, hours=None):
        self.data["dataset"] = {
            "train_videos": list(train_videos),
            "val_videos": list(val_videos),
            "num_train_windows": int(num_windows),
            "events_per_class": {k: int(v) for k, v in events_per_class.items()},
            "pos_weight": {
                c: round(float(w), 2)
                for c, w in zip(cfg.CLASSES, pos_weight)
            },
            "hours_of_footage": round(hours, 2) if hours else None,
        }

    # -- per epoch -----------------------------------------------------------
    def log_epoch(self, epoch: int, loss: float, lr: float, seconds: float):
        self.data["epochs"].append({
            "epoch": int(epoch),
            "train_loss": round(float(loss), 6),
            "learning_rate": float(lr),
            "seconds": round(float(seconds), 2),
            "validation": None,
        })
        self.save()

    def log_validation(self, epoch: int, mean_f1: float, report: dict):
        """Attach a validation result to an already-logged epoch."""
        for record in self.data["epochs"]:
            if record["epoch"] == epoch:
                record["validation"] = {
                    "mean_f1": round(float(mean_f1), 4),
                    "per_class": report,
                }
                break

        best = self.data["best"]
        if best is None or mean_f1 > best["mean_f1"]:
            self.data["best"] = {
                "epoch": int(epoch),
                "mean_f1": round(float(mean_f1), 4),
                "per_class": report,
            }
        self.save()

    # -- output --------------------------------------------------------------
    def finalise(self, checkpoint_path=None):
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        self.data["total_seconds"] = round(time.perf_counter() - self.started, 1)
        if checkpoint_path:
            self.data["checkpoint"] = str(checkpoint_path)
        self.save()
        return self.path

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))