# Football action spotter

Match video in, highlight reel out. Frozen pretrained backbone for frame
features; a small temporal model you own and can read in one sitting.

No SoccerNet devkit, no detection, no tracking, no homography.

## What's here

| File | Role |
|---|---|
| `config.py` | Every tunable in one place |
| `annotate.html` | Browser annotation tool — open it, no install |
| `check_labels.py` | Catch annotation errors before they cost you a training run |
| `extract_features.py` | Video → cached frame features (run once per match) |
| `dataset.py` | Label CSV → training windows with soft targets |
| `model.py` | The temporal model — **this is the part you own** |
| `train.py` | Training loop with class-weighted loss |
| `postprocess.py` | Probability curve → discrete events, plus evaluation |
| `spot.py` | Trained model → events JSON |
| `make_reel.py` | Events JSON → cut and concatenated reel |
| `smoke_test.py` | Validate the pipeline with no video at all |

Roughly 700 lines total. The only third-party pieces are PyTorch, OpenCV as a
video decoder, and one ImageNet-pretrained ResNet used as a fixed function.

## Setup

```bash
pip install torch torchvision opencv-python numpy
# ffmpeg must be on PATH for make_reel.py
```

## Verify before you invest

```bash
python smoke_test.py
python train.py --features _smoke/features --labels _smoke/labels.csv \
    --train-videos m1 m2 m3 --val-videos val1 --epochs 15 --out _smoke/checkpoints
```

Expect `val_f1` above 0.85 within about ten epochs. This proves the plumbing
works on a signal that is trivially learnable — it says nothing about how hard
your real footage will be, but if it fails, the bug is in the code.

## Workflow

```bash
# 1. Cache features (slow, once per match)
python extract_features.py --video matches/liverpool_arsenal.mp4

# 2. Label events in labels.csv (see below)

# 3. Train — seconds per epoch, since features are cached
python train.py --train-videos m1 m2 m3 m4 --val-videos m5 m6

# 4. Spot events on a new match
python spot.py --video-id m7

# 5. Cut the reel
python make_reel.py --video matches/m7.mp4 --spots spots/m7.json --max-clips 12
```

## Labelling

The entire annotation format:

```csv
video_id,timestamp_s,label
liverpool_arsenal,1247.3,goal
liverpool_arsenal,2891.0,card
liverpool_arsenal,415.5,shot
```

`timestamp_s` is seconds into the **video file**, not match clock.

Open `annotate.html` in any browser and drag a match file onto it. Nothing is
uploaded — the browser reads the file locally. Skim at 4×, hit space when
something happens, step back, mark with `G`/`S`/`C`, nudge with `,`/`.` if you
were slightly off. Export gives you one CSV per match; concatenate them into a
single `labels.csv`, keeping one header row:

```bash
head -1 m1_labels.csv > labels.csv
tail -q -n +2 *_labels.csv >> labels.csv
python check_labels.py
```

`check_labels.py` catches timestamps past the end of a video, typo'd class
names, matches labelled but never feature-extracted, near-duplicate marks
(usually a replay logged as live), and matches with suspiciously few events. It
exits non-zero on errors, so you can gate a training script on it. The one
thing it cannot detect is a **missed** event — only a second pass catches those.

**Guidance that matters more than any hyperparameter:**

- **Three classes, not seventeen.** Every class multiplies your labelling
  burden, and rare classes will simply not be learned.
- **Aim for 15–20 matches.** Below about 10 the model memorises specific
  camera angles rather than learning what a goal looks like.
- **Be consistent about the anchor.** Pick a rule — "the moment of contact for
  a shot", "the ball crossing the line for a goal" — and never deviate. The
  model learns your convention; an inconsistent one is unlearnable.
- **Label everything of a class, or nothing.** A missed shot becomes a
  training example teaching the model that shots are not events.
- **Broadcast tip:** label from the live action, not the replay. Then check
  whether your replay suppression catches the duplicate.

## Reading the output

`train.py` reports precision, recall and F1 per class on held-out matches,
after peak picking. This is deliberately end-to-end: validation loss can
improve while the model gets worse at producing usable spots.

- **High recall, low precision** → too many false spots. Raise
  `PEAK_THRESHOLD`, or `NMS_GAP_S` if you're getting clusters around one event.
- **High precision, low recall** → missing events. Lower `PEAK_THRESHOLD` first;
  if that doesn't help, you need more labelled examples of that class.
- **One class much worse than others** → almost always a labelling volume
  problem, not a model problem.

For a highlight reel, precision matters more than recall. A missed shot costs
you nothing visible; a clip of a throw-in makes the reel look broken.

## Design notes

**Why a frozen backbone.** The ResNet does generic image understanding — edges,
textures, people, grass. Nothing football-specific, and nothing you'd improve
by retraining on 20 matches. Fine-tuning it with your data volume would overfit
immediately. It's a fixed feature extractor, and treating it that way is what
makes this project feasible.

**Why dilated convolutions.** Dilation doubles per block, so receptive field
grows exponentially with depth: six blocks see 127 timesteps, about 64 seconds.
A shot isn't recognisable from a single frame — it needs the build-up before
and the reaction after. An LSTM would work too but trains more slowly and gives
you less control over exactly how much context each prediction sees.

**Why soft targets.** Each event becomes a Gaussian bump rather than one hot
frame. Football events occupy seconds, not instants, and a hard target punishes
the model for being half a second early. `SIGMA_STEPS` in `config.py` controls
the width.

**Why `pos_weight` and the negative output bias.** Events are perhaps 0.5% of
timesteps. Without both of these, the loss is dominated by easy negatives and
the model converges to predicting nothing — the same class-imbalance failure
that defeats naive ball detectors. The bias initialisation of `-4.0` means the
model *starts* at "nothing is happening" so early gradients go into learning
the events rather than learning the base rate.

## Where this goes next

`spots/<video>.json` is already the beginning of the event stream your
commentary layer will consume. It has timestamps, labels and confidences. What
it lacks is actors and location — that's what the detection and tracking work
would add, and it's the point at which "who" and "where" become available.
