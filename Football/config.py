## Sampling
FPS = 2.0
FEATURE_DIM = 2048

## Classes
CLASSES = ["goal", "shot", "card"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


## Temporal Model
WINDOW = 128 #timesteps per training window (128/2fps = 64s)

HIDDEN = 256
DILATIONS = [1,2,4,8,16,32]
DROPOUT = 0.2


## Target Construction
SIGMA_STEPS = 3.0

## Training
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40
NEG_RATIO = 3

## Inference
STRIDE = WINDOW // 2
PEAK_THRESHOLD = 0.35

NMS_GAP_S = 25.0

## Clipping
PRE_ROLL_S = 8.0
POST_ROLL_S = 5.0
