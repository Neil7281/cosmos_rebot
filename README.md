# cosmos-rebot — Cosmos 3 Nano × reBot motor-box training

Post-train **Cosmos 3 Nano (16B)** on the `manavgoel4/molmo_act2_motor_box`
dataset to build a world model for the reBot B601 motor-box pick-and-place task.

---

## The bug this repo fixes

From the dataset README (verbatim):

> *"the `observation.images.wrist` and `observation.images.top` streams were
> **swapped** during data collection due to a collection-time labeling error.
> This dataset therefore requires post-processing to swap those two camera
> streams back before training or analysis."*

`scripts/00_fix_cameras.py` renames the video directories and writes a marker
file so the fix is idempotent.

---

## Step-by-step: training the model

Run these in order on the GPU machine. Every step is idempotent, so if
something fails partway through you can fix it and re-run from the top —
already-completed steps are skipped automatically.

### Step 0 — Clone and configure

```bash
git clone https://github.com/Neil7281/cosmos_rebot && cd cosmos_rebot

# Install the bootstrap pip packages (huggingface_hub, uv, pandas, pyarrow)
pip install -r requirements.txt

# Configure
cp .env.example .env
#   → edit .env and set N_GPUS (≥ 2) and CUDA_GROUP (cu128 or cu130)
```

### Step 1 — Download the dataset (~300 MB)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='manavgoel4/molmo_act2_motor_box',
    repo_type='dataset',
    local_dir='./workdir/molmo_act2_motor_box',
)
"
```

### Step 2 — Fix the camera-swap bug

```bash
python scripts/00_fix_cameras.py ./workdir/molmo_act2_motor_box
```

### Step 3 — Convert LeRobot v3 → cosmos-framework JSONL

```bash
pip install pandas pyarrow --quiet
python scripts/01_prepare_dataset.py \
    --dataset-dir ./workdir/molmo_act2_motor_box \
    --output-dir  ./workdir/cosmos_dataset

export DATASET_PATH=./workdir/cosmos_dataset
```

### Step 4 — Clone and install NVIDIA/cosmos-framework

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/cosmos-framework.git ./workdir/cosmos-framework
cd ./workdir/cosmos-framework

# Install uv if you don't already have it
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Install training dependencies (use cu128-train or cu130-train to match your drivers)
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --group=cu128-train
cd -
```

### Step 5 — Download Cosmos 3 Nano weights and convert to DCP (~30 GB, 30–60 min)

```bash
./workdir/cosmos-framework/.venv/bin/python -m cosmos_framework.scripts.convert_model_to_dcp \
    -o ./workdir/checkpoints/Cosmos3-Nano \
    --checkpoint-path Cosmos3-Nano

export BASE_CHECKPOINT_PATH=./workdir/checkpoints/Cosmos3-Nano
```

### Step 6 — Download the Wan 2.2 VAE / video tokenizer (~5 GB)

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Wan-AI/Wan2.2-TI2V-5B',
    filename='Wan2.2_VAE.pth',
    local_dir='./workdir/checkpoints/wan22_vae',
)
"

export WAN_VAE_PATH=./workdir/checkpoints/wan22_vae/Wan2.2_VAE.pth
```

### Step 7 — Launch Vision SFT training

```bash
cd ./workdir/cosmos-framework

.venv/bin/torchrun \
    --standalone \
    --nproc_per_node="$N_GPUS" \
    --master_port=29500 \
    -m cosmos_framework.scripts.train \
    --sft-toml=../../examples/toml/sft_config/rebot_motor_box_nano.toml
```

Checkpoints land in `./workdir/cosmos-framework/outputs/rebot_motor_box_nano/`.

---

### Shortcut — run all 7 steps at once

```bash
bash scripts/launch_training.sh
```

This runs Steps 1–7 above in order and skips anything already completed —
handy for first-time setup or for resuming after an interruption.

Or use **VS Code Tasks** (`Ctrl+Shift+B`) to run individual steps from the editor.

---

## Hardware requirements

| Setup | VRAM | ~Time (5 000 steps) |
|---|---|---|
| 2 × A100 80 GB | 160 GB | 10–14 h |
| 4 × A100 80 GB | 320 GB | 5–7 h |
| 8 × H100 80 GB | 640 GB | 2–3 h |

Minimum: **2 × A100 80 GB** (Cosmos 3 Nano is 16B params).

---

## Folder layout

```
cosmos_rebot/
├── .vscode/
│   ├── settings.json      VS Code Python interpreter + format-on-save
│   ├── tasks.json         Ctrl+Shift+B tasks for each step
│   ├── launch.json        Debug configs for each script
│   └── extensions.json    Recommended extensions
│
├── scripts/
│   ├── 00_fix_cameras.py      Fix camera-swap bug
│   ├── 01_prepare_dataset.py  Convert LeRobot v3 → cosmos JSONL
│   ├── launch_training.sh     One-shot full pipeline (Steps 1–7)
│   └── 03_run_inference.py    Call trained model at inference time
│
├── examples/
│   └── toml/sft_config/
│       └── rebot_motor_box_nano.toml   Training config
│
├── workdir/                   Created by the launch script (gitignored)
│   ├── molmo_act2_motor_box/  Downloaded dataset
│   ├── cosmos_dataset/        Converted JSONL + videos
│   ├── cosmos-framework/      Cloned + installed framework (with .venv)
│   └── checkpoints/           Cosmos3-Nano DCP + Wan2.2 VAE
│
├── .env.example               Copy to .env and fill in
├── requirements.txt           Bootstrap pip packages
└── README.md                  This file
```

---

## Dataset facts (from meta/stats.json)

| | |
|---|---|
| Robot | seeed_b601_dm_follower |
| Episodes | 50 |
| Frames | 8 648 @ 10 Hz |
| Action dim | 7 (degrees) |
| Joint names | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper |
| Cameras | wrist + top (swap fixed by `00_fix_cameras.py`) |

### Action ranges (raw degrees from stats.json)

```
shoulder_pan   [-43.5°,   40.2°]   mean -3.8°   std 14.4°
shoulder_lift  [-170.0°,   0.3°]   mean -71.8°  std 49.0°
elbow_flex     [-199.2°,   1.0°]   mean -58.2°  std 45.0°
wrist_flex     [-71.8°,   89.2°]   mean +19.0°  std 22.6°
wrist_yaw      [-42.4°,   17.8°]   mean -11.1°  std  9.2°
wrist_roll     [-62.8°,   10.1°]   mean -18.4°  std 17.0°
gripper        [-270.0°,   0.0°]   mean -84.4°  std 105°   ← continuous, NOT binary
```

The gripper uses raw motor encoder degrees — **not** 0/1. 0° = open, -270° = closed.

---

## After training — inference server

```bash
# Start the server (one GPU is enough for inference)
cd ./workdir/cosmos-framework
.venv/bin/python -m cosmos_framework.scripts.inference \
    --model-path outputs/rebot_motor_box_nano/latest \
    --port 8080

# In a second terminal — 
get a 16-step action chunk from a camera frame
python scripts/03_run_inference.py \
    --mode  policy \
    --image ./workdir/frame.png
```

Output:
```
step   shoulder_pan  shoulder_lift  elbow_flex  wrist_flex  wrist_yaw  wrist_roll  gripper
   0       -9.6123      -71.8340    -58.2100     19.4200    -11.3200    -18.7100  -84.3300
   1       -9.5800      -71.5100    -58.0200     19.6700    -11.1800    -18.5200  -85.1200
  ...
```

---

## What's next (action SFT recipe)

This package runs **Vision SFT** — it trains Cosmos 3 as a world model on the
robot's video demonstrations.  NVIDIA's cosmos-framework (launched May 31 2026)
currently ships Vision SFT recipes only.  A dedicated action post-training
recipe (training on `(video, joint_positions)` pairs in policy mode) is
expected soon in `cookbooks/cosmos3/generator/action/`.

Watch: https://github.com/NVIDIA/cosmos

When it lands, the only change here is swapping the JSONL format to include
per-frame action sequences and using the `action_nano` TOML — the rest of the
pipeline stays identical.
