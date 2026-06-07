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

## Quick start

```bash
# 1. Clone / download this folder and open it in VS Code

# 2. Install the bootstrap pip packages on the GPU machine
pip install huggingface_hub uv pandas pyarrow

# 3. Configure
cp .env.example .env
#   → set N_GPUS and CUDA_GROUP in .env

# 4. Run everything (download → fix → convert → install → train)
bash scripts/launch_training.sh
```

That's it. The script is re-entrant — re-running it skips already-completed
steps.

Or use **VS Code Tasks** (`Ctrl+Shift+B`) to run individual steps.

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

# In a second terminal — get a 16-step action chunk from a camera frame
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
