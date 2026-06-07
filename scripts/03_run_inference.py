#!/usr/bin/env python3
"""
Call the trained Cosmos 3 Nano model via the inference server.

Start the server first (in a separate terminal):
  cd ./workdir/cosmos-framework
  .venv/bin/python -m cosmos_framework.scripts.inference \
      --model-path outputs/rebot_motor_box_nano/latest \
      --port 8080

Then run this script:
  python scripts/03_run_inference.py \\
      --mode  policy \\
      --image ./workdir/frame.png \\
      --task  "Pick up the motor box and place it in the target location"

Returns a JSON action chunk with shape [16, 7]:
  16 timesteps × 7 joint angles (degrees)
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper]

Note on gripper encoding:
  The gripper column uses raw degrees in [-270°, 0°].
  0° = fully open, -270° = fully closed.
  Convert to reBot CAN command with your motor's degree-per-count ratio.

Modes:
  policy            current frame + task  →  next 16-step action chunk
  inverse_dynamics  start frame + end frame  →  actions that produced the motion
"""

import argparse
import base64
import json
import time
from pathlib import Path

import requests

SERVER_URL = "http://localhost:8080"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]


def encode_image(path: str) -> str:
    data = Path(path).read_bytes()
    ext  = Path(path).suffix.lstrip(".")
    b64  = base64.b64encode(data).decode()
    return f"data:image/{ext};base64,{b64}"


def predict_action(
    mode: str,
    image_path: str,
    task: str,
    action_chunk_size: int = 16,
    raw_action_dim: int = 7,
) -> list[list[float]]:
    payload = {
        "model_mode":        mode,
        "domain_name":       "bridge_orig_lerobot",
        "vision_path":       encode_image(image_path),
        "prompt":            task,
        "image_size":        480,
        "fps":               10,
        "action_chunk_size": action_chunk_size,
        "raw_action_dim":    raw_action_dim,
        "num_steps":         30,
        "guidance":          1.0,
        "shift":             5.0,
        "seed":              0,
    }

    # Policy and inverse_dynamics both go through the async jobs endpoint
    resp = requests.post(f"{SERVER_URL}/v1/videos", json=payload, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]

    for _ in range(120):
        r      = requests.get(f"{SERVER_URL}/v1/videos/{job_id}", timeout=10)
        result = r.json()
        if result.get("status") == "completed":
            return result["sample_outputs"][0]["content"]["action"]
        time.sleep(1.0)

    raise TimeoutError(f"Inference job {job_id} did not complete in 120 s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["policy", "inverse_dynamics"], default="policy")
    parser.add_argument("--image", required=True, help="Wrist camera frame (PNG or JPG)")
    parser.add_argument("--task",  default="Pick up the motor box and place it in the target location")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--json",  action="store_true", help="Print raw JSON instead of table")
    args = parser.parse_args()

    print(f"Sending request (mode={args.mode}) …")
    actions = predict_action(
        mode             = args.mode,
        image_path       = args.image,
        task             = args.task,
        action_chunk_size= args.chunk_size,
    )

    if args.json:
        print(json.dumps(actions, indent=2))
        return

    print(f"\nAction chunk — {len(actions)} steps × {len(actions[0])} DoF (degrees)\n")
    header = f"{'step':>4}  " + "  ".join(f"{n:>14}" for n in JOINT_NAMES)
    print(header)
    print("─" * len(header))
    for i, step in enumerate(actions):
        vals = "  ".join(f"{v:>14.4f}" for v in step)
        print(f"{i:>4}  {vals}")


if __name__ == "__main__":
    main()
