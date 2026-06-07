#!/usr/bin/env python3
"""
Convert manavgoel4/molmo_act2_motor_box (LeRobot v3) to the JSONL format
that cosmos-framework vision SFT expects.

cosmos-framework requires:
  $DATASET_PATH/
    train/
      video_dataset_file.jsonl   ← one JSON object per line
    videos/
      episode_0000.mp4
      episode_0001.mp4
      ...

Each JSONL line:
  {"video": "videos/episode_NNNN.mp4", "text": "<task description>"}

Dataset facts (from meta/stats.json):
  • 50 episodes, 8 648 frames @ 10 Hz
  • 7-DoF actions in degrees: [shoulder_pan, shoulder_lift, elbow_flex,
      wrist_flex, wrist_yaw, wrist_roll, gripper]
  • Action ranges (from stats.json):
      shoulder_pan   [-43.5°,  40.2°]
      shoulder_lift  [-170°,    0.3°]
      elbow_flex     [-199°,    1.0°]
      wrist_flex     [-71.8°,  89.2°]
      wrist_yaw      [-42.4°,  17.8°]
      wrist_roll     [-62.8°,  10.1°]
      gripper        [-270°,    0.0°]  ← continuous degrees, NOT binary!
  • Mean / std for z-score normalisation available in meta/stats.json

Usage:
  python scripts/01_prepare_dataset.py \\
      --dataset-dir ./workdir/molmo_act2_motor_box \\
      --output-dir  ./workdir/cosmos_dataset
  # or via VS Code Tasks → "Prepare Dataset Only"
"""

import argparse
import json
import shutil
from pathlib import Path


def read_task(dataset_dir: Path) -> str:
    """Pull the task string from meta/tasks.parquet."""
    tasks_parquet = dataset_dir / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(tasks_parquet)
            if "task" in df.columns:
                task = str(df["task"].iloc[0]).strip()
                if task and task.lower() not in {"nan", "none", ""}:
                    print(f"  Task (from tasks.parquet): «{task}»")
                    return task
        except Exception as e:
            print(f"  [warn] could not parse tasks.parquet: {e}")

    fallback = "Pick up the motor box and place it in the target location using the robot arm."
    print(f"  Task (fallback): «{fallback}»")
    return fallback


def find_episode_videos(dataset_dir: Path, camera_key: str) -> list[Path]:
    """
    Return sorted list of episode mp4 files under videos/<camera_key>/.
    LeRobot v3: one mp4 per episode, under chunk-NNN/file-NNN.mp4
    """
    cam_dir = dataset_dir / "videos" / camera_key
    if not cam_dir.exists():
        raise FileNotFoundError(
            f"Camera directory not found: {cam_dir}\n"
            f"  → Run 00_fix_cameras.py first!"
        )
    mp4s = sorted(cam_dir.rglob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No mp4 files found under {cam_dir}")
    return mp4s


def build_jsonl(dataset_dir: Path, output_dir: Path) -> None:
    dataset_dir = dataset_dir.resolve()
    output_dir  = output_dir.resolve()

    if not (dataset_dir / ".camera_fix_applied").exists():
        print("\n⚠  WARNING: camera fix not detected. Run 00_fix_cameras.py first.\n")

    info           = json.loads((dataset_dir / "meta" / "info.json").read_text())
    total_episodes = info["total_episodes"]
    print(f"Dataset  : {total_episodes} episodes, {info['total_frames']} frames @ {info['fps']} Hz")

    # After the camera fix, wrist = actual wrist camera
    camera_key = "observation.images.wrist"
    print(f"Camera   : {camera_key}")

    episode_mp4s = find_episode_videos(dataset_dir, camera_key)
    print(f"MP4 files: {len(episode_mp4s)}")

    task = read_task(dataset_dir)

    # ── output layout ──────────────────────────────────────────────────────────
    train_dir  = output_dir / "train"
    videos_dst = output_dir / "videos"
    train_dir.mkdir(parents=True, exist_ok=True)
    videos_dst.mkdir(parents=True, exist_ok=True)

    # ── copy videos + build JSONL ─────────────────────────────────────────────
    jsonl_path = train_dir / "video_dataset_file.jsonl"
    records    = []

    for ep_idx, src_mp4 in enumerate(episode_mp4s[:total_episodes]):
        dst_name = f"episode_{ep_idx:04d}.mp4"
        dst_mp4  = videos_dst / dst_name

        if not dst_mp4.exists():
            shutil.copy2(src_mp4, dst_mp4)
            print(f"  copied episode {ep_idx:02d}  →  {dst_name}", end="\r")

        records.append({"video": f"videos/{dst_name}", "text": task})

    print(f"\n  {len(records)} episodes copied")

    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n✓  JSONL  : {jsonl_path}")
    print(f"✓  Videos : {videos_dst}")
    print(f"\n  Set before training:")
    print(f"    export DATASET_PATH={output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True,
                        help="Path to the downloaded molmo_act2_motor_box dataset")
    parser.add_argument("--output-dir", required=True,
                        help="Destination for the cosmos-framework JSONL dataset root")
    args = parser.parse_args()

    build_jsonl(Path(args.dataset_dir), Path(args.output_dir))
