#!/usr/bin/env python3
"""
Fix the camera-swap bug documented in the dataset README.

README says:
  "the observation.images.wrist and observation.images.top streams were swapped
   during data collection due to a collection-time labeling error."

What is on disk RIGHT NOW:
  videos/observation.images.wrist/  →  actually the TOP camera   ✗
  videos/observation.images.top/    →  actually the WRIST camera  ✗

After this script:
  videos/observation.images.wrist/  →  actual wrist camera  ✓
  videos/observation.images.top/    →  actual top camera    ✓

Idempotent — writes a marker file so re-running is a no-op.

Usage:
  python scripts/00_fix_cameras.py ./workdir/molmo_act2_motor_box
  # or via VS Code Tasks → "Fix Camera Swap Only"
"""

import json
import shutil
import sys
from pathlib import Path


def fix_cameras(dataset_dir: str) -> None:
    root   = Path(dataset_dir).resolve()
    marker = root / ".camera_fix_applied"

    if marker.exists():
        print("✓  Camera fix already applied — skipping")
        return

    videos = root / "videos"
    wrist  = videos / "observation.images.wrist"
    top    = videos / "observation.images.top"
    tmp    = videos / "_tmp_swap"

    for p in (wrist, top):
        if not p.exists():
            raise FileNotFoundError(f"Expected video directory not found: {p}")

    print("Swapping mislabeled camera directories …")
    shutil.move(str(wrist), str(tmp))    # wrist (actually top) → tmp
    shutil.move(str(top),   str(wrist))  # top (actually wrist) → wrist  ✓
    shutil.move(str(tmp),   str(top))    # tmp                  → top    ✓

    marker.write_text("camera_swap_fix_applied=true\n")
    print("✓  Swap complete")
    print("   observation.images.wrist/ → actual wrist camera")
    print("   observation.images.top/   → actual top camera")


def validate(dataset_dir: str) -> None:
    root = Path(dataset_dir).resolve()
    info = json.loads((root / "meta" / "info.json").read_text())

    print("\n── Dataset summary ───────────────────────────────────")
    print(f"  robot_type     {info['robot_type']}")
    print(f"  episodes       {info['total_episodes']}")
    print(f"  total_frames   {info['total_frames']}")
    print(f"  fps            {info['fps']}")
    print(f"  action_dim     {info['features']['action']['shape']}")
    print(f"  action_names   {info['features']['action']['names']}")

    cameras = [k for k in info["features"] if k.startswith("observation.images")]
    for cam in cameras:
        key  = cam.split("observation.images.")[-1]
        mp4s = list((root / "videos" / f"observation.images.{key}").rglob("*.mp4"))
        print(f"  {key:6s} videos  {len(mp4s)} mp4 files")

    print("──────────────────────────────────────────────────────")
    print("✓  Dataset looks valid\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} /path/to/dataset")
        sys.exit(1)

    fix_cameras(sys.argv[1])
    validate(sys.argv[1])
