#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/launch_training.sh
#
# One-shot script — runs the full pipeline from download to torchrun.
#
#   Step 1  Download dataset from HuggingFace
#   Step 2  Fix the camera-swap bug (documented in the dataset README)
#   Step 3  Convert LeRobot v3 → cosmos-framework JSONL format
#   Step 4  Clone NVIDIA/cosmos-framework + install via uv
#   Step 5  Download Cosmos 3 Nano weights + convert to DCP
#   Step 6  Download Wan 2.2 VAE (video tokeniser)
#   Step 7  Launch Vision SFT with torchrun
#
# Requirements before first run:
#   pip install huggingface_hub uv pandas pyarrow
#
# Usage:
#   cd /path/to/cosmos_rebot
#   cp .env.example .env        # fill in N_GPUS and CUDA_GROUP
#   bash scripts/launch_training.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Load .env if present ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
  echo "Loading $PROJECT_ROOT/.env"
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi

# ── Config (override via .env or shell exports) ───────────────────────────────
N_GPUS="${N_GPUS:-2}"
CUDA_GROUP="${CUDA_GROUP:-cu128}"
WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/workdir}"

# ── Derived paths ─────────────────────────────────────────────────────────────
DATASET_LOCAL="$WORK_DIR/molmo_act2_motor_box"
COSMOS_DATA="$WORK_DIR/cosmos_dataset"
FRAMEWORK_DIR="$WORK_DIR/cosmos-framework"
CKPT_DIR="$WORK_DIR/checkpoints"
BASE_CHECKPOINT_PATH="$CKPT_DIR/Cosmos3-Nano"
WAN_VAE_PATH="$CKPT_DIR/wan22_vae/Wan2.2_VAE.pth"
TOML_FILE="$PROJECT_ROOT/examples/toml/sft_config/rebot_motor_box_nano.toml"

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN]  $*${NC}"; }
die()  { echo -e "${RED}[ERROR] $*${NC}"; exit 1; }

mkdir -p "$WORK_DIR" "$CKPT_DIR"

# ── Step 1 — Download dataset ─────────────────────────────────────────────────
if [ -f "$DATASET_LOCAL/meta/info.json" ]; then
  log "Step 1  Dataset already at $DATASET_LOCAL — skipping download"
else
  log "Step 1  Downloading manavgoel4/molmo_act2_motor_box (~300 MB) …"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='manavgoel4/molmo_act2_motor_box',
    repo_type='dataset',
    local_dir='$DATASET_LOCAL',
    ignore_patterns=['*.gitattributes'],
)
print('Download complete.')
"
fi

# ── Step 2 — Fix camera swap ──────────────────────────────────────────────────
log "Step 2  Applying camera-swap fix …"
python "$PROJECT_ROOT/scripts/00_fix_cameras.py" "$DATASET_LOCAL"

# ── Step 3 — Convert to JSONL ─────────────────────────────────────────────────
if [ -f "$COSMOS_DATA/train/video_dataset_file.jsonl" ]; then
  log "Step 3  JSONL already at $COSMOS_DATA — skipping"
else
  log "Step 3  Converting dataset to cosmos-framework JSONL format …"
  pip install pandas pyarrow --quiet
  python "$PROJECT_ROOT/scripts/01_prepare_dataset.py" \
    --dataset-dir "$DATASET_LOCAL" \
    --output-dir  "$COSMOS_DATA"
fi
export DATASET_PATH="$COSMOS_DATA"

# ── Step 4 — Clone cosmos-framework and install ───────────────────────────────
if [ -d "$FRAMEWORK_DIR/.venv" ]; then
  log "Step 4  cosmos-framework already installed — skipping"
else
  log "Step 4  Cloning NVIDIA/cosmos-framework …"
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/cosmos-framework.git "$FRAMEWORK_DIR"
  cd "$FRAMEWORK_DIR"

  if ! command -v uv &>/dev/null; then
    log "  Installing uv …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
  fi

  log "  Installing dependencies (group=${CUDA_GROUP}-train) …"
  GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --group="${CUDA_GROUP}-train"
  cd - > /dev/null
fi

PYTHON="$FRAMEWORK_DIR/.venv/bin/python"
TORCHRUN="$FRAMEWORK_DIR/.venv/bin/torchrun"

# ── Step 5 — Download + convert Cosmos 3 Nano to DCP ─────────────────────────
if [ -d "$BASE_CHECKPOINT_PATH" ] && [ "$(ls -A "$BASE_CHECKPOINT_PATH" 2>/dev/null)" ]; then
  log "Step 5  Cosmos3-Nano DCP already at $BASE_CHECKPOINT_PATH — skipping"
else
  log "Step 5  Downloading Cosmos3-Nano and converting to DCP (~30 GB, 30-60 min) …"
  "$PYTHON" -m cosmos_framework.scripts.convert_model_to_dcp \
    -o "$BASE_CHECKPOINT_PATH" \
    --checkpoint-path Cosmos3-Nano
fi
export BASE_CHECKPOINT_PATH

# ── Step 6 — Download Wan 2.2 VAE ────────────────────────────────────────────
if [ -f "$WAN_VAE_PATH" ]; then
  log "Step 6  Wan 2.2 VAE already at $WAN_VAE_PATH — skipping"
else
  log "Step 6  Downloading Wan 2.2 VAE (~5 GB) …"
  mkdir -p "$(dirname "$WAN_VAE_PATH")"
  python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Wan-AI/Wan2.2-TI2V-5B',
    filename='Wan2.2_VAE.pth',
    local_dir='$(dirname "$WAN_VAE_PATH")',
)
print('VAE download complete.')
"
fi
export WAN_VAE_PATH

# ── Step 7 — Launch training ──────────────────────────────────────────────────
log "Step 7  Launching Vision SFT on $N_GPUS GPU(s) …"
echo ""
echo "  DATASET_PATH         = $DATASET_PATH"
echo "  BASE_CHECKPOINT_PATH = $BASE_CHECKPOINT_PATH"
echo "  WAN_VAE_PATH         = $WAN_VAE_PATH"
echo "  TOML                 = $TOML_FILE"
echo "  GPUs                 = $N_GPUS"
echo ""

cd "$FRAMEWORK_DIR"

"$TORCHRUN" \
  --standalone \
  --nproc_per_node="$N_GPUS" \
  --master_port=29500 \
  -m cosmos_framework.scripts.train \
  --sft-toml="$TOML_FILE"

log "✓  Training complete."
log "   Checkpoints saved to: $FRAMEWORK_DIR/outputs/rebot_motor_box_nano/"

# ── Step 8 — Push checkpoint to Hugging Face Hub (optional) ──────────────────
if [ -n "${HF_PUSH_REPO:-}" ]; then
  CKPT_OUT_DIR="$FRAMEWORK_DIR/outputs/rebot_motor_box_nano/latest"
  log "Step 8  Pushing checkpoint to https://huggingface.co/$HF_PUSH_REPO …"
  [ -n "${HF_TOKEN:-}" ] || die "HF_PUSH_REPO is set but HF_TOKEN is missing — add it to .env"
  [ -d "$CKPT_OUT_DIR" ] || die "Checkpoint dir not found: $CKPT_OUT_DIR"

  HF_PUSH_REPO="$HF_PUSH_REPO" HF_TOKEN="$HF_TOKEN" CKPT_OUT_DIR="$CKPT_OUT_DIR" python -c "
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = os.environ['HF_PUSH_REPO']

api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True)
api.upload_folder(
    folder_path=os.environ['CKPT_OUT_DIR'],
    repo_id=repo_id,
    repo_type='model',
    commit_message='Upload rebot_motor_box_nano checkpoint',
)
print(f'Uploaded to https://huggingface.co/{repo_id}')
"
  log "✓  Push complete — https://huggingface.co/$HF_PUSH_REPO"
else
  log "Step 8  HF_PUSH_REPO not set — skipping Hub push"
fi
