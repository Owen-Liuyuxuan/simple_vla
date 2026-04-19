#!/bin/bash
# simple_vla/run.sh -- Quick inference launcher

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

: "${CHECKPOINT:=$PROJECT_ROOT/UniDriveVLA_Stage3_Nuscenes_2B.pt}"
: "${OCCWORLD_VAE_PATH:=$PROJECT_ROOT/occvae_latest.pth}"
: "${VLM_PRETRAINED_PATH:=owlvla/UniDriveVLA_Nusc_Base_Stage1}"
: "${NUM_SAMPLES:=4}"
: "${IMG_HEIGHT:=544}"
: "${IMG_WIDTH:=960}"

export CHECKPOINT OCCWORLD_VAE_PATH VLM_PRETRAINED_PATH NUM_SAMPLES IMG_HEIGHT IMG_WIDTH

if [ -d "$PROJECT_ROOT/.venv" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "=============================================="
echo "  simple_vla inference"
echo "=============================================="
echo "  Checkpoint:    $CHECKPOINT"
echo "  VLM weights:   $VLM_PRETRAINED_PATH"
echo "  VAE weights:   $OCCWORLD_VAE_PATH"
echo "  Config:        $SCRIPT_DIR/configs/simple_inference_stage2_2b.py"
echo "  Samples:       $NUM_SAMPLES"
echo "  Image size:    ${IMG_WIDTH}x${IMG_HEIGHT}"
echo "  Python:        $PYTHON"
echo "=============================================="

"$PYTHON" "$SCRIPT_DIR/inference.py"
