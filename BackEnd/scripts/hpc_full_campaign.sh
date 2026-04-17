#!/bin/bash
# =============================================================================
# hpc_full_campaign.sh — Full 10-trainer SEAL benchmark campaign for HPC
#
# Step 1: Train the 5 missing RL trainers (Gossip, HierFed, FedDistill,
#         MeanField, CTDE) on grid-3x3 and grid-5x5. Saves ranked.pkl to
#         example_weights/ICCPS/Final/{Trainer}/{topology}/.
#         Resumable — skips trainers that already have example weights.
#
# Step 2: Run the full 10-trainer evaluation campaign (100 runs per topology:
#         8 RL trainers + 2 baselines × 10 MC seeds).
#
# Usage:
#   bash scripts/hpc_full_campaign.sh
#
# SLURM example:
#   sbatch --job-name=seal_campaign \
#          --time=12:00:00 \
#          --cpus-per-task=8 \
#          --mem=32G \
#          --output=logs/seal_%j.out \
#          scripts/hpc_full_campaign.sh
#
# Requirements:
#   - conda environment RoadsideVenv activated (or PYTHON set below)
#   - SUMO on PATH (libsumo will activate automatically on Linux)
#   - Run from BackEnd/ directory
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these if needed
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python}"          # override with full path if needed
TOPOLOGIES="grid-3x3 grid-5x5"
N_EPISODES=50
N_EVAL_RUNS=10
CAMPAIGN_NAME="baseline"
LOG_DIR="logs"
NUM_GPUS=1                          # GPUs per training job
PARALLEL_JOBS=4                     # run this many training jobs simultaneously (= number of GPUs)
NUM_WORKERS=8                       # Ray rollout workers per job (set to ~CPU_cores / PARALLEL_JOBS)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.."             # ensure we're in BackEnd/
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRAIN_LOG="$LOG_DIR/train_missing_${TIMESTAMP}.log"
CAMPAIGN_LOG="$LOG_DIR/campaign_${TIMESTAMP}.log"

echo "============================================================"
echo "SEAL Full Campaign — $(date)"
echo "Working dir : $(pwd)"
echo "Python      : $($PYTHON --version 2>&1)"
echo "Topologies  : $TOPOLOGIES"
echo "Episodes    : $N_EPISODES"
echo "Eval runs   : $N_EVAL_RUNS"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Train missing trainers
# ---------------------------------------------------------------------------
echo "STEP 1: Training missing RL trainers..."
echo "  Log: $TRAIN_LOG"
echo ""

$PYTHON scripts/train_missing_trainers.py \
    --topologies $TOPOLOGIES \
    --n-episodes $N_EPISODES \
    --num-workers $NUM_WORKERS \
    --num-gpus $NUM_GPUS \
    --parallel $PARALLEL_JOBS \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}
if [ $TRAIN_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Training step failed (exit $TRAIN_EXIT). See $TRAIN_LOG"
    echo "Re-running this script will resume from where training stopped."
    exit $TRAIN_EXIT
fi

echo ""
echo "Training complete."
echo ""

# ---------------------------------------------------------------------------
# Step 2: Full evaluation campaign
# ---------------------------------------------------------------------------
echo "STEP 2: Running full 10-trainer evaluation campaign..."
echo "  Log: $CAMPAIGN_LOG"
echo ""

$PYTHON scripts/run_campaign.py \
    --campaign "$CAMPAIGN_NAME" \
    --topologies $TOPOLOGIES \
    --n-eval-runs $N_EVAL_RUNS \
    2>&1 | tee "$CAMPAIGN_LOG"

CAMPAIGN_EXIT=${PIPESTATUS[0]}
if [ $CAMPAIGN_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Campaign step failed (exit $CAMPAIGN_EXIT). See $CAMPAIGN_LOG"
    exit $CAMPAIGN_EXIT
fi

echo ""
echo "============================================================"
echo "CAMPAIGN COMPLETE — $(date)"
echo "Results: results/campaigns/${CAMPAIGN_NAME}/results.json"
echo "Logs   : $TRAIN_LOG"
echo "         $CAMPAIGN_LOG"
echo "============================================================"
