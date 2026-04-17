#!/bin/bash
# =============================================================================
# SEAL HPC Ablation Campaign — Master Script
#
# Runs everything needed on the HPC in ~10 hours (4 GPUs):
#   Phase 1: Train 8 RL trainers on grid-7x7 (new topology)
#   Phase 2: Evaluate all 10 trainers on grid-7x7 (10 seeds)
#   Phase 3: Cologne-8 extended training (200 episodes) + eval
#   Phase 4: Demand sweep (eval-only at 150/600 VPLPH, all 4 topologies)
#   Phase 5: Fed step ablation (retrain + eval, fed_step=3,5,10)
#   Phase 6: Alpha ablation (retrain + eval, alpha=0.1,0.3,0.7)
#   Phase 7: FedProx mu ablation (retrain + eval, mu=0.01,0.1,1.0)
#
# Prerequisites:
#   - conda env with Ray, RLlib, torch, sumo, libsumo
#   - SUMO on PATH
#   - Run from BackEnd/ directory
#   - Existing weights in example_weights/ICCPS/Final/ (grid-3x3, grid-5x5, cologne-8)
#
# Usage:
#   cd BackEnd
#   bash hpc/run_all.sh
#
# SLURM:
#   sbatch --job-name=seal_ablation \
#          --time=24:00:00 \
#          --gres=gpu:4 \
#          --cpus-per-task=32 \
#          --mem=64G \
#          --output=logs/ablation_%j.out \
#          hpc/run_all.sh
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
LOG_DIR="logs/ablation"
NUM_GPUS=1
PARALLEL_JOBS=4
NUM_WORKERS=8
N_EPISODES=50
N_EVAL_RUNS=5

cd "$(dirname "$0")/.."
mkdir -p "$LOG_DIR"

TS=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "SEAL HPC Ablation Campaign — $(date)"
echo "Working dir : $(pwd)"
echo "Python      : $($PYTHON --version 2>&1)"
echo "GPUs        : $PARALLEL_JOBS"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Train grid-7x7 (8 RL trainers)
# ---------------------------------------------------------------------------
echo "PHASE 1: Training 8 RL trainers on grid-7x7..."
echo "  Log: $LOG_DIR/phase1_train_7x7_${TS}.log"

$PYTHON scripts/train_missing_trainers.py \
    --trainers MARL SARL FedRL MeanField CTDE Gossip HierFed FedDistill \
    --topologies grid-7x7 \
    --n-episodes $N_EPISODES \
    --num-workers $NUM_WORKERS \
    --num-gpus $NUM_GPUS \
    --parallel $PARALLEL_JOBS \
    2>&1 | tee "$LOG_DIR/phase1_train_7x7_${TS}.log"

echo ""
echo "Phase 1 complete."
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Evaluate grid-7x7 (10 trainers x 5 seeds)
# ---------------------------------------------------------------------------
echo "PHASE 2: Evaluating all trainers on grid-7x7..."
echo "  Log: $LOG_DIR/phase2_eval_7x7_${TS}.log"

$PYTHON scripts/run_campaign.py \
    --campaign baseline \
    --topologies grid-7x7 \
    --n-eval-runs $N_EVAL_RUNS \
    --output-name baseline_7x7 \
    2>&1 | tee "$LOG_DIR/phase2_eval_7x7_${TS}.log"

echo ""
echo "Phase 2 complete."
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Cologne-8 extended training (200 episodes) + eval
# Tests whether RL can outperform baselines with more training
# ---------------------------------------------------------------------------
echo "PHASE 3: Cologne-8 extended training (200 episodes)..."
echo "  Log: $LOG_DIR/phase3_cologne_extended_${TS}.log"

$PYTHON hpc/run_ablation.py \
    --ablation cologne_extended \
    --num-workers $NUM_WORKERS \
    --num-gpus $NUM_GPUS \
    2>&1 | tee "$LOG_DIR/phase3_cologne_extended_${TS}.log"

echo ""
echo "Phase 3 complete."
echo ""

# ---------------------------------------------------------------------------
# Phase 4: Demand sweep (eval-only, 150/600 VPLPH)
# ---------------------------------------------------------------------------
echo "PHASE 4: Demand sensitivity sweep..."
echo "  Log: $LOG_DIR/phase4_demand_${TS}.log"

$PYTHON hpc/run_ablation.py \
    --ablation demand \
    --num-workers $NUM_WORKERS \
    --num-gpus $NUM_GPUS \
    2>&1 | tee "$LOG_DIR/phase4_demand_${TS}.log"

echo ""
echo "Phase 4 complete."
echo ""

# ---------------------------------------------------------------------------
# Phase 5: Fed step frequency sweep
# ---------------------------------------------------------------------------
echo "PHASE 5: Fed step ablation (fed_step=3,5,10)..."
echo "  Log: $LOG_DIR/phase5_fedstep_${TS}.log"

$PYTHON hpc/run_ablation.py \
    --ablation fed_step \
    --num-workers $NUM_WORKERS \
    --num-gpus $NUM_GPUS \
    2>&1 | tee "$LOG_DIR/phase5_fedstep_${TS}.log"

echo ""
echo "Phase 5 complete."
echo ""

# ---------------------------------------------------------------------------
# Phase 6: Alpha (cooperative reward) sweep
# ---------------------------------------------------------------------------
echo "PHASE 6: Alpha ablation (alpha=0.1,0.3,0.7)..."
echo "  Log: $LOG_DIR/phase6_alpha_${TS}.log"

$PYTHON hpc/run_ablation.py \
    --ablation alpha \
    --num-workers $NUM_WORKERS \
    --num-gpus $NUM_GPUS \
    2>&1 | tee "$LOG_DIR/phase6_alpha_${TS}.log"

echo ""
echo "Phase 6 complete."
echo ""

# ---------------------------------------------------------------------------
# Phase 7: FedProx mu sweep
# ---------------------------------------------------------------------------
echo "PHASE 7: FedProx mu ablation (mu=0.01,0.1,1.0)..."
echo "  Log: $LOG_DIR/phase7_fedprox_${TS}.log"

$PYTHON hpc/run_ablation.py \
    --ablation fedprox \
    --num-workers $NUM_WORKERS \
    --num-gpus $NUM_GPUS \
    2>&1 | tee "$LOG_DIR/phase7_fedprox_${TS}.log"

echo ""
echo "============================================================"
echo "ALL PHASES COMPLETE — $(date)"
echo ""
echo "Results:"
echo "  results/campaigns/baseline_7x7/results.json"
echo "  results/campaigns/ablation_cologne_extended/results.json"
echo "  results/campaigns/ablation_demand/results.json"
echo "  results/campaigns/ablation_fed_step/results.json"
echo "  results/campaigns/ablation_alpha/results.json"
echo "  results/campaigns/ablation_fedprox/results.json"
echo ""
echo "Weights:"
echo "  example_weights/ICCPS/Final/*/grid-7x7/ranked.pkl"
echo "  example_weights/ICCPS/Ablation/*/ranked.pkl"
echo ""
echo "Copy these back to your local machine:"
echo "  scp -r results/campaigns/ user@local:BackEnd/results/campaigns/"
echo "  scp -r example_weights/ICCPS/ user@local:BackEnd/example_weights/ICCPS/"
echo "============================================================"
