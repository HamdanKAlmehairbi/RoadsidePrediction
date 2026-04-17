# HPC Ablation Campaign

## What this runs (~10 hours on 4 GPUs)

| Phase | What | Training | Eval | Est. Time |
|-------|------|:--------:|:----:|:---------:|
| 1 | Train 8 RL trainers on grid-7x7 | 8 runs (4 parallel) | -- | ~3 hr |
| 2 | Evaluate grid-7x7 (10 trainers x 5 seeds) | -- | 50 trials | ~30 min |
| 3 | Demand sweep (150/600 VPLPH, all 4 topos) | -- | 80 configs | ~1 hr |
| 4 | Fed step sweep (fs=3,5,10) | 18 runs | 18 configs | ~1.5 hr |
| 5 | Alpha sweep (a=0.1,0.3,0.7) | 18 runs | 18 configs | ~1.5 hr |
| 6 | FedProx mu sweep (mu=0.01,0.1,1.0) | 6 runs | 6 configs | ~30 min |
| | **Total** | | | **~10 hr** |

## Prerequisites

- conda env with: ray, rllib, torch, sumo, libsumo
- SUMO installed and on PATH
- Existing weights from local training already in `example_weights/ICCPS/Final/`

## Setup

```bash
# Copy entire BackEnd/ to HPC
scp -r BackEnd/ user@hpc:/path/to/seal/BackEnd/

# SSH in
ssh user@hpc
cd /path/to/seal/BackEnd

# Activate environment
conda activate RoadsideVenv  # or your env name
```

## Run

```bash
# Full campaign (all 6 phases)
bash hpc/run_all.sh

# Or via SLURM
sbatch --job-name=seal_ablation \
       --time=24:00:00 \
       --gres=gpu:4 \
       --cpus-per-task=32 \
       --mem=64G \
       --output=logs/ablation_%j.out \
       hpc/run_all.sh
```

## Run individual phases

```bash
# Just grid-7x7 training
python scripts/train_missing_trainers.py \
    --trainers MARL SARL FedRL MeanField CTDE Gossip HierFed FedDistill \
    --topologies grid-7x7 --parallel 4 --num-gpus 1 --num-workers 8

# Just demand ablation
python hpc/run_ablation.py --ablation demand

# Just fed step ablation
python hpc/run_ablation.py --ablation fed_step

# All ablations
python hpc/run_ablation.py --ablation all
```

## Resumability

Everything is resumable:
- Training skips trainers whose `ranked.pkl` already exists
- Evaluation skips configs already in `results.json`
- If the job dies, just rerun the same command

## Output files to copy back

```bash
# Results (JSON)
results/campaigns/baseline_7x7/results.json
results/campaigns/ablation_demand/results.json
results/campaigns/ablation_fed_step/results.json
results/campaigns/ablation_alpha/results.json
results/campaigns/ablation_fedprox/results.json

# Weights (grid-7x7 + ablation variants)
example_weights/ICCPS/Final/*/grid-7x7/ranked.pkl
example_weights/ICCPS/Ablation/

# Logs
logs/ablation/
```
