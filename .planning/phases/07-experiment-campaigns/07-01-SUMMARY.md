---
phase: 07-experiment-campaigns
plan: 01
subsystem: backend-evaluation
tags: [campaign, config, experiment-runner, fedrl, baseline-reproduction]
dependency_graph:
  requires:
    - api/evaluation/monte_carlo.py (MCConfig, run_monte_carlo)
    - api/training_runner.py (create_trainer, run_training_loop)
    - api/evaluation/store.py (save_evaluation pattern)
    - example_weights/ICCPS/Final/ (pre-trained paper weights)
  provides:
    - ExtensionConfig dataclass (all experiment parameters including Phase 4 extensions)
    - CampaignResult dataclass (train-then-evaluate result container)
    - EXAMPLE_WEIGHTS_MAP (6 entries: FedRL/MARL/SARL x grid-3x3/grid-5x5)
    - resolve_example_weights() helper
    - run_campaign.py CLI (baseline paper reproduction)
  affects:
    - Future ablation scripts (07-02 will import ExtensionConfig)
    - Future chart generation (07-03 will consume CampaignResult JSON)
tech_stack:
  added: []
  patterns:
    - dataclasses.asdict() for config serialization
    - sys.path.insert() for script-mode imports (no FastAPI server)
    - json.dumps(default=str) for safe serialization of arbitrary types
    - try/except around full campaign with error capture in CampaignResult.error
key_files:
  created:
    - BackEnd/api/evaluation/campaign_config.py
    - BackEnd/scripts/run_campaign.py
    - BackEnd/tests/test_campaign.py
  modified: []
decisions:
  - "EXAMPLE_WEIGHTS_MAP uses actual .pkl filenames (v3_naive-aggr_ranked.pkl for FedRL, v3_ranked.pkl for MARL/SARL) not the generic resolve_weights_path() naming — bypasses naming convention mismatch"
  - "Non-RL baselines (fixed-time, max-pressure) use sentinel weights_path='__baseline__' and create MCConfig directly — they don't go through train_and_evaluate main path"
  - "result_to_dict() round-trips through json.loads/json.dumps with default=str for safety — handles MCAggregatedResult which contains TrialMetrics dataclasses"
  - "scripts/ directory contains run_campaign.py without __init__.py — scripts are standalone CLI tools, not importable package members"
metrics:
  duration: 2m 19s
  completed_date: "2026-03-23T04:41:30Z"
  tasks_completed: 2
  files_created: 3
  tests_passed: 5
---

# Phase 07 Plan 01: Campaign Config and Runner Summary

**One-liner:** Typed campaign infrastructure with ExtensionConfig/CampaignResult dataclasses, 6-entry example weights map, and a train-then-evaluate CLI runner for SEAL paper baseline reproduction.

## What Was Built

### BackEnd/api/evaluation/campaign_config.py
- `ExtensionConfig` dataclass: captures all experiment parameters including Phase 4 extensions (fedprox_mu, alpha, time_of_day, use_time_encoding)
- `CampaignResult` dataclass: holds training rewards, MC evaluation result, weights path, error, and duration
- `EXAMPLE_WEIGHTS_MAP`: maps 6 (trainer, topology) tuples to actual paper-original .pkl paths under example_weights/ICCPS/Final/
- `resolve_example_weights(trainer, topology)`: returns absolute path or None (e.g. for grid-7x7 which has no example weights)
- `config_to_dict()` / `result_to_dict()`: JSON-safe serialization helpers

### BackEnd/scripts/run_campaign.py
- `train_and_evaluate(config)`: full train-then-evaluate pipeline; skips training if weights_path provided
- `run_baseline_campaign(topologies, n_eval_runs, dry_run_seeds)`: paper reproduction using example weights; includes fixed-time and max-pressure baselines
- `save_campaign_results(results, campaign_name)`: writes results.json + config.json to BackEnd/results/campaigns/
- argparse CLI: `--campaign`, `--topologies`, `--n-eval-runs`, `--dry-run`, `--output-name`

### BackEnd/tests/test_campaign.py
5 pure unit tests (no SUMO, no Ray):
- `test_extension_config_defaults` — verifies default values
- `test_example_weights_map_paths_exist` — verifies all 6 .pkl files exist on disk
- `test_resolve_example_weights_found` — verifies correct filename returned
- `test_resolve_example_weights_missing` — verifies None for grid-7x7
- `test_config_to_dict_roundtrip` — verifies all expected keys present

## Verification Results

```
5 passed in 0.28s
imports OK (from scripts.run_campaign import train_and_evaluate, run_baseline_campaign, save_campaign_results)
EXAMPLE_WEIGHTS_MAP — 6 entries confirmed
def train_and_evaluate — function confirmed
```

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all data flows are wired. The campaign runner produces real JSON output from real MC evaluations. Example weights map points to actual existing .pkl files on disk.

## Self-Check: PASSED
