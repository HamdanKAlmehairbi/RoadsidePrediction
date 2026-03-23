---
phase: 07-experiment-campaigns
plan: 02
subsystem: backend-evaluation
tags: [ablation, fedprox, cooperative-reward, time-of-day, experiment-campaigns]
dependency_graph:
  requires:
    - api/evaluation/campaign_config.py (ExtensionConfig, CampaignResult)
    - scripts/run_campaign.py (train_and_evaluate, save_campaign_results)
  provides:
    - run_extension_ablation.py CLI (FedProx, cooperative, time-of-day ablations)
    - build_fedprox_configs() — mu sweep {0.0, 0.01, 0.1}
    - build_cooperative_configs() — alpha sweep {1.0, 0.5, 0.1}
    - build_time_of_day_configs() — fixed vs ToD+encoding
    - run_ablation() — generic ablation executor with dry-run support
  affects:
    - Future experiment execution (actual SUMO runs)
    - Analysis pipeline (07-03 consumes campaign results)
tech_stack:
  added: []
  patterns:
    - sys.path.insert() for script-mode imports
    - argparse with --dry-run override for smoke testing
    - Sequential ablation execution with per-config logging
key_files:
  created:
    - BackEnd/scripts/run_extension_ablation.py
    - BackEnd/tests/test_ablation_configs.py
  modified: []
decisions:
  - "alpha=0.0 excluded from cooperative configs — creates degenerate zero reward signal (research pitfall 6)"
  - "Time-of-day pairs time_of_day=True with use_time_encoding=True as the natural full-ToD treatment"
  - "Each ablation saves under its own campaign name (fedprox-ablation, cooperative-ablation, tod-ablation)"
metrics:
  completed_date: "2026-03-23T04:45:00Z"
  tasks_completed: 2
  files_created: 2
  tests_passed: 5
---

# Phase 07 Plan 02: Extension Ablation Script Summary

**One-liner:** CLI ablation runner with three parameterized experiment definitions (FedProx mu sweep, cooperative alpha sweep, time-of-day toggle) plus config validation tests.

## What Was Built

### BackEnd/scripts/run_extension_ablation.py
- `build_fedprox_configs()`: 3 configs with mu in {0.0, 0.01, 0.1} — mu=0.0 is FedAvg baseline
- `build_cooperative_configs()`: 3 configs with alpha in {1.0, 0.5, 0.1} — alpha=0.0 excluded per pitfall 6
- `build_time_of_day_configs()`: 2 configs — fixed demand vs. ToD curriculum + time encoding
- `run_ablation()`: generic runner that iterates configs, calls train_and_evaluate, saves results
- argparse CLI: `--ablation {fedprox,cooperative,time-of-day,all}`, `--topology`, `--n-episodes`, `--n-eval-runs`, `--dry-run`
- Summary table printed at end with config name, completed/failed counts, avg wait time, duration

### BackEnd/tests/test_ablation_configs.py
5 unit tests (no SUMO required):
- `test_fedprox_configs_count_and_mu_values` — verifies 3 configs with correct mu values
- `test_cooperative_configs_no_zero_alpha` — verifies alpha=0.0 is NOT included
- `test_time_of_day_configs_encoding_pairing` — verifies ToD + encoding are paired correctly
- `test_configs_respect_topology_override` — verifies topology propagates
- `test_configs_respect_episodes_override` — verifies n_episodes propagates

## Verification Results

```
5 passed — all config builders produce correct parameter values
alpha=0.0 exclusion verified by test
imports OK (from scripts.run_extension_ablation import build_fedprox_configs, ...)
```

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all config builders produce real ExtensionConfig instances. Actual SUMO training requires running the script with a live SUMO installation.

## Self-Check: PASSED
