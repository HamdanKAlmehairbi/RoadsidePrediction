---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Completed 08-02-PLAN.md
last_updated: "2026-03-23T09:46:31.621Z"
progress:
  total_phases: 11
  completed_phases: 3
  total_plans: 20
  completed_plans: 13
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-22)

**Core value:** Real FedRL experiments working end-to-end with publishable results
**Current focus:** Phase 09 — pre-experiment-hardening (before HPC run)

## Current Position

Phase: 09 (pre-experiment-hardening) — READY TO PLAN
Plan: 0 of 3

## Performance Metrics

**Velocity:**

- Total plans completed: 14 (Phases 0-2 + 03-01, 03-02, 03-03, 03-04, 03-05)
- Average duration: N/A (bulk session)
- Total execution time: ~3 hours (single session) + 12 min (03-01) + ~10 min (03-02) + 2 min (03-03) + 8 min (03-04) + 10 min (03-05)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 0. Env Setup | 2 | ~20 min | ~10 min |
| 1. SEAL Engine | 3 | ~45 min | ~15 min |
| 2. Wire API | 4 | ~90 min | ~22 min |
| 3. Evaluation Framework | 5/5 done | ~42 min | ~8.4 min |
| Phase 04 P02 | 1min | 1 tasks | 1 files |
| Phase 04 P01 | 2min | 2 tasks | 3 files |
| Phase 04 P03 | 3min | 2 tasks | 6 files |
| Phase 07 P01 | 139 | 2 tasks | 3 files |
| Phase 08 P01 | 5 | 2 tasks | 4 files |
| Phase 08 P02 | 30 | 2 tasks | 7 files |

## Accumulated Context

### Decisions

Decisions logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Phase 1]: Ray 2.x old API stack mode for backward compatibility
- [Phase 2]: Singleton Ray init at server startup, frame drop on WS queue
- [Phase 2]: SUMO_HOME auto-detection for portable deployment
- [Phase 3, 03-01]: PPOTorchPolicy standalone (not full algorithm) for eval inference
- [Phase 3, 03-01]: TrialResult dataclass (typed) not plain dict for downstream safety
- [Phase 3, 03-01]: resolve_weights_path() falls back to example_weights for zero-config use
- [Phase 3, 03-01]: max-pressure falls back to fixed-time if TraCI unavailable
- [Phase 3, 03-03]: Stdlib-only aggregation (statistics+math) — avoids numpy in orchestration layer
- [Phase 3, 03-03]: Weights resolved once before N-run loop — fast-fail on missing weights
- [Phase 3, 03-03]: on_progress callback exceptions swallowed — UI errors never abort campaigns
- [Phase 3, 03-03]: seed = base_seed + i for deterministic reproducibility across runs
- [Phase 3, 03-04]: Dual-path result lookup (in-memory fast path, disk slow path for restart)
- [Phase 3, 03-04]: json.dump(default=str) for safe serialization of arbitrary result types
- [Phase 3, 03-04]: Error results also persisted to disk so GET always returns useful response
- [Phase 3, 03-04]: include_transfer defaults False — transfer matrix is opt-in (expensive)
- [Phase 3, 03-05]: raise_server_exceptions=False on TestClient for 202 async endpoint tests
- [Phase 3, 03-05]: monkeypatch RESULTS_DIR for hermetic store tests (no pollution of real storage)
- [Phase 3, 03-05]: SUMO_AVAILABLE guard pattern for graceful skip of SUMO-dependent tests
- [Phase 04]: Cooperative reward: alpha before super().__init__(), isolated nodes use own reward as neighbor_mean, _get_reward wrapper preserved for abstract contract
- [Phase 04]: FedProx as PPOTorchPolicy subclass with proximal loss, global weights stored after FedAvg set_weights
- [Phase 04]: Time encoding appended AFTER ranking to preserve rank indices 10-13
- [Phase 04]: Per-episode demand sampling for curriculum diversity, time_of_day is training-only
- [Phase 04]: Full Phase 4 param passthrough: TrainRequest -> create_trainer -> env_config_fn -> SumoEnv
- [Phase 07]: EXAMPLE_WEIGHTS_MAP uses actual .pkl filenames (v3_naive-aggr_ranked.pkl for FedRL) not resolve_weights_path() naming — bypasses naming convention mismatch
- [Phase 07]: Non-RL baselines (fixed-time, max-pressure) use MCConfig directly in run_baseline_campaign — they skip train_and_evaluate
- [Phase 08]: plot_combined_comparison selects FedRL trainer row from baseline for cross-ablation comparison
- [Phase 08]: generate_report.py wraps each section in try/except — missing campaign dirs produce 'not available' notes without aborting whole report
- [Phase 08]: Baseline campaign approved by user with 0 errors across 10 configs — D-06 gate passed, ablation runs may proceed
- [Phase 08]: results_to_dataframe and wilcoxon_compare unwrap campaign_to_dict serialization layer — fix ensures generate_tables.py compatibility

### Roadmap Evolution

- Phase 7 added: Experiment Campaigns — reproduce paper results and evaluate all extensions
- Phase 8 added: Experiment Execution & Analysis — run all campaigns against live SUMO and generate publishable artifacts

### Pending Todos

None.

### Blockers/Concerns

- Cologne-8 demand levels: 150/360/600 VPLPH may be too high for cologne-8 (many more lanes). Consider 50-150 VPLPH for cologne-8.
- Cross-seed stats aggregation: reporting code doesn't yet aggregate across training seeds — Phase 11 task
- Training comm bytes tracked but not surfaced in MC eval stats — Phase 11 task
- FedDistill consensus logits computed from zero observation (weak signal, functional)
- Dummy envs in on_policy_setup() not explicitly closed (potential SUMO process leak)

## Session Continuity

Last session: 2026-04-06
Stopped at: Phase 9 — implemented all 10 training strategies, fixed audit issues, ready for HPC
Resume file: None

### What was done this session (2026-04-06):
1. Cleaned up MDs, moved non-dev docs to archive/
2. Installed Codex plugin, enabled review gate
3. Added Codex verification to GSD verify-phase workflow
4. Pre-flight check: fixed pandas/numpy compat, null eval crash, missing matplotlib in requirements
5. Fixed MARL eval (per-agent policy save), alpha computation (shift-normalize), both from CODEX-AUDIT
6. Added 6 PRE-experiment requirements (training seeds, multi-demand, throughput, comm cost, bootstrap CI, effect size) — all implemented
7. Added 3 new ablation configs: strategy comparison (with baselines), aggregation (pure: naive/reward/traffic), fedrl-variants
8. Fixed results-overwrite bug in save_campaign_results (now appends)
9. Fixed aggregation ablation confound (replaced FedProx arm with traffic-weighted)
10. Fixed cooperative eval (alpha now flows to MCConfig → run_trial)
11. Fixed PPO hyperparameters in CLAUDE.md to match code
12. Added Bonferroni correction to Wilcoxon tests
13. Implemented 5 NEW training strategies: Gossip RL, Mean Field RL, CTDE, HierFed, FedDistill
14. Fixed critical bugs from audit: training_data init, CTDE env init order, Gossip save_test_policy, FedRL episode_data accumulation
15. Created paper-targets.md with tiered task lists (ITSC / T-ITS / NeurIPS D&B)
16. Updated experiment design to 2-tier: Tier 1 (10 strategies, 270 runs), Tier 2 (ablations after results)

### Next steps:
- Run Tier 1 experiments on HPC: `--ablation strategy --topologies grid-3x3 grid-5x5 cologne-8 --demand-levels 150 360 600 --training-seeds 42 123 456`
- After HPC results: Phase 11 post-experiment analysis
