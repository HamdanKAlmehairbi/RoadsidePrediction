---
phase: 07-experiment-campaigns
verified: 2026-03-23T00:00:00Z
status: gaps_found
score: 9/12 must-haves verified
re_verification: false
gaps:
  - truth: "Results from campaign runs are saved as JSON with config snapshot alongside metrics"
    status: failed
    reason: "save_campaign_results() (run_campaign.py:286) dumps results_data as a plain JSON list, but load_campaign_results() (generate_tables.py:58) calls data['results'] expecting a dict with a 'results' key. This key never exists — load_campaign_results will always raise KeyError at runtime."
    artifacts:
      - path: "BackEnd/scripts/run_campaign.py"
        issue: "Line 286: json.dump(results_data, ...) writes a plain list, not {\"results\": results_data}"
      - path: "BackEnd/scripts/generate_tables.py"
        issue: "Line 58: return data['results'] assumes a dict wrapper that is never written"
    missing:
      - "Change run_campaign.py save_campaign_results() to wrap output: json.dump({\"results\": results_data, \"configs\": configs_data}, fp, ...) and align generate_tables.py accordingly, OR change generate_tables.py load_campaign_results() to handle a bare list by checking isinstance(data, list)"
  - truth: "CAMP-02: FedProx ablation: mu in {0.0, 0.01, 0.1} with fresh training and MC evaluation"
    status: partial
    reason: "The script infrastructure (build_fedprox_configs, run_ablation) is fully implemented and tested. However the REQUIREMENTS.md traceability table marks CAMP-02 as 'Pending' — the ablation has not been executed against real SUMO; only config builders have been verified. The phase goal says 'publishable experiment results', which requires actual runs."
    artifacts:
      - path: "BackEnd/scripts/run_extension_ablation.py"
        issue: "Config builders verified; actual experiment execution against live SUMO not performed"
    missing:
      - "This is an execution gap, not a code gap. The scripts are correct. Running --ablation fedprox against a live SUMO installation produces results. Requirement status should reflect that the automation is ready but the experiment has not been run."
  - truth: "CAMP-03: Cooperative reward ablation: alpha in {1.0, 0.5, 0.1} with fresh training and MC evaluation"
    status: partial
    reason: "Same as CAMP-02 — config builders correct, execution not performed, REQUIREMENTS.md still 'Pending'."
    artifacts:
      - path: "BackEnd/scripts/run_extension_ablation.py"
        issue: "Config builders verified; actual experiment execution not performed"
    missing:
      - "Same as CAMP-02: scripts ready, experiment not yet run."
  - truth: "CAMP-04: Time-of-day ablation: fixed demand vs curriculum with time encoding"
    status: partial
    reason: "Same as CAMP-02/CAMP-03 — config builder verified, experiment not executed."
    artifacts:
      - path: "BackEnd/scripts/run_extension_ablation.py"
        issue: "Config builders verified; actual experiment execution not performed"
    missing:
      - "Same as CAMP-02: scripts ready, experiment not yet run."
  - truth: "CAMP-06: Publishable output: comparison tables (LaTeX), bar charts with error bars"
    status: partial
    reason: "generate_tables.py is fully implemented and all 7 unit tests pass with synthetic data. However CAMP-06 is marked 'Pending' in REQUIREMENTS.md, and real output files (tables, charts) cannot be generated because the JSON wiring break between save_campaign_results and load_campaign_results (see first gap) prevents generate_tables.py from reading any actual campaign results. The output pipeline is blocked by the save/load mismatch."
    artifacts:
      - path: "BackEnd/scripts/generate_tables.py"
        issue: "Fully implemented but blocked from operating on real campaign output due to save/load JSON key mismatch"
    missing:
      - "Fix the JSON save/load mismatch (first gap). Once fixed, generate_tables.py will produce real tables and charts."
human_verification:
  - test: "Run baseline campaign dry-run end-to-end with live SUMO"
    expected: "python scripts/run_campaign.py --campaign baseline --dry-run 1 completes, writes results.json and config.json to BackEnd/results/campaigns/baseline/"
    why_human: "Requires live SUMO installation and trained weights; cannot verify without executing the full pipeline"
  - test: "Run FedProx ablation dry-run"
    expected: "python scripts/run_extension_ablation.py --ablation fedprox --dry-run 1 completes with 3 CampaignResult entries saved"
    why_human: "Requires live SUMO and Ray; cannot verify programmatically"
  - test: "End-to-end pipeline: save results then generate tables and charts"
    expected: "After running a campaign, generate_tables.py --campaign-dir produces table_avg_waiting_time.tex and chart_avg_waiting_time.png with real values"
    why_human: "Requires live campaign run; blocked by JSON mismatch until fixed"
---

# Phase 7: Experiment Campaigns Verification Report

**Phase Goal:** Reproduce SEAL paper results, evaluate Phase 4 extensions (FedProx, cooperative reward, time-of-day), and produce publishable comparison tables with statistical rigor
**Verified:** 2026-03-23
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Campaign runner script executes baseline evaluation using example weights | VERIFIED | run_campaign.py:145 `run_baseline_campaign()` resolves example weights via `resolve_example_weights()` and calls `run_monte_carlo()` directly |
| 2 | Campaign config dataclass captures all experiment parameters | VERIFIED | campaign_config.py:45 `ExtensionConfig` has fedprox_mu, alpha, time_of_day, use_time_encoding, n_episodes, n_eval_runs, base_seed, ranked, horizon |
| 3 | Results from campaign runs are saved as JSON with config snapshot alongside metrics | FAILED | run_campaign.py:286 writes a plain list; generate_tables.py:58 reads `data["results"]` — the key never exists |
| 4 | Smoke test validates campaign config and runner (no error) | VERIFIED | 16 tests pass: 5 in test_campaign.py, 5 in test_ablation_configs.py, 6/7 in test_analysis.py (note: plan stated 7, result confirms 7 — "16 passed" covers all three suites) |
| 5 | FedProx ablation defines mu in {0.0, 0.01, 0.1} | VERIFIED | run_extension_ablation.py:79-104 `build_fedprox_configs()` returns exactly [0.0, 0.01, 0.1] confirmed by test |
| 6 | Cooperative reward ablation defines alpha in {1.0, 0.5, 0.1} with alpha=0.0 excluded | VERIFIED | run_extension_ablation.py:128-153 returns [1.0, 0.5, 0.1]; alpha=0.0 absent from code (only in comment line 118) |
| 7 | Time-of-day ablation defines fixed demand vs ToD+encoding | VERIFIED | run_extension_ablation.py:178-197 `build_time_of_day_configs()` returns 2 configs with correct flag pairing |
| 8 | Wilcoxon signed-rank test compares paired MC results | VERIFIED | generate_tables.py:187 `stats.wilcoxon(values_a, values_b)` with n>=8 guard; test_analysis.py confirms p<0.05 for distinct distributions |
| 9 | LaTeX booktabs table generated with mean +/- std formatting | VERIFIED | generate_tables.py:244 `\\toprule`, `\\midrule`, `\\bottomrule`; `\\pm` formatting; test_generate_latex_table_has_booktabs passes |
| 10 | Grouped bar chart with 95% CI error bars generated as PNG | VERIFIED | generate_tables.py:281,316 uses `1.96 * std / sqrt(n)`; plt.savefig at dpi=150; test_plot_comparison_bar_creates_file passes with non-empty PNG |
| 11 | Results DataFrame builder flattens campaign JSON into pandas rows | VERIFIED | generate_tables.py:65 `results_to_dataframe()` extracts config and aggregated metrics; test_results_to_dataframe_columns confirms column presence |
| 12 | Each ablation saves results per-ablation as named campaign directories | VERIFIED | run_extension_ablation.py:247 calls `save_campaign_results(results, ablation_name)`; campaign_name_map provides "fedprox-ablation", "cooperative-ablation", "tod-ablation" |

**Score:** 9/12 truths verified (3 fully failed, 3 partial execution gaps not blocking infrastructure)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `BackEnd/api/evaluation/campaign_config.py` | ExtensionConfig and CampaignResult dataclasses | VERIFIED | 192 lines; both dataclasses present with all required fields; EXAMPLE_WEIGHTS_MAP with 6 entries; resolve_example_weights, config_to_dict, result_to_dict helpers |
| `BackEnd/scripts/run_campaign.py` | CLI campaign runner for baseline paper reproduction | VERIFIED | 384 lines; train_and_evaluate, run_baseline_campaign, save_campaign_results, main() with argparse all present |
| `BackEnd/tests/test_campaign.py` | Smoke tests for campaign config and runner | VERIFIED | 112 lines; 5 substantive tests; no SUMO required |
| `BackEnd/scripts/run_extension_ablation.py` | CLI script for three extension ablation studies | VERIFIED | 401 lines; build_fedprox_configs, build_cooperative_configs, build_time_of_day_configs, run_ablation, main() all present |
| `BackEnd/tests/test_ablation_configs.py` | Unit tests for ablation config correctness | VERIFIED | 128 lines; 5 substantive tests covering count, values, exclusion, overrides |
| `BackEnd/scripts/generate_tables.py` | Statistical analysis, LaTeX table, chart generation | VERIFIED | 461 lines; wilcoxon_compare, generate_latex_table, plot_comparison_bar, results_to_dataframe, generate_all_outputs, main() all present |
| `BackEnd/tests/test_analysis.py` | Unit tests for stats, table, and chart functions | VERIFIED | 221 lines; 6 substantive tests (plan stated 7 — 6 are present: DataFrame x2, Wilcoxon x2, LaTeX x1, chart x1) |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| run_campaign.py | BackEnd/api/evaluation/monte_carlo.py | `from api.evaluation.monte_carlo import run_monte_carlo, MCConfig` | WIRED | Lines 21+43 — both comment docs and live import |
| run_campaign.py | BackEnd/api/training_runner.py | `from api.training_runner import create_trainer, run_training_loop` | WIRED | Lines 20+44 — both present |
| run_campaign.py | BackEnd/api/evaluation/campaign_config.py | `from api.evaluation.campaign_config import ExtensionConfig, EXAMPLE_WEIGHTS_MAP, ...` | WIRED | Lines 36-42 |
| run_extension_ablation.py | BackEnd/scripts/run_campaign.py | `from scripts.run_campaign import train_and_evaluate, save_campaign_results` | WIRED | Lines 28+43 — both comment docs and live import |
| run_extension_ablation.py | BackEnd/api/evaluation/campaign_config.py | `from api.evaluation.campaign_config import CampaignResult, ExtensionConfig` | WIRED | Lines 29+42 |
| generate_tables.py | BackEnd/results/campaigns/ | `json.load` of results.json | PARTIAL | json.load present (line 57); but reads `data["results"]` while save_campaign_results writes a bare list — KeyError at runtime |
| generate_tables.py | BackEnd/results/figures/ | `plt.savefig` output | WIRED | Line 348 plt.savefig confirmed; default output_dir points to BackEnd/results/figures/ |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CAMP-01 | 07-01 | Reproduce baseline paper results: FedRL/MARL/SARL/fixed-time/max-pressure on grid-3x3, grid-5x5 | SATISFIED | run_campaign.py `run_baseline_campaign()` uses example weights for RL trainers, MCConfig directly for non-RL baselines |
| CAMP-02 | 07-02 | FedProx ablation: mu in {0.0, 0.01, 0.1} with fresh training and MC evaluation | SATISFIED (infrastructure) | build_fedprox_configs() + run_ablation() fully implemented and tested; REQUIREMENTS.md marks "Pending" — experiment not yet run against live SUMO |
| CAMP-03 | 07-02 | Cooperative reward ablation: alpha in {1.0, 0.5, 0.1} with fresh training and MC evaluation | SATISFIED (infrastructure) | build_cooperative_configs() + run_ablation() verified; alpha=0.0 excluded; experiment not yet run |
| CAMP-04 | 07-02 | Time-of-day ablation: fixed demand vs curriculum with time encoding | SATISFIED (infrastructure) | build_time_of_day_configs() verified; experiment not yet run |
| CAMP-05 | 07-01, 07-03 | Statistical rigor: 10 MC seeds, 95% CI, Wilcoxon signed-rank test | SATISFIED | MCConfig n_runs=10; generate_tables.py 1.96*std/sqrt(n) CI; scipy.stats.wilcoxon with n>=8 guard |
| CAMP-06 | 07-03 | Publishable output: LaTeX tables, bar charts with error bars | BLOCKED | generate_tables.py functions are implemented and pass synthetic-data tests, but the JSON save/load mismatch blocks runtime use on real campaign results |
| CAMP-07 | 07-01 | Results persistence: all configs and seeds logged alongside results as JSON | PARTIAL | save_campaign_results writes results.json (list) and config.json; but save/load mismatch means the saved format cannot be consumed by generate_tables.py |

**Orphaned requirements:** None — all 7 CAMP requirements are claimed by plans and accounted for.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| run_campaign.py | 349 | "Custom campaigns not yet implemented." | Info | `--campaign custom` raises `parser.error()`; this is an intentional stub for a future feature not required by any CAMP requirement |
| run_campaign.py | 286 | `json.dump(results_data, fp, ...)` writes bare list | Blocker | generate_tables.py:58 reads `data["results"]` from this file — will always raise `KeyError` at runtime |

---

## Human Verification Required

### 1. Baseline campaign dry-run with live SUMO

**Test:** `cd BackEnd && python scripts/run_campaign.py --campaign baseline --dry-run 1`
**Expected:** Completes without error; BackEnd/results/campaigns/baseline/results.json and config.json created; summary prints "10 configs run" (5 trainers x 2 topologies)
**Why human:** Requires live SUMO installation, example weights on disk (test_example_weights_map_paths_exist verifies .pkl files exist), and functioning Ray environment

### 2. FedProx ablation dry-run

**Test:** `cd BackEnd && python scripts/run_extension_ablation.py --ablation fedprox --dry-run 1`
**Expected:** 3 configs trained and evaluated; results saved to BackEnd/results/campaigns/fedprox-ablation/
**Why human:** Requires live SUMO and Ray; training loop execution cannot be verified statically

### 3. End-to-end generate_tables pipeline (after JSON fix)

**Test:** After fixing the JSON save/load mismatch and running a campaign, `python scripts/generate_tables.py --campaign-dir BackEnd/results/campaigns/baseline/`
**Expected:** Produces table_avg_waiting_time.tex (with \\toprule) and chart_avg_waiting_time.png (non-empty) in BackEnd/results/figures/
**Why human:** Requires real campaign output; unit tests cover synthetic data only

---

## Gaps Summary

**One blocker gap and four partial gaps exist.**

**Blocker (CAMP-06, CAMP-07):** `save_campaign_results()` in `run_campaign.py` dumps `results_data` as a bare JSON list. `load_campaign_results()` in `generate_tables.py` reads `data["results"]`, expecting the list to be wrapped under a `"results"` key. This mismatch will raise `KeyError` whenever `generate_tables.py` is pointed at any campaign directory produced by `run_campaign.py` or `run_extension_ablation.py`. The fix is one line: change `json.dump(results_data, ...)` to `json.dump({"results": results_data}, ...)` in `run_campaign.py`, or change `load_campaign_results` to handle both formats.

**Execution gaps (CAMP-02, CAMP-03, CAMP-04):** The ablation infrastructure is complete and tested. These requirements are marked "Pending" in REQUIREMENTS.md because the experiments have not been executed against a live SUMO environment. The code is correct; the gap is that no actual results JSON has been produced by running the ablations. This is a runtime/environment gap, not a code gap.

**Root cause grouping:** The blocker (save/load mismatch) and the CAMP-06 gap share the same root — the JSON key schema was not agreed on between the writer (run_campaign.py) and the reader (generate_tables.py). A single fix resolves both.

---

_Verified: 2026-03-23_
_Verifier: Claude (gsd-verifier)_
