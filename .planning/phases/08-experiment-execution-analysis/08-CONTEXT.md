# Phase 8: Experiment Execution & Analysis - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Run all experiment campaigns (baseline reproduction, FedProx ablation, cooperative reward ablation, time-of-day ablation) against live SUMO on grid-3x3 and grid-5x5, then generate publishable analysis artifacts (LaTeX tables, bar charts, training curves, Wilcoxon significance tests, and a summary report).

All campaign scripts and analysis tools already exist from Phase 7. This phase RUNS them and produces the actual results and figures.

</domain>

<decisions>
## Implementation Decisions

### Execution Order & Scope
- **D-01:** Topologies: grid-3x3 AND grid-5x5 (matches full SEAL paper scope)
- **D-02:** Training episodes: 50 per config (Phase 7 default)
- **D-03:** MC seeds: 10 per config (already configured in scripts)
- **D-04:** Execution order: baseline first, then all 3 ablations sequentially
- **D-05:** Dry-run (1 seed) baseline only before committing to full runs
- **D-06:** After baseline completes, checkpoint for user review before proceeding to ablations

### Failure Handling
- **D-07:** On failure: log error in CampaignResult, skip failed config, continue remaining configs
- **D-08:** No automatic retries — failed configs are logged for manual investigation

### Tables (LaTeX booktabs, mean +/- std)
- **D-09:** Baseline comparison table: FedRL vs MARL vs SARL vs fixed-time vs max-pressure (avg wait time, travel time)
- **D-10:** FedProx ablation table: mu=0.0 vs 0.01 vs 0.1 with Wilcoxon p-values
- **D-11:** Cooperative ablation table: alpha=1.0 vs 0.5 vs 0.1 with Wilcoxon p-values
- **D-12:** Combined extensions table: baseline vs best-FedProx vs best-cooperative vs ToD — single summary comparing all extensions

### Charts
- **D-13:** Grouped bar charts per ablation (already in generate_tables.py) — avg wait time with 95% CI error bars
- **D-14:** Training convergence curves: episode reward over time for each config (NEW — not in generate_tables.py)
- **D-15:** Combined comparison chart: baseline + best config from each ablation side-by-side
- **D-16:** Per-topology breakdown: separate charts for grid-3x3 and grid-5x5 showing extension scaling

### Summary Report
- **D-17:** Auto-generated markdown report pulling together all tables, charts, significance results, and key takeaways — ready to paste into paper draft
- **D-18:** Report saved to BackEnd/results/figures/EXPERIMENT_REPORT.md

### Claude's Discretion
- Script orchestration approach (wrapper script vs plan tasks calling existing CLIs)
- Training convergence curve implementation details (matplotlib style, smoothing)
- Report markdown formatting and narrative structure
- "Best config" selection logic for combined table/chart (lowest avg wait time)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Campaign Infrastructure (Phase 7)
- `BackEnd/api/evaluation/campaign_config.py` — ExtensionConfig dataclass, CampaignResult, EXAMPLE_WEIGHTS_MAP
- `BackEnd/scripts/run_campaign.py` — Baseline campaign runner CLI (--campaign baseline, --dry-run, --topologies)
- `BackEnd/scripts/run_extension_ablation.py` — Extension ablation CLI (--ablation {fedprox,cooperative,time-of-day,all})
- `BackEnd/scripts/generate_tables.py` — Analysis: wilcoxon_compare, generate_latex_table, plot_comparison_bar, generate_all_outputs

### Evaluation Framework (Phase 3)
- `BackEnd/api/evaluation/monte_carlo.py` — MCConfig, run_monte_carlo, MCAggregatedResult
- `BackEnd/api/evaluation/metrics.py` — Metric computation from SUMO tripinfo

### Phase 7 Plans (for understanding existing capabilities)
- `.planning/phases/07-experiment-campaigns/07-01-SUMMARY.md` — Campaign config and runner details
- `.planning/phases/07-experiment-campaigns/07-02-SUMMARY.md` — Ablation script details
- `.planning/phases/07-experiment-campaigns/07-03-SUMMARY.md` — Analysis pipeline details

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `run_campaign.py` CLI: ready to run baseline with `--campaign baseline --topologies grid-3x3 grid-5x5`
- `run_extension_ablation.py` CLI: ready to run all ablations with `--ablation all --topology grid-3x3`
- `generate_tables.py` CLI: generates LaTeX tables and bar charts from campaign results
- `wilcoxon_compare()`: paired significance testing already implemented
- `results_to_dataframe()`: campaign JSON -> pandas DataFrame for analysis

### What Needs to Be Built NEW
- Training convergence curve plotter (D-14) — train_and_evaluate captures training_rewards in CampaignResult but no chart exists yet
- Combined comparison chart (D-15) — requires selecting "best" config from each ablation
- Per-topology breakdown charts (D-16) — generate_tables.py groups by name, needs topology filtering
- Combined extensions table (D-12) — requires cross-ablation result aggregation
- Summary report generator (D-17) — new script combining all outputs into markdown

### Integration Points
- Campaign results saved to `BackEnd/results/campaigns/{name}/results.json`
- Figures saved to `BackEnd/results/figures/`
- CampaignResult.training_rewards contains per-episode reward data for convergence curves

</code_context>

<specifics>
## Specific Ideas

- Baseline checkpoint: after baseline runs, user reviews avg wait times to sanity-check before committing to expensive ablation runs
- Training curves should show smoothed line (moving average) with raw data as faint background
- Combined table should highlight "best" values in bold (lowest wait time, highest reward)
- Report should include a "Key Findings" section at the top summarizing whether extensions improve over baseline

</specifics>

<deferred>
## Deferred Ideas

- grid-7x7 experiments — no example weights, very slow, defer to v2
- Transfer matrix analysis (train on grid-3x3, test on grid-5x5) — separate analysis phase
- Interactive dashboard for exploring results — frontend phase

</deferred>

---

*Phase: 08-experiment-execution-analysis*
*Context gathered: 2026-03-23*
