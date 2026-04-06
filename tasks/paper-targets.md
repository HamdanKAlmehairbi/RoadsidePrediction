# Paper Target Task List

## Tier 1: ITSC 2026 (current + these = accept)

- [ ] **T1-01**: Learning curves with confidence bands — plot training_rewards across 3 seeds per strategy, shaded CI regions
- [ ] **T1-02**: Explicit limitations section — acknowledge: toy scale, PPO-only, reward-KPI gap, training budget sensitivity, fixed-time baseline is "always stay"
- [ ] **T1-03**: Threats to validity subsection — training seed independence, weight reset confound, observation non-locality
- [ ] **T1-04**: Align all paper claims with actual code — hyperparameters, aggregation descriptions, variant names
- [ ] **T1-05**: Main results table — strategy comparison (FedRL/MARL/SARL/fixed-time/max-pressure) × 3 topologies × 3 demands with bootstrap CI and Bonferroni-corrected p-values

## Tier 2: IEEE T-ITS (Tier 1 + these = weak accept → accept)

- [ ] **T2-01**: Convergence analysis — episodes-to-threshold (e.g. 90% of final performance) per strategy, plotted per topology
- [ ] **T2-02**: Transfer experiment — train on grid-3x3, eval on grid-5x5 (and reverse). Code already supports this via weights_path + different topology in MCConfig
- [ ] **T2-03**: Mechanistic explanation — why FedRL differs from MARL/SARL. Analyze weight divergence across agents over training. Measure policy entropy per intersection. Connect aggregation to implicit regularization.
- [ ] **T2-04**: Hierarchical statistical analysis — model-level means (3 seeds) as primary unit, MC runs as nested. Report both levels of variance separately.
- [ ] **T2-05**: Wall-clock and sample efficiency table — training time per strategy, episodes-to-threshold, comm bytes per unit of performance gain
- [ ] **T2-06**: Thorough related work — position against FedLight, CoLight, PressLight, LibSignal, FLOW. Explicitly state what's new vs prior art.
- [ ] **T2-07**: Demand heterogeneity narrative — strategy ranking changes at 150/360/600 VPLPH. Which strategy wins at each level and WHY?
- [ ] **T2-08**: Per-intersection fairness metric — Gini coefficient or max/min reward ratio. Does SARL sacrifice low-traffic intersections? Does MARL overfit to local?
- [ ] **T2-09**: Interaction check — does FedProx help MORE at high demand? Does ToD help MORE on larger networks? At least one 2-factor analysis.
- [ ] **T2-10**: Stronger baseline documentation — describe exactly what fixed-time and max-pressure do. Consider implementing actuated control (phase extension on detector occupancy) as a third non-RL baseline.

## Tier 3: NeurIPS D&B (Tier 1 + Tier 2 + these = competitive submission)

- [ ] **T3-01**: Multi-simulator — add CityFlow backend alongside SUMO. Show same experiments produce consistent ranking across simulators.
- [ ] **T3-02**: Multi-algorithm — add DQN and A2C alongside PPO. Show training strategy effects are not PPO-specific.
- [ ] **T3-03**: Scale to 16+ intersections — grid-7x7 (16 agents) or a real large network. Show how strategy differences change with scale.
- [ ] **T3-04**: Open-source benchmark release — pip-installable package, config-driven experiments, reproducible seeds/checkpoints published.
- [ ] **T3-05**: Leaderboard — hosted results table where others can submit new strategies and compare.
- [ ] **T3-06**: Privacy analysis — differential privacy integration or at minimum membership inference analysis on shared weights.
- [ ] **T3-07**: 20+ page paper — comprehensive analysis, extensive ablation appendix, full experimental protocol documentation.
- [ ] **T3-08**: Robustness experiments — demand shock (sudden spike), partial observability (sensor failure), network modification (road closure mid-episode).
- [ ] **T3-09**: Real-world network validation — at least 2 real city networks beyond cologne-8 (e.g. Ingolstadt from SUMO scenarios, Manhattan grid from FLOW).
- [ ] **T3-10**: Calibration discussion — SUMO-to-reality gap analysis. What assumptions break in deployment?

## Phase Mapping

| Tasks | GSD Phase | When |
|-------|-----------|------|
| T1-01 to T1-05 | Phase 10 (post-experiment) | After HPC results |
| T2-01 to T2-10 | Phase 10 + new Phase 11 | After ITSC draft |
| T3-01 to T3-10 | v2 milestone | Long-term |
