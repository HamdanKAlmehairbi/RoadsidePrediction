# Project Completion Plan — From Midterm to Final

## Reframed Title
"When Does Federation Help? A Controlled Comparison of Training Strategies for Multi-Intersection Traffic Signal Control"

## Core Finding to Prove
The optimal RL training strategy depends on deployment conditions — network size, traffic demand, and bandwidth constraints. FedRL's advantage grows with network scale and persists across demand levels.

---

## Phase 1: Full Multi-Demand Evaluation (~32 hours compute)

### What to run
4 demand settings × 4 strategies × 2 topologies = 32 configurations

| Demand | VPLPH | Purpose |
|--------|-------|---------|
| Low | 150 | Does RL even matter when traffic is light? |
| Medium | 360 | Current baseline — already done |
| High | 600 | Stress test — how do strategies degrade? |
| Variable | 200-700 | Adaptability to changing conditions |

### Strategies
- SARL (single shared policy)
- MARL (independent policies, central coordinator)
- FedRL (independent policies, periodic reward-weighted aggregation)
- Fixed-Time (non-learning baseline)

### Config per run
- 25 episodes training
- 10 Monte Carlo evaluation runs (seeds 42-51)
- Same PPO hyperparameters, same observation space

### What to modify in code
- Add `demand_level` parameter to `run_all_training.py`
- Pass VPLPH override to `generate_random_routes()`
- Save results to `results/campaigns/demand-{level}/results.json`

### Time estimate
- Grid 3x3: 30 min/config × 16 configs = ~8 hours
- Grid 5x5: 90 min/config × 16 configs = ~24 hours
- Total: ~32 hours (run overnight over 2 nights)

### Already done (subtract from total)
- Medium demand (360 VPLPH): 6 of 8 configs done (3 strategies × 2 topologies from training-curves)
- Need: Fixed-Time at 360 on both grids (already in baseline campaign)

### Deliverable
Performance curves: waiting time vs demand level for each strategy on each topology. The crossover point where FedRL overtakes SARL is the headline finding.

---

## Phase 2: Real-World Topology Validation (~4 hours compute)

### What to run
4 strategies × 1 real topology × 1 demand level (360 VPLPH)

### Topology
Cologne 8 from RESCO benchmark (8 intersections, real-world German city)
- Network file already downloaded: `configs/SMARTCOMP/cologne8.net.xml`
- Issue: SEAL observation code crashes on real-world networks due to lane structure assumptions
- Fix needed: debug `light.py` observation code for arbitrary intersection geometries

### Alternative if Cologne fails
RESCO `grid4x4` (synthetic but different geometry than our grids — 16 intersections, different lane pattern)
- Would still prove framework handles non-standard topologies

### Deliverable
Evidence that SARL-vs-FedRL trend holds (or does not hold) on non-grid networks. Either outcome is publishable.

---

## Phase 3: Algorithm Independence Check (~4 hours compute)

### What to run
2 strategies (SARL, FedRL) × 2 topologies × 1 demand (360) with DQN instead of PPO

### What to modify
- Create a DQN config option in `create_trainer()` using Ray RLlib's DQN
- Keep everything else identical — same observations, same reward, same evaluation

### Deliverable
If the SARL-vs-FedRL ordering is preserved under DQN, the finding is algorithm-independent. If it reverses, that is equally interesting — it means the training strategy interacts with the algorithm, which no one has shown.

---

## Phase 4: Aggregation Variant Comparison (~6 hours compute)

### What to run on Grid 3x3 and Grid 5x5 at 360 VPLPH
- Naive FedAvg (equal weight) — already implemented, just change weight_fn parameter
- Reward-weighted FedAvg (current default) — already done
- FedProx mu=0.01 — already done on Grid 3x3, need Grid 5x5
- FedProx mu=0.1 — already done on Grid 3x3, need Grid 5x5

### What to add
- FedProx at all 4 demand levels on Grid 3x3 to see if mu matters more under stress

### Deliverable
Table showing whether advanced aggregation provides meaningful improvement over naive averaging, and under what conditions.

---

## Phase 5: Statistical Rigor (~16 hours compute)

### What to change
- Increase Monte Carlo evaluation runs from 5 to 10 for all key configurations
- Run Wilcoxon signed-rank tests for every pairwise strategy comparison at each demand level
- Compute effect sizes (Cohen's d) alongside p-values
- Report bootstrap 95% confidence intervals

### What to implement
- Add `scipy.stats.wilcoxon` to `generate_tables.py`
- Add p-value column to LaTeX output tables
- Add significance markers (* p<0.05, ** p<0.01) to figures

### Deliverable
Every claim in the paper is backed by a significance test. "FedRL outperforms MARL on Grid 5x5 (p=0.03, Wilcoxon signed-rank)" instead of "FedRL achieves 19.2s vs 24.7s."

---

## Phase 6: Robustness Testing (~4 hours compute)

### Demand spike test
- Mid-episode demand surge: VPLPH jumps from 360 to 700 at timestep 225 (halfway through episode)
- Run all 4 strategies on both grids
- Measures: which strategy recovers fastest, which degrades least

### Sensor failure test (optional)
- Corrupt one intersection's observations (set to zeros or random noise)
- Does FedRL's aggregation propagate the corrupted weights to other agents?
- Does SARL's shared policy collapse because one input source is garbage?

### Deliverable
Robustness comparison table. Cite T-REX (2025) and position these tests as complementary.

---

## Phase 7: Paper Revision

### Structure changes
- Retitle: "When Does Federation Help? A Controlled Comparison of Training Strategies for Multi-Intersection Traffic Signal Control"
- Lead with the finding, not the framework
- Framework becomes Section III (Methodology), finding becomes the story

### New sections/content
- Performance curves figure (waiting time vs demand level, one line per strategy)
- Crossover analysis: at what network size and demand level does FedRL overtake SARL
- Algorithm independence result (PPO vs DQN comparison)
- Aggregation variant table
- Robustness results
- Actionable recommendation: "Use SARL for networks under X intersections, FedRL for networks over X intersections under bandwidth constraints"

### References to add
- Jayawardana et al., "Impact of task underspecification," NeurIPS 2022
- Li et al., "Federated deep RL-based urban TSC," Scientific Reports 2025
- T-REX robustness benchmark, arXiv 2025
- PyTSC, Sensors 2025
- LightSim, arXiv 2025

### Limitations section (honest, preempts reviewers)
- Synthetic grids + one real topology — not city-scale
- Communication cost is theoretical, not measured on hardware
- No formal privacy guarantees (standard FL privacy only)
- No convergence proofs
- Findings established under PPO and DQN; other algorithms untested

---

## Phase 8: Figures and Tables

### New figures needed
1. Performance curves: waiting time vs VPLPH for each strategy (both grids)
2. Crossover plot: at what point does FedRL overtake SARL
3. Aggregation variant comparison bar chart
4. Robustness: recovery curves after demand spike
5. Updated heatmap: strategy × topology × demand level (3D)

### Updated figures
6. Learning curves with 10 MC run error bands
7. Waiting time bars with significance markers
8. Communication cost with real-world bandwidth context

### New tables
9. Full results matrix: 4 strategies × 4 demands × 2 topologies with p-values
10. Aggregation variant comparison
11. Robustness results
12. Actionable recommendation table

---

## Timeline

| Week | Phase | Compute | Output |
|------|-------|---------|--------|
| 1 | Phase 1: Multi-demand (Grid 3x3) | 8 hrs | 16 configs done |
| 1 | Phase 3: DQN validation | 4 hrs | Algorithm independence check |
| 2 | Phase 1: Multi-demand (Grid 5x5) | 24 hrs | Full 32-config matrix |
| 2 | Phase 4: Aggregation variants | 6 hrs | FedAvg vs FedProx comparison |
| 3 | Phase 2: Real-world topology | 4 hrs | Cologne/Ingolstadt results |
| 3 | Phase 5: Statistical rigor | 16 hrs | 10 MC runs + Wilcoxon tests |
| 3 | Phase 6: Robustness | 4 hrs | Demand spike results |
| 4 | Phase 7: Paper revision | 0 hrs | Rewritten paper |
| 4 | Phase 8: Figures and tables | 0 hrs | Publication-ready artifacts |

Total compute: ~66 hours over 3 weeks
Total human effort: Paper rewrite in week 4

---

## Success Criteria

The paper is ready when:
1. Every strategy comparison has a Wilcoxon p-value
2. The crossover point (where FedRL overtakes SARL) is identified across demand levels
3. At least one real-world topology validates the trend
4. At least one non-PPO algorithm validates algorithm independence
5. The recommendation table gives actionable guidance: "use X when Y"
6. The title leads with the finding, not the framework

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Grid 5x5 runs take too long | Run overnight, use --resume for fault tolerance |
| Cologne observation code crashes | Fall back to RESCO grid4x4 (different geometry, still non-standard) |
| DQN does not converge | Report this as a finding — "PPO-specific result" is still valuable |
| FedRL does not win at high demand | Report honestly — "federation helps at moderate demand but not under extreme load" is a real finding |
| Compute exceeds available time | Prioritize Phase 1 (multi-demand) and Phase 5 (statistics) — these have highest impact |

---

## What Makes This Hit Hard

The paper stops being "we built a framework" and becomes "here is when you should use federated learning for traffic signal control, and here is when you should not, backed by 32+ controlled experiments across multiple demand levels, topologies, and algorithms." That is actionable. That is what gets cited.
