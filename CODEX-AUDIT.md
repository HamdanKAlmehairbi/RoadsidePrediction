# Codex Experimental Design Audit
**Date:** 2026-04-02
**Scope:** Full experimental pipeline review from academic researcher perspective

---

## Summary of Findings

Two CRITICAL issues and three MODERATE confounds were identified.

---

## Critical Finding 1 — Evaluation Destroys MARL Specialization (CRITICAL)

**Issue:** `on_make_final_policy()` averages all per-intersection policies into ONE policy before saving. Then `run_trial()` loads that one averaged policy and applies it to every agent during evaluation.

- MARL's entire value proposition is per-intersection specialization — but evaluation destroys that by averaging all policies.
- FedRL clustered and partial variants also lose their specialized/personalized layers at eval time.
- `base.py:219` — `save_test_policy()` calls `on_make_final_policy()` which averages
- `multi_agent.py:26` — MARL `on_make_final_policy()` naive-averages all policies
- `fed_agent.py:66` — FedRL `on_make_final_policy()` fedavg-averages all policies
- `runner.py:176` — loads single policy, applies to all agents

**Impact:** Reported MC evaluation results for MARL do NOT reflect what a deployed MARL system would achieve. They reflect the average policy. This weakens the cross-strategy comparison.

**Fix required:** Either (a) evaluate each agent with its own policy, or (b) clearly state in the paper that evaluation uses the averaged final policy.

---

## Critical Finding 2 — Paper Numbers Don't Match results.json (CRITICAL)

The paper's quantitative tables, learning-curve claims, and communication costs do not match the data in `results/campaigns/training-curves/results.json`.

### Learning curves
- Paper claims: "FedRL achieves the highest final reward on both grids" and "overtakes SARL around episode 15"
- results.json Grid-3x3 final: FedRL=-7.38, SARL=-7.12 → **SARL is better**
- results.json Grid-5x5 final: FedRL=-16.48, SARL=-16.25 → **SARL is better**

### Waiting time (Table I)
| Strategy | Paper (3x3) | results.json (3x3) | Paper (5x5) | results.json (5x5) |
|---|---|---|---|---|
| FedRL | 15.4 ± 1.3 | 11.5 ± 1.5 | 19.2 ± 0.8 | 16.8 ± 1.4 |
| MARL | 14.8 ± 1.5 | 13.9 ± ? | 24.7 ± 1.3 | 23.6 ± ? |
| SARL | 14.2 ± 0.8 | 10.4 ± ? | 19.8 ± 1.1 | 17.4 ± ? |

**Root cause:** The paper uses numbers from the baseline campaign (pre-trained example weights, eval-only). The slides and training-curves campaign used freshly trained weights. Two different experiments — two different sets of numbers. The paper needs to be consistent.

### Communication costs
- Paper claims 102.6 MB (3x3) and 285 MB (5x5) for FedRL — but `total_comm_cost` is 0.0 in every results.json entry.
- These are computed from theoretical data structure sizes, not measured. Paper should state "estimated theoretical communication cost."

### Alpha computation
- Paper claims shift-and-normalize for negative rewards: `alpha_k = (R_k - R_min) / sum(R_j - R_min)`
- Code in `weight_aggr.py:36`: `reward / total_reward` — no min-shift
- These are different formulas. Negative rewards would give negative coefficients with the code formula.

---

## Moderate Finding 1 — "Only Policy Organization Changes" is Overstated (MODERATE)

The paper claims training strategy is the "sole independent variable." This is approximately true for the core comparison but not fully accurate:

- FedRL agents have weights RESET to the averaged model after every episode. SARL/MARL weights persist and accumulate.
- FedProx changes the loss function (different optimization objective)
- Clustered FedRL changes which agents share knowledge
- Partial FedRL changes which layers are updated

**Impact:** The claim "same PPO algorithm, only policy organization changes" is valid for the SARL vs MARL vs basic FedRL comparison. It is overstated when comparing FedRL aggregation variants.

---

## Moderate Finding 2 — Aggregation Variant Fairness (MODERATE)

The 7 FedRL variants get the same number of training iterations, but not the same parameter-sharing constraints:
- Partial FedRL keeps personalized layers — effectively higher model capacity
- Clustered FedRL creates sub-group models — different information flow
- Soft-update blends with prior global — smoother but constrained trajectory

These are valid scientific comparisons but cannot be described as "same training, different aggregation rule only."

---

## Moderate Finding 3 — FedRL Gets Fewer Consecutive Local Updates (MODERATE)

After each episode, FedRL resets weights to the average. This means FedRL agents never accumulate more than 1 episode of consecutive local gradient updates before reset. SARL and MARL accumulate across all 25 episodes.

This is a known property of federated learning (it's the whole point), but it is a real difference in optimization dynamics beyond "policy organization."

---

## Clean: Demand Threading (CLEAN)

VPLPH correctly flows from ExtensionConfig → create_trainer → BaseTrainer → env_config_fn → rand_route_args → generate_random_routes → randomTrips.py period calculation. No issues found.

---

## Action Items

| Priority | Action |
|---|---|
| CRITICAL | Fix or document the evaluation averaging issue for MARL |
| CRITICAL | Reconcile paper tables with results.json — pick one source and be consistent |
| CRITICAL | Fix alpha computation in weight_aggr.py OR fix paper equation |
| HIGH | State communication costs are theoretical estimates |
| HIGH | Clarify in paper that "sole variable" applies to SARL/MARL/FedRL comparison only |
| MEDIUM | Acknowledge that FedRL weight reset is an optimization confound |
| MEDIUM | Clarify aggregation variant fairness limitations |
