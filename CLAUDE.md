# RL Training Strategy Benchmarking — Agent Instructions

## Project Context
This is a **benchmarking framework** for evaluating RL training strategies for multi-intersection traffic signal control. 10 strategies are compared across a spectrum from fully independent to fully shared — same environment, same observations, same reward, same algorithm — isolating the training strategy as the only variable.

## Read These Files First
- `CLAUDE.md` — this file (hard rules, key patterns)
- `PROJECT_SUMMARY.md` — full project knowledge: architecture, strategies, setup, results, known issues

## Hard Rules
- `BackEnd/` is owned exclusively by the backend agent.
- `FrontEnd/` is owned exclusively by the frontend agent.

## Training Strategies (10 total)

```
Independent <----------------------------------------------------> Shared

MARL -> MeanField -> CTDE -> Gossip -> HierFed -> FedDistill -> FedRL -> SARL
 (0)     (obs)    (critic)  (mesh)    (tree)     (logits)     (star)  (full)
```

| # | Strategy | Trainer Type | File |
|---|----------|-------------|------|
| 1 | SARL | `SARL` | `seal/trainer/single_agent.py` |
| 2 | MARL | `MARL` | `seal/trainer/multi_agent.py` |
| 3 | FedRL | `FedRL` | `seal/trainer/fed_agent.py` |
| 4 | Gossip | `Gossip` | `seal/trainer/gossip_agent.py` |
| 5 | HierFed | `HierFed` | `seal/trainer/hierfed_agent.py` |
| 6 | FedDistill | `FedDistill` | `seal/trainer/feddistill_agent.py` |
| 7 | MeanField | `MeanField` | `seal/trainer/mean_field_agent.py` |
| 8 | CTDE | `CTDE` | `seal/trainer/ctde_agent.py` |
| 9 | fixed-time | `fixed-time` | (eval only) |
| 10 | max-pressure | `max-pressure` | (eval only) |

## Known Patterns (avoid regressions)
- FedProx/FedDistill `__init__` must set custom attributes BEFORE `super().__init__()` (TorchPolicyV2 calls loss() during setup)
- `save_test_policy()` must save `__multi_policy__` format for MARL/Gossip/HierFed/FedDistill/CTDE/MeanField
- CTDE eval zero-pads global state portion; `__ctde__` flag in pickle triggers this in runner.py
- MeanField eval uses `MeanFieldSumoEnv` to produce augmented observations
- `save_campaign_results()` appends to existing results.json (not overwrites) for multi-sweep runs
- `training_data` initialized in `BaseTrainer.__init__()` so subclasses can write to it before `on_setup()`
- FedRL `episode_data` reset after each aggregation to avoid cumulative weighting bias
- `demand_dir` in `base.py` is `d{vplph}_s{training_seed}` — weights scoped by demand AND seed
- `abstract_env.py` cleanup is in `__del__` not `close()` — moving it back triggers SARL race condition
- Config names always include `_d{demand}_s{seed}` suffix for unique resume keys
- `avg_waiting_time` is misleading when trip completion rates differ (survivorship bias)

## Ports
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
