# FedRL Traffic Benchmarking — Agent Instructions

## Project Context
This is a **benchmarking framework** for evaluating RL training strategies (SARL, MARL, FedRL) for multi-intersection traffic signal control. The core contribution is the standardized evaluation platform — same environment, same observations, same reward, same algorithm — isolating the training strategy as the only variable.

## Read These Files First
Before doing any work, read:
- `BUILD-RULE.md` — workflow rules (plan mode, verification, elegance, bug fixing)
- `PROJECT-PLAN.md` — full spec: architecture, API contract, page designs, acceptance criteria
- `file-structure.md` — live project tree; **you must update this file as you create files**
- `tasks/todo.md` — current status and bug fix history
- `tasks/bugfix-plan.md` — if it exists, this is your active task

## Hard Rules
- `SUMO-FedRL-main/` — **never modify**. Already copied into BackEnd during initial build.
- `LovableOutput/` — **never modify**. Already copied into FrontEnd during initial build.
- `BackEnd/` is owned exclusively by the backend agent.
- `FrontEnd/` is owned exclusively by the frontend agent.
- Do NOT re-copy from source directories — all files are already in place. Only edit `BackEnd/` and `FrontEnd/` directly.

## Keep file-structure.md Current
Every time you create a file or directory, add it to `file-structure.md` under the correct parent with a brief comment. This is the single source of truth for what has been built. Do not batch updates — add entries as you go.

## Architecture
```
LovableOutput/ (read-only)  →  FrontEnd/  ←WebSocket/REST→  BackEnd/  ←TraCI→  SUMO
SUMO-FedRL-main/ (read-only) →  BackEnd/
```

## Benchmarking Framework — Key Design Principles

The framework guarantees fair comparison across training strategies by controlling 8 layers:

1. **Same simulator** — SUMO 1.26.0, same physics, same binary
2. **Same network** — identical .net.xml loaded for all strategies
3. **Same demand** — identical VPLPH, same randomTrips parameters, same seeds
4. **Same observations** — 14-feature intersection-agnostic vector computed by shared code
5. **Same reward** — r = -(o + h)^2, shared code path
6. **Same algorithm** — PPO with identical hyperparameters (lr=5e-5, batch=4000, clip=0.3)
7. **Same training budget** — identical episode count per config
8. **Same evaluation** — Monte Carlo with identical seeds, metrics from SUMO tripinfo

**The only variable is how policies are organized:** SARL (1 shared), MARL (N independent), FedRL (N with periodic aggregation).

## Key Files

### Training Strategies (the variable under test)
- `BackEnd/seal/trainer/single_agent.py` — SARL: one shared policy, `policy_mapping_fn = lambda _ : "sarl-policy"`
- `BackEnd/seal/trainer/multi_agent.py` — MARL: per-intersection policies, no aggregation during training
- `BackEnd/seal/trainer/fed_agent.py` — FedRL: per-intersection policies + reward-weighted FedAvg every episode
- `BackEnd/seal/trainer/fedprox_policy.py` — FedProx extension: proximal loss term

### Shared Infrastructure (held constant)
- `BackEnd/seal/sumo/env.py` — SUMO environment, observation computation, reward function
- `BackEnd/seal/sumo/kernel/trafficlight/light.py` — TraCI wrappers, observation features
- `BackEnd/seal/sumo/config.py` — Feature indices (14-dim observation space)
- `BackEnd/api/training_runner.py` — Trainer factory, topology map, PPO config
- `BackEnd/api/evaluation/monte_carlo.py` — Monte Carlo evaluation pipeline
- `BackEnd/scripts/run_all_training.py` — Experiment campaign runner (incremental save, resume-safe)
- `BackEnd/scripts/generate_all_figures.py` — Figure generation from results

### Results
- `BackEnd/results/campaigns/baseline/` — Baseline evaluation (pre-trained weights, 10 configs)
- `BackEnd/results/campaigns/training-curves/` — Training + evaluation (9 configs: 3 strategies x 2 topologies + FedProx + ToD)
- `BackEnd/results/figures/` — Generated charts

### Dashboard
- `BackEnd/api/routes/simulate.py` → `_run_mock_simulation()` — vehicle positions, TLS states via WebSocket
- `BackEnd/api/routes/train.py` → `_run_mock_training()` — reward curves via WebSocket
- `FrontEnd/src/components/SimCanvas.tsx` — Canvas 2D renderer
- `FrontEnd/src/hooks/useSimStream.ts` — Simulation WebSocket hook
- `FrontEnd/src/hooks/useTrainStream.ts` — Training WebSocket hook

## Known Patterns (avoid regressions)
- Recharts `Line` with frequent data updates needs `isAnimationActive={false}` or the animation restarts endlessly
- SimCanvas uses a single rAF loop with frame interpolation — do not add a second draw loop
- `ErrorBoundary` wraps both Simulation and Compare pages — do not remove
- TLS phases `["GGrr", "rrGG", "yyrr", "rryy"]` — first half = horizontal, second half = vertical
- Vehicle queuing should only trigger near intersections (use `_is_near_intersection` helper)
- `run_all_training.py` saves after every config — safe to interrupt and resume with `--resume`
- FedProx `__init__` must set `_fedprox_mu` BEFORE `super().__init__()` (TorchPolicyV2 calls loss() during setup)
- MC evaluation must pass `use_time_encoding` from ExtensionConfig to MCConfig for ToD-trained weights

## Upcoming Work (Second Half)
- Multi-demand evaluation: low (150 VPLPH), medium (360), high (600) — characterize strategy behavior under varying load
- Trade-off matrix: define when each strategy wins/loses based on evidence
- Aggregation variants: reward-weighted FedAvg vs naive FedAvg vs FedProx
- Wilcoxon signed-rank significance tests for pairwise comparisons

## Task Tracking
- Update `tasks/todo.md` with progress as you go.
- Update `tasks/lessons.md` after any correction or unexpected discovery.

## Ports
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
- CORS: backend allows `http://localhost:5173`
