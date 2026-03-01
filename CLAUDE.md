# SEAL Dashboard — Agent Instructions

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

Key files:
- **Mock simulation**: `BackEnd/api/routes/simulate.py` → `_run_mock_simulation()` — generates vehicle positions, TLS states, streams via WebSocket
- **Mock training**: `BackEnd/api/routes/train.py` → `_run_mock_training()` — generates reward curves, streams via WebSocket
- **Canvas renderer**: `FrontEnd/src/components/SimCanvas.tsx` — Canvas 2D: roads, vehicles (white=moving, red=halted), TLS split semicircles
- **Sim WebSocket hook**: `FrontEnd/src/hooks/useSimStream.ts`
- **Train WebSocket hook**: `FrontEnd/src/hooks/useTrainStream.ts`

## Known Patterns (avoid regressions)
- Recharts `Line` with frequent data updates needs `isAnimationActive={false}` or the animation restarts endlessly
- SimCanvas uses a single rAF loop with frame interpolation — do not add a second draw loop
- `ErrorBoundary` wraps both Simulation and Compare pages — do not remove
- TLS phases `["GGrr", "rrGG", "yyrr", "rryy"]` — first half = horizontal, second half = vertical
- Vehicle queuing should only trigger near intersections (use `_is_near_intersection` helper)

## Task Tracking
- Update `tasks/todo.md` with progress as you go.
- Update `tasks/lessons.md` after any correction or unexpected discovery.

## Ports
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
- CORS: backend allows `http://localhost:5173`
