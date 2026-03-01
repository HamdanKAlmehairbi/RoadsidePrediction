# SEAL Dashboard — Agent Instructions

## Read These Files First
Before doing any work, read:
- `BUILD-RULE.md` — workflow rules (plan mode, verification, elegance, bug fixing)
- `PROJECT-PLAN.md` — full spec: architecture, API contract, page designs, acceptance criteria
- `file-structure.md` — live project tree; **you must update this file as you create files**

## Hard Rules
- `SUMO-FedRL-main/` — **never modify**. Backend copies from it, never edits it.
- `LovableOutput/` — **never modify**. Frontend copies from it, never edits it.
- `BackEnd/` is owned exclusively by the backend agent.
- `FrontEnd/` is owned exclusively by the frontend agent.

## Keep file-structure.md Current
Every time you create a file or directory, add it to `file-structure.md` under the correct parent with a brief comment. This is the single source of truth for what has been built. Do not batch updates — add entries as you go.

## Architecture in One Line
```
LovableOutput/ (read-only)  →  FrontEnd/  ←WebSocket/REST→  BackEnd/  ←TraCI→  SUMO
SUMO-FedRL-main/ (read-only) →  BackEnd/
```

## Task Tracking
- Write your plan to `tasks/todo.md` before starting implementation.
- Update `tasks/lessons.md` after any correction or unexpected discovery.

## Ports
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
- CORS: backend allows `http://localhost:5173`
