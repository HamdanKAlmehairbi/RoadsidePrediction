# SEAL Dashboard — Build Plan

## Team: 2 agents (Backend + Frontend)

### Backend Agent Tasks
- [x] Copy `seal/`, `configs/SMARTCOMP/`, `example_weights/ICCPS/Final/`, `netfiles.py` from SUMO-FedRL-main/
- [x] Create `BackEnd/requirements.txt`
- [x] Create `BackEnd/api/main.py` — FastAPI app with CORS for localhost:5173
- [x] Create `BackEnd/api/jobs.py` — in-memory job store
- [x] Create `BackEnd/api/routes/networks.py` — GET /api/networks, GET /api/network/{topology}
- [x] Create `BackEnd/api/routes/weights.py` — GET /api/weights
- [x] Create `BackEnd/api/routes/simulate.py` — POST /api/simulate
- [x] Create `BackEnd/api/routes/train.py` — POST /api/train
- [x] Create `BackEnd/api/routes/results.py` — GET /api/results/{job_id}
- [x] Create `BackEnd/api/ws/simulate.py` — WS /ws/simulate/{job_id}
- [x] Create `BackEnd/api/ws/train.py` — WS /ws/train/{job_id}
- [x] Create `BackEnd/README.md` — SUMO install instructions
- [x] Validate: uvicorn starts, curl tests pass

### Frontend Agent Tasks
- [x] Copy `LovableOutput/seal-traffic-flow-main/` into `FrontEnd/`
- [x] Create `FrontEnd/src/lib/api.ts` — typed REST wrappers
- [x] Create `FrontEnd/src/hooks/useSimStream.ts` — WebSocket hook
- [x] Create `FrontEnd/src/hooks/useTrainStream.ts` — WebSocket hook
- [x] Create `FrontEnd/src/components/SimCanvas.tsx` — Canvas 2D renderer
- [x] Wire `FrontEnd/src/pages/Simulation.tsx` — replace SVG placeholder with SimCanvas
- [x] Wire `FrontEnd/src/pages/Compare.tsx` — two SimCanvas + two streams
- [x] Wire `FrontEnd/src/pages/Training.tsx` — POST /api/train + useTrainStream
- [x] Wire `FrontEnd/src/pages/Communication.tsx` — GET /api/results
- [x] Validate: npm run build passes, dev server starts

### Integration / Lead Validation
- [x] Backend on :8000, frontend on :5173
- [x] Simulation page: Run simulation → vehicles animate on canvas
- [x] Compare page: Two canvases animate simultaneously
- [x] Training page: Chart extends episode by episode
- [x] No modifications to SUMO-FedRL-main/ or LovableOutput/
