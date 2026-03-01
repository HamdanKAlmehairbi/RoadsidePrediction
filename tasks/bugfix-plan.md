# SEAL Dashboard — Bug Fix Plan

> Use with: `/build-with-agent-team tasks/bugfix-plan.md 2`

---

## Context Files (Read First)

- `CLAUDE.md` — hard rules, file-structure update requirement
- `BUILD-RULE.md` — workflow: plan mode, verification, lessons.md
- `PROJECT-PLAN.md` — full spec, API contract, page designs, acceptance criteria
- `file-structure.md` — current project tree

## Architecture

```
LovableOutput/ (read-only)  →  FrontEnd/  ←WebSocket/REST→  BackEnd/  ←TraCI→  SUMO
SUMO-FedRL-main/ (read-only) →  BackEnd/
```

- Backend: `http://localhost:8000` (FastAPI + uvicorn)
- Frontend: `http://localhost:5173` (React + Vite)

---

## Bug 1 — CRITICAL: Simulation page goes blank (grey screen)

**What happens:** On the Simulation page (`/simulation`), a few seconds after starting a simulation, the entire page turns to the background colour (`#0f1117`). Everything disappears — the dashboard layout, top nav, sidebar, controls, canvas, metrics — all gone. Only the blank background remains.

**Where to look:**
- `FrontEnd/src/pages/Simulation.tsx` — the page component; likely an unhandled error or state change that unmounts the entire layout
- `FrontEnd/src/hooks/useSimStream.ts` — WebSocket hook; may throw or set state that triggers a full re-render / error boundary
- `FrontEnd/src/components/SimCanvas.tsx` — Canvas 2D renderer; may throw during animation frame
- `FrontEnd/src/components/DashboardLayout.tsx` — layout wrapper; check if an error propagates up and kills it
- `BackEnd/api/ws/simulate.py` — WebSocket endpoint; check if it closes unexpectedly or sends malformed frames

**Likely root causes to investigate:**
1. Uncaught exception in SimCanvas `requestAnimationFrame` loop or useSimStream that propagates up and unmounts the page
2. WebSocket `onclose`/`onerror` handler sets state that causes the page to go blank
3. React error boundary (or lack of one) catching a render error and showing nothing
4. Backend WebSocket closes after simulation ends (`done: true`) and frontend doesn't handle the close gracefully
5. Canvas ref becomes null after a re-render, causing a crash on the next animation frame

**Acceptance criteria:**
- [ ] Simulation page remains fully rendered (nav, sidebar, controls, metrics) throughout the entire simulation and after it ends
- [ ] When simulation completes (`done: true`), the page stays visible with final state shown
- [ ] No uncaught errors in the browser console during or after simulation

---

## Bug 2 — LOW PRIORITY: Vehicle jitter during simulation

**What happens:** Cars on the SimCanvas jitter/stutter as they move instead of gliding smoothly along roads. The movement is visually choppy.

**Where to look:**
- `FrontEnd/src/components/SimCanvas.tsx` — rendering loop; check how vehicle positions are drawn each frame
- `FrontEnd/src/hooks/useSimStream.ts` — check if frames are being buffered and interpolated, or just painted raw
- `PROJECT-PLAN.md` line 427 — specifies: "The frontend should interpolate vehicle positions between frames for smooth animation"

**Likely root causes:**
1. No interpolation between WebSocket frames — vehicles jump directly from position A to position B every ~100ms instead of smoothly transitioning over 6-7 `requestAnimationFrame` ticks
2. Canvas is being cleared and redrawn on every WebSocket message instead of on every `requestAnimationFrame` tick
3. Vehicle positions are snapping to integer pixels (rounding) instead of using sub-pixel rendering
4. The animation loop is tied to WebSocket message rate (~10fps) instead of `requestAnimationFrame` rate (~60fps)

**The fix (from the spec):**
- Buffer the latest two frames from WebSocket
- On each `requestAnimationFrame` tick, linearly interpolate each vehicle's `x`, `y`, and `angle` between the previous frame and the current frame based on elapsed time
- This gives smooth 60fps visual movement even though data arrives at ~10fps

**Acceptance criteria:**
- [ ] Vehicles glide smoothly across the canvas without visible jumping or stuttering
- [ ] Animation runs at browser's `requestAnimationFrame` rate (~60fps)
- [ ] Interpolation handles edge cases: new vehicles appearing mid-frame, vehicles disappearing

---

## Agent Ownership

### Backend Agent
- **Owns:** `BackEnd/`
- **Does NOT touch:** `FrontEnd/`, `SUMO-FedRL-main/`, `LovableOutput/`
- **Focus:** Investigate whether the WebSocket at `BackEnd/api/ws/simulate.py` sends malformed data, closes unexpectedly, or has timing issues that could cause the frontend to crash. Fix if needed.

### Frontend Agent
- **Owns:** `FrontEnd/`
- **Does NOT touch:** `BackEnd/`, `SUMO-FedRL-main/`, `LovableOutput/`
- **Focus:** Fix Bug 1 (grey screen) first — this is the critical issue. Then fix Bug 2 (jitter) by adding frame interpolation to SimCanvas. Add error boundaries to prevent full-page crashes.

## Validation

### After Bug 1 Fix
```bash
cd FrontEnd && npm run build   # 0 errors
# Start both servers, open /simulation
# Run a simulation — page must stay fully rendered for the entire duration
# After simulation ends — page must still be visible with final state
# Check browser console — no uncaught errors
```

### After Bug 2 Fix
```bash
# Start both servers, open /simulation
# Run a simulation — vehicles should glide smoothly, no visible jitter
# Compare visual smoothness at 1× and 5× speed
```

### End-to-End
1. Backend on `:8000`, frontend on `:5173`
2. Navigate all 5 pages — no blank screens, no console errors
3. Run simulation start to finish — page stays rendered, vehicles move smoothly
4. Run Compare page — both canvases work, no blank screen
5. `SUMO-FedRL-main/` and `LovableOutput/` have zero modifications
