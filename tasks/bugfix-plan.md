# SEAL Dashboard — Bug Fix Plan (Round 2)

> Use with: `/build-with-agent-team tasks/bugfix-plan.md 2`

---

## Context Files (Read First)

- `CLAUDE.md` — hard rules, file-structure update requirement
- `BUILD-RULE.md` — workflow: plan mode, verification, lessons.md
- `PROJECT-PLAN.md` — full spec, API contract, page designs
- `file-structure.md` — current project tree

## Architecture

```
LovableOutput/ (read-only)  →  FrontEnd/  ←WebSocket/REST→  BackEnd/  ←TraCI→  SUMO
SUMO-FedRL-main/ (read-only) →  BackEnd/
```

- Backend: `http://localhost:8000` (FastAPI + uvicorn)
- Frontend: `http://localhost:5173` (React + Vite)

---

## Bug 1 — Simulation page clears/resets after simulation ends

**What happens:** The Simulation page no longer blackscreens (the ErrorBoundary fixed that), but when the simulation finishes, the canvas empties — all vehicles disappear, traffic light colours reset, and it returns to showing just the bare road grid. The final state should freeze on screen.

**Root cause analysis:**

The `done` frame arrives from the backend with `done: true`. In `useSimStream.ts` line 30–33:
```typescript
if (frame.done) {
  setIsDone(true);
  ws.close();
}
```
The WebSocket closes, which triggers `onclose` → `setIsConnected(false)`. The `latestFrame` state still holds the done frame with vehicles — so far so good.

However the issue is likely that after the WS closes, the `latestFrame` is the `done` frame (which DOES have vehicles from the backend — see `BackEnd/api/routes/simulate.py` lines 272–286), but something in the React render cycle or SimCanvas causes the vehicles to not render. Investigate:

1. **SimCanvas early return** (`SimCanvas.tsx` line 168–171): `if (!currFrame || !currFrame.vehicles || currFrame.vehicles.length === 0)` — if the done frame's vehicles array is somehow empty after serialization, the canvas would show roads only and return before drawing vehicles
2. **Frame ref getting cleared**: The `currFrameRef` is a ref, but if the component re-mounts for any reason after done, the ref resets to `null`
3. **State update timing**: The `setLatestFrame(frame)` and `ws.close()` happen synchronously in the same handler. Check if the close triggers a re-render that somehow nulls out state before the frame is painted

**Files to examine and fix:**
- `FrontEnd/src/hooks/useSimStream.ts` — the done/close handling
- `FrontEnd/src/components/SimCanvas.tsx` — the early return condition at line 168
- `FrontEnd/src/pages/Simulation.tsx` — check if any state change on `isDone` causes re-render that affects frame display

**The fix should ensure:**
- When `done: true` arrives, `latestFrame` is preserved and the canvas keeps drawing the final frame indefinitely
- The rAF loop keeps running after done (it already does — it has `[]` deps), but make sure `currFrameRef` isn't nulled
- Consider: stop shifting `prevFrameRef`/`currFrameRef` once done, so the canvas freezes on the last rendered state

**Acceptance criteria:**
- [ ] After simulation completes, vehicles remain visible on canvas in their final positions
- [ ] Traffic light colours remain showing final state
- [ ] Metrics sidebar still shows final values (halted, speed, reward, timestep)
- [ ] "Done" label appears in controls bar
- [ ] User can click "Run Simulation" again to start a new run

---

## Bug 2 — Compare page blackscreens

**What happens:** The Compare page (`/compare`) still goes completely blank after starting a simulation — the entire page disappears to background colour `#0f1117`.

**Root cause:** `Compare.tsx` does NOT have an `<ErrorBoundary>` wrapper. The Simulation page was fixed by adding one (see `Simulation.tsx` lines 69 and 173), but the same fix was never applied to Compare.

When either `useSimStream` or `SimCanvas` throws an error on the Compare page, there is no error boundary to catch it. The error propagates up to the React root and unmounts the entire page.

**Files to fix:**
- `FrontEnd/src/pages/Compare.tsx` — wrap the return JSX in `<ErrorBoundary>` (same pattern as `Simulation.tsx`)
- `FrontEnd/src/components/ErrorBoundary.tsx` — verify it exists and renders a fallback UI instead of blank

Additionally, `Compare.tsx` has TWO SimCanvas and TWO useSimStream instances running simultaneously — double the chance of an uncaught error. The fix from Bug 1 (preserving frame state on done) must also work correctly when two streams finish at different times.

**The fix:**
1. Add `<ErrorBoundary>` wrapper around the Compare page's JSX (import already exists in the project — check `FrontEnd/src/components/ErrorBoundary.tsx`)
2. Ensure both streams' done states are handled independently — one finishing should not clear the other's canvas
3. Test that the page stays rendered even if one stream errors

**Acceptance criteria:**
- [ ] Compare page stays fully rendered throughout simulation and after both finish
- [ ] Both canvases show vehicles simultaneously
- [ ] If one simulation finishes before the other, its canvas freezes while the other continues
- [ ] No console errors

---

## Bug 3 — Traffic lights never show red

**What happens:** Intersection circles only appear green or yellow, never red. This makes it look like traffic lights aren't working properly.

**Root cause:** The `tlsColor` function in `SimCanvas.tsx` lines 11–17 is too simplistic:
```typescript
function tlsColor(state: string): string {
  if (!state) return "#6b7280";
  const s = state.toLowerCase();
  if (s.includes("g")) return "#22c55e";   // ANY green lane → whole intersection green
  if (s.includes("y")) return "#eab308";
  return "#ef4444";                         // only "rrrr" reaches here
}
```

SUMO TLS state strings encode **per-lane** phases. For example `"GGrr"` means lanes 0–1 are green, lanes 2–3 are red. The mock simulation cycles through `["GGrr", "rrGG", "yyrr", "rryy"]` — every single one contains either `g` or `y`, so the function NEVER returns red.

**The fix:** Change `tlsColor` to show the **dominant** phase — count the characters and pick whichever has the majority. Or better yet, render the intersection as a split circle (top half = one direction's phase, bottom half = other direction), so you can see that some lanes are green while others are red. This is more realistic and visually informative.

A simpler first approach: take the FIRST half of the state string as one direction and the second half as the other, then render two semicircles. For `"GGrr"`: left semicircle green, right semicircle red.

**Files to fix:**
- `FrontEnd/src/components/SimCanvas.tsx` — replace `tlsColor` function and the intersection drawing code (lines 150–165)

**Acceptance criteria:**
- [ ] Intersections visually show that some directions are red while others are green
- [ ] Phase changes are visible as the simulation runs (intersections alternate between states)
- [ ] Yellow phase is visible during transitions

---

## Bug 4 — Cars never stop at red lights (mock mode)

**What happens:** Vehicles move continuously and never stop at intersections, even when the traffic light is red. This makes the simulation look unrealistic.

**Root cause:** The mock simulation in `BackEnd/api/routes/simulate.py` (function `_run_mock_simulation`, line 161–297) does not simulate traffic light interaction. Vehicles just move in a straight line with random speed adjustments (line 229: `v["speed"] = max(0.0, min(13.89, v["speed"] + rng.uniform(-0.5, 0.5)))`). They never check whether their nearest intersection's light is red.

**The fix:** In the mock simulation loop, for each vehicle:
1. Find the nearest intersection ahead of the vehicle (in its direction of travel)
2. Check that intersection's current TLS phase for the vehicle's travel direction
3. If the phase is red (`r`) and the vehicle is within ~20 units of the intersection, decelerate the vehicle toward 0 speed
4. If the phase is green (`G` or `g`), let the vehicle accelerate back to normal speed
5. If the vehicle's speed drops below 0.5, it counts as halted (this is already tracked at line 248)

This makes the mock simulation visually convincing: vehicles approach intersections, stop on red, and go on green. The `total_halted` metric will now reflect actual stopped vehicles, and the red car colouring (speed < 0.5 → `#ef4444`) will appear at intersections naturally.

**Files to fix:**
- `BackEnd/api/routes/simulate.py` — modify `_run_mock_simulation` to add intersection-aware stopping logic

**Acceptance criteria:**
- [ ] Vehicles visibly slow down and stop near intersections when the light is red
- [ ] Vehicles resume moving when the light turns green
- [ ] Halted vehicles turn red on the canvas (this already works via `speed < 0.5` in SimCanvas)
- [ ] The `total_halted` metric in the sidebar reflects actual stopped vehicles
- [ ] Mock simulation still runs smoothly without performance issues

---

## Note: Red cars are halted vehicles (not a bug)

The user noticed cars sometimes turn red. This is **intentional** per the design system in `PROJECT-PLAN.md`:
- Vehicle colour: `#e2e8f0` (white/light grey) — moving normally
- Halted vehicle: `#ef4444` (red) — speed < 0.5 m/s

This is implemented in `SimCanvas.tsx` line 208: `ctx.fillStyle = drawSpeed < 0.5 ? "#ef4444" : "#e2e8f0"`

However, there is no **legend** explaining this. The frontend agent should add a small legend overlay on the canvas or in the sidebar explaining colour codes:
- White car = moving
- Red car = halted (speed < 0.5 m/s)
- Green circle = green light
- Yellow circle = yellow light
- Red circle = red light

**File to fix:**
- `FrontEnd/src/pages/Simulation.tsx` — add a small legend card in the sidebar or as a canvas overlay
- `FrontEnd/src/pages/Compare.tsx` — same legend if space permits

---

## Agent Ownership

### Backend Agent
- **Owns:** `BackEnd/`
- **Does NOT touch:** `FrontEnd/`, `SUMO-FedRL-main/`, `LovableOutput/`
- **Priority order:**
  1. **Bug 4** — Add intersection-aware stopping to mock simulation so vehicles stop at red lights and go on green
  2. Verify the `done` frame (step 300) includes full vehicle + TLS data (it currently does — confirm no regression)
  3. Verify WebSocket closes cleanly after done frame is sent

### Frontend Agent
- **Owns:** `FrontEnd/`
- **Does NOT touch:** `BackEnd/`, `SUMO-FedRL-main/`, `LovableOutput/`
- **Priority order:**
  1. **Bug 1** — Fix simulation page reset: preserve final frame on canvas after done
  2. **Bug 2** — Add `<ErrorBoundary>` to Compare page + ensure both stream done states work independently
  3. **Bug 3** — Fix `tlsColor` to show red phases (split circle or dominant-phase logic)
  4. **Legend** — Add colour legend to Simulation and Compare pages
  5. Run `npm run build` — must pass with 0 errors

---

## Validation

### After Frontend Fixes
```bash
cd FrontEnd && npm run build   # 0 errors
# Start both servers
# Simulation page:
#   - Run simulation → vehicles animate → simulation ends → vehicles STAY on canvas
#   - Traffic lights show red, green, and yellow phases
#   - Legend is visible explaining colours
# Compare page:
#   - Run Both → both canvases animate → no blackscreen
#   - When one finishes first, its canvas freezes, other continues
#   - Page stays rendered after both finish
```

### After Backend Fixes
```bash
cd BackEnd
python -m uvicorn api.main:app --port 8000 --reload
# Connect to WS /ws/simulate/{job_id}
# Verify: vehicles slow down and stop near red-light intersections
# Verify: halted count in metrics increases when lights are red
# Verify: done frame still includes all vehicle + TLS data
```

### End-to-End
1. Backend on `:8000`, frontend on `:5173`
2. Simulation: vehicles stop at red lights, move on green, canvas freezes at end with final state
3. Compare: both canvases work, no blackscreen, both freeze at end
4. All 5 pages — no blank screens, no console errors
5. `SUMO-FedRL-main/` and `LovableOutput/` have zero modifications
