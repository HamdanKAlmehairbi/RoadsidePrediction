# SEAL Dashboard — Bug Fix Plan (Round 4)

> Use with: `/build-with-agent-team tasks/bugfix-plan.md 1`
>
> **Single agent (backend only)** — all issues are in `BackEnd/api/routes/simulate.py`, function `_run_mock_simulation`.

---

## Context Files (Read First)

- `CLAUDE.md` — hard rules, file-structure update requirement
- `BUILD-RULE.md` — workflow: plan mode, verification, lessons.md
- `PROJECT-PLAN.md` — full spec, API contract, page designs
- `file-structure.md` — current project tree

## What's Already Working (Do NOT Break)

- ErrorBoundary on Simulation and Compare pages
- Split semicircle TLS rendering (intersections show red/green per direction)
- Colour legend in Simulation sidebar and Compare page
- Frame freeze on done — SimCanvas keeps final frame after simulation ends
- Compare page with two simultaneous streams
- Red-light stopping logic (vehicles do stop at red lights now)
- Vehicle queuing behind stopped vehicles
- Yellow = stop for approaching vehicles
- Phase duration at `step // 80` (correct)
- Detection range at 45 units (correct)
- Hard stop at < 8 units (correct)
- Live Training chart updates in real-time (`isAnimationActive={false}`)

---

## Bug 1: Vehicles stop mid-road between intersections

**What happens:** Red (halted) vehicles appear stopped in the middle of road segments, far from any intersection. They shouldn't be stopping there — it looks unnatural.

**File:** `BackEnd/api/routes/simulate.py` — `_run_mock_simulation` (line 161+)

### Root Cause A — Overly aggressive queuing logic

```python
# Lines 257-278: current queuing
for other in vehicles:
    if other["id"] == v["id"]:
        continue
    # ... checks if other vehicle is ahead within 15 units and speed < 0.5
    if same_road and 0 < ahead_dist < 15 and other["speed"] < 0.5:
        v["speed"] = max(0.0, v["speed"] - 2.0)
        break
```

The queuing check brakes behind ANY halted vehicle, even ones that stopped for no good reason (e.g., random perturbation pushed speed below 0.5, or a vehicle that just respawned from wrap-around at speed 0). This creates chain-reaction stops mid-road where no intersection exists.

**Fix:** Only queue behind a vehicle that is itself **near an intersection** (within 50 units of one). Add a helper check:

```python
def _is_near_intersection(veh, intersections, threshold=50):
    """Return True if vehicle is within threshold distance of any intersection."""
    for inter in intersections:
        dx = veh["x"] - inter["x"]
        dy = veh["y"] - inter["y"]
        if (dx * dx + dy * dy) < threshold * threshold:
            return True
    return False
```

Then change the queuing condition to:
```python
if same_road and 0 < ahead_dist < 15 and other["speed"] < 0.5 and _is_near_intersection(other, intersections):
    v["speed"] = max(0.0, v["speed"] - 2.0)
    break
```

This way vehicles only queue behind cars that stopped near an intersection (i.e., legitimately at a red light), not behind random mid-road stops.

### Root Cause B — Wrap-around respawns at random positions with low speed

```python
# Lines 290-301: wrap-around
if v["x"] < 0 or v["x"] > max_coord:
    col = rng.randint(1, grid_n)
    v["x"] = col * spacing + rng.choice([-1.6, 1.6])
    v["y"] = rng.uniform(spacing, max_coord - spacing)
    v["angle"] = rng.choice([0.0, 180.0])
```

When a vehicle wraps around, it keeps its current speed (which might be 0 or very low from braking). The next step, other vehicles see a halted car mid-road and queue behind it.

**Fix:** Reset speed to a reasonable value on wrap-around. After each wrap-around block, add:
```python
v["speed"] = rng.uniform(8.0, 13.89)
```

This ensures respawned vehicles enter the network at cruising speed, not halted.

---

## Bug 2: Vehicles stay stopped too long after green

**What happens:** When a red light turns green, queued vehicles are slow to clear. They stay red (halted) for several steps after the light changes, because acceleration is only `+0.5/step`. From speed 0, it takes ~28 steps (0.45 seconds) to reach full speed — and during the first 1-2 steps the vehicle is still below the 0.5 threshold so it keeps rendering as red/halted, and vehicles behind it are still queuing.

**Fix:** Increase green-phase acceleration from `0.5` to `1.5`:

```python
# Line 251 — change:
v["speed"] = min(13.89, v["speed"] + 0.5)
# To:
v["speed"] = min(13.89, v["speed"] + 1.5)
```

Also increase the general cruise acceleration (line 254) to match:
```python
# Line 254 — change:
v["speed"] = min(13.89, v["speed"] + 0.5)
# To:
v["speed"] = min(13.89, v["speed"] + 1.5)
```

This means vehicles reach full speed in ~10 steps instead of ~28, clearing intersections much faster after green. Queued vehicles break free quickly because the leader crosses the 0.5 speed threshold within 1 step.

---

## Summary of All Changes

```
File: BackEnd/api/routes/simulate.py — function _run_mock_simulation

1. Add _is_near_intersection() helper function (new, above _run_mock_simulation)
2. Queuing: only queue behind vehicles near an intersection
3. Wrap-around: reset speed to 8.0-13.89 on respawn (both X and Y wraps)
4. Acceleration: increase from 0.5 to 1.5 (both green-phase and cruise)
```

---

## Agent Ownership

### Backend Agent (sole agent)
- **Owns:** `BackEnd/`
- **Does NOT touch:** `FrontEnd/`, `SUMO-FedRL-main/`, `LovableOutput/`
- **Scope:** Only modify `BackEnd/api/routes/simulate.py`
- **Do NOT change:** the TraCI simulation path (`_run_traci_simulation`), REST endpoints, WebSocket handlers, or any frontend files

### Priority Order
1. Add `_is_near_intersection` helper
2. Update queuing condition to use it
3. Reset speed on wrap-around respawn
4. Increase acceleration from 0.5 to 1.5
5. Verify the done frame still includes all data
6. Test that simulation completes in reasonable time (~5 seconds)

---

## Validation

```bash
cd BackEnd
python -m uvicorn api.main:app --port 8000 --reload
```

### What to verify visually (via frontend at localhost:5173):
1. **No mid-road stops** — halted (red) vehicles should only appear near intersections, not in the middle of road segments
2. **Fast green clearance** — when a light turns green, queued vehicles should start moving within 1-2 frames and clear the intersection quickly
3. **Queuing still works** — vehicles still form short queues at red-light intersections (the fix didn't remove queuing, just made it smarter)
4. **Respawned vehicles are moving** — no stationary vehicles appearing out of nowhere mid-road
5. **Traffic flow looks natural** — vehicles cruise between intersections, slow at red lights, stop near the intersection, then clear quickly on green

### What NOT to break:
- [ ] Red-light stopping still works (vehicles stop at red/yellow lights)
- [ ] Hard stop near intersection (< 8 units) still works
- [ ] Phase duration unchanged (step // 80)
- [ ] Detection range unchanged (45 units)
- [ ] Simulation page: ErrorBoundary, legend, frame freeze on done
- [ ] Compare page: ErrorBoundary, both canvases render
- [ ] WebSocket contract unchanged — frame JSON shape identical
- [ ] `SUMO-FedRL-main/` and `LovableOutput/` have zero modifications
