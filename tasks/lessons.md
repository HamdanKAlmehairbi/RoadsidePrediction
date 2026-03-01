# Lessons Learned

## Frontend

- **Recharts animation kills live updates**: `Line` component defaults to `isAnimationActive={true}` with ~1500ms animation. When data arrives faster than the animation duration (e.g., every 150ms during training), the animation restarts endlessly and only the first data point is visible. Fix: `isAnimationActive={false}` on any Line receiving real-time data.

- **SimCanvas must use a single rAF loop**: Adding multiple `requestAnimationFrame` loops causes double-drawing and flicker. The one loop in the `useEffect([], ...)` is self-sustaining — never add another.

- **ErrorBoundary prevents page blackscreens**: Any uncaught error in SimCanvas or hooks will unmount the entire page without an ErrorBoundary. Both Simulation and Compare pages need it.

- **Frame freeze on done**: When `currFrameRef.current?.done && frame.done`, skip updating frame refs. This keeps the last frame visible after simulation ends instead of clearing to black.

## Backend (Mock Simulation)

- **Phase duration matters visually**: `step // 30` = 0.48s per phase — too fast to see vehicles stop. `step // 80` = 1.3s — enough time for visible stopping and queuing.

- **Detection range determines stop location**: 20 units = vehicles brake too late, stop mid-road. 45 units = vehicles start braking at ~half the road segment, stop near the intersection.

- **Hard stop prevents overshoot**: Without `if best_dist < 8: speed = 0`, vehicles at low speed can drift past the intersection and then accelerate away on the "no intersection nearby" branch.

- **Yellow must equal stop**: `all(c == 'r')` misses yellow phases entirely. `all(c in ('r', 'y'))` treats yellow as stop for approaching vehicles.

- **Queuing needs proximity filter**: Without `_is_near_intersection()`, vehicles queue behind ANY halted car — even ones that stopped randomly mid-road. This creates chain-reaction stops far from intersections.

- **Wrap-around must reset speed**: Vehicles that wrap around the grid edge keep their old speed (possibly 0 from braking). This creates phantom halted vehicles mid-road. Reset to `rng.uniform(8.0, 13.89)` on respawn.

- **Acceleration rate affects queue clearance**: 0.5/step is too slow — vehicles stay "halted" (< 0.5 speed) for multiple frames after green. 1.5/step clears queues within 1-2 frames.

## Project Setup

- **settings.json env vars**: Must be under `"env"` key, not top-level. Value must be string `"1"`, not integer `1`.

- **Agent team source rules**: `SUMO-FedRL-main/` and `LovableOutput/` are read-only sources that were copied during initial build. Agents should never re-copy or modify them — only edit `BackEnd/` and `FrontEnd/` directly.
