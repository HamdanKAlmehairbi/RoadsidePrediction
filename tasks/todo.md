# SEAL Dashboard — Status

## Initial Build (Complete)

All backend and frontend tasks from the original build are done. Both servers start, all 5 pages render, WebSocket streaming works.

## Bug Fix History

### Round 1 — Simulation blackscreen + vehicle jitter
- [x] Added ErrorBoundary to Simulation page
- [x] Frame interpolation for smooth 60fps vehicle animation

### Round 2 — Compare blackscreen + TLS never red
- [x] Added ErrorBoundary to Compare page
- [x] Split semicircle TLS rendering (first half / second half of state string)
- [x] Colour legend in Simulation sidebar and Compare page
- [x] Frame freeze on done (SimCanvas keeps final frame)

### Round 3 — Vehicles not stopping at red lights
- [x] Phase duration: step // 30 → step // 80
- [x] Detection range: 20 → 45 units
- [x] Hard stop near intersection (< 8 units → speed = 0)
- [x] Yellow treated as stop (not go)
- [x] Vehicle queuing behind stopped vehicles
- [x] Vehicle count increased to grid_n² × 5

### Round 4 — Mid-road stops + slow green clearance
- [x] `_is_near_intersection` helper added
- [x] Queuing only triggers behind vehicles near intersections
- [x] Wrap-around respawn resets speed to 8.0–13.89
- [x] Acceleration increased from 0.5 to 1.5

### Training chart fix
- [x] Added `isAnimationActive={false}` to live training Line component

## Current State

All known bugs are fixed. Dashboard is functional:
- Simulation page: vehicles stop at red lights, queue near intersections, clear on green
- Compare page: two simultaneous streams with diff panel
- Training page: live chart updates in real-time
- Communication page: charts and table render with mock data
