# Phase 6: Frontend Enhancements — Research

**Researched:** 2026-03-23
**Domain:** React / TypeScript frontend wiring — Recharts, React Query, WebSocket hooks, new page creation
**Confidence:** HIGH (all findings verified against live source code)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| UI-01 | Communication page wired to real evaluation comm cost data | GET /api/evaluation/{job_id} returns `comm_costs` per trial; page currently uses hardcoded arrays |
| UI-02 | Training page shows per-client reward breakdown | WS /ws/train/{job_id} frame has `mean_reward`; per-client breakdown not in current frame — requires backend addition or post-hoc derivation from evaluation results |
| UI-03 | Training history: list past runs, reload/compare | GET /api/evaluations returns summary list; training job results survive in-memory only (no disk persistence for training — only evaluation has disk store) |
| UI-04 | New Evaluation page with controls, results table, grouped bar charts | POST /api/evaluate + GET /api/evaluation/{job_id} + WS /ws/evaluate/{job_id} all exist and ready |
| UI-05 | Transfer matrix heatmap on Evaluation page | GET /api/evaluation/{job_id} includes `transfer` key when `include_transfer=true` was requested |
| UI-06 | Enhanced Compare page with trainer dropdown cycling | Compare.tsx already has trainer dropdowns; enhancement = add "cycle through trainers" button |
| UI-07 | Index page fetches real stats from evaluation API | GET /api/evaluations + GET /api/evaluation/{job_id} provide all four displayed stat values |
| UI-08 | Emergency vehicles rendered in distinct color (blue) | SimCanvas.tsx vehicle color logic: `white=moving, red=halted`; emergency flag not yet in SimFrame |
</phase_requirements>

---

## Summary

Phase 6 is a pure-frontend phase. The backend APIs it depends on (Phase 3 evaluation endpoints, Phase 2 training endpoints) are all implemented. All wiring work lives in `FrontEnd/src/` exclusively — no `BackEnd/` edits required for UI-01 through UI-07. UI-08 requires a small SimCanvas extension plus a backend SimFrame field addition.

The project uses React 18 + TypeScript + Recharts 2.x + TanStack React Query 5 + react-router-dom 6 + shadcn/ui components. These are already installed. No new npm packages are needed unless a heatmap library is required for the transfer matrix (Recharts can render one with a custom cell approach, or a lightweight dedicated library).

The key constraint for the planner is: **the existing pages are hardcoded with mock data and must be replaced with real API fetches**. Patterns for how to do this already exist in `Compare.tsx` (uses `useQuery`) and `Training.tsx` (uses `useTrainStream` hook + `startTraining` API call). New pages follow these same patterns.

**Primary recommendation:** Wire each page directly to its specific API endpoint using `useQuery` for REST and a new `useEvalStream` hook for the evaluation WebSocket. Build the Evaluation page as a new route before touching the existing pages to reduce regression risk.

---

## Standard Stack

### Core (all already installed)

| Library | Version (installed) | Purpose | Why Standard |
|---------|---------------------|---------|--------------|
| recharts | ^2.15.4 | All charts: bar, line, scatter, custom cell heatmap | Already in use across all pages; `isAnimationActive={false}` pattern established |
| @tanstack/react-query | ^5.83.0 | REST data fetching with cache, loading/error states | Already powering Compare page's `getNetworks`, `getWeights`, `getNetwork` calls |
| react-router-dom | ^6.30.1 | Page routing | Already configured in App.tsx |
| lucide-react | ^0.462.0 | Icons | Already in use |
| shadcn/ui (radix) | various | Cards, Tables, Select, Slider, Badge, Button | Full set already installed |

### No New Packages Required

The transfer matrix heatmap (UI-05) can be built with Recharts `<Cell>` + `<BarChart>` (grouped) or a custom SVG grid. Both approaches use zero new dependencies. Recommend the custom SVG grid — simpler and lighter for a 3x3 matrix.

**Installation:** No new `npm install` needed.

---

## Architecture Patterns

### Recommended Project Structure (additions only)

```
FrontEnd/src/
├── pages/
│   ├── Evaluation.tsx          # NEW — UI-04, UI-05 (new page)
│   ├── Communication.tsx       # EDIT — replace hardcoded data with useQuery
│   ├── Training.tsx            # EDIT — add history panel (UI-02, UI-03)
│   ├── Index.tsx               # EDIT — fetch real stats (UI-07)
│   └── Compare.tsx             # EDIT — trainer cycling (UI-06)
├── hooks/
│   └── useEvalStream.ts        # NEW — mirrors useTrainStream for /ws/evaluate/{job_id}
└── lib/
    └── api.ts                  # EDIT — add evaluation API functions
```

### Pattern 1: REST Data Fetch with React Query (established pattern)

Already used in Compare.tsx. Apply to Communication, Index, and Training history.

```typescript
// Source: FrontEnd/src/pages/Compare.tsx (lines 24-30)
const { data: networksData, isLoading, error } = useQuery({
  queryKey: ["evaluations"],
  queryFn: getEvaluations,   // GET /api/evaluations
  staleTime: 30_000,         // evaluations change infrequently
});
```

### Pattern 2: WebSocket Hook (established pattern)

`useTrainStream` is the model. `useEvalStream` follows identical structure but handles `progress` and `complete` frame types instead of episode frames.

```typescript
// Source: FrontEnd/src/hooks/useTrainStream.ts (full file)
// New hook: useEvalStream.ts — same skeleton, different message shape:
// { type: "progress", completed: int, total: int, trainer: str, topology: str }
// { type: "complete", job_id: str, status: str }
export function useEvalStream(jobId: string | null) {
  // Same pattern: useState, useRef<WebSocket>, useCallback connect, useEffect cleanup
  // Returns: { frames, progress, isConnected, isDone }
}
```

### Pattern 3: Recharts isAnimationActive={false} (mandatory rule)

All `<Line>` components with streaming or frequently-updated data MUST have `isAnimationActive={false}`. Already enforced in Training.tsx live chart. Apply to any new charts that update dynamically.

### Pattern 4: ErrorBoundary Wrapping

`ErrorBoundary` wraps Simulation and Compare pages. Wrap the new Evaluation page too since it launches async jobs.

### Pattern 5: Transfer Matrix Heatmap — Custom SVG Grid

A 3x3 grid (train_topo × test_topo) is small enough for a custom SVG approach — no library needed.

```typescript
// Data shape from GET /api/evaluation/{job_id}:
// results.transfer["FedRL"] = [
//   { train_topology, test_topology, metrics: { mean_reward: {...} } }
// ]
// Render as <svg> with <rect> colored by mean_reward value (red=low, green=high)
```

### Pattern 6: New Route Registration

Add Evaluation route in App.tsx:

```typescript
// Source: FrontEnd/src/App.tsx (lines 22-29)
import Evaluation from "./pages/Evaluation";
// Inside <Routes>:
<Route path="/evaluation" element={<Evaluation />} />
```

Also add nav item in AppSidebar.tsx:

```typescript
// Source: FrontEnd/src/components/AppSidebar.tsx (lines 15-21)
// Add: { title: "Evaluation", url: "/evaluation", icon: ClipboardList }
```

### Anti-Patterns to Avoid

- **Second rAF loop on SimCanvas:** SimCanvas uses a single rAF loop. For UI-08 (emergency vehicles in blue), only modify the vehicle color selection inside the existing draw loop — do not add any new `requestAnimationFrame` call.
- **Animation on streaming charts:** Never add `isAnimationActive={true}` (or omit the prop) on Line components receiving live data — the animation will restart on every data point.
- **Removing ErrorBoundary:** The Compare page ErrorBoundary must stay. Add one to the new Evaluation page.
- **Re-copying from LovableOutput/:** CLAUDE.md explicitly forbids re-copying from `LovableOutput/`. Edit `FrontEnd/` directly.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Data fetching + cache + loading state | Custom fetch wrapper | `useQuery` from @tanstack/react-query | Already in project; handles stale-while-revalidate, retry, error state |
| UI components (cards, tables, selects) | Custom React components | shadcn/ui + Radix (already installed) | Full set already present; consistent styling |
| Chart rendering | Custom SVG charts | Recharts (already in project) | Recharts handles scales, tooltips, legends |
| WebSocket lifecycle | Ad-hoc useEffect | Mirror `useTrainStream.ts` pattern into `useEvalStream.ts` | The pattern is already proven and handles cleanup correctly |

**Key insight:** Everything needed is already installed. The risk in this phase is regressions (breaking existing pages), not missing libraries.

---

## API Contract (Verified from BackEnd source)

### Endpoints used by Phase 6 UI

| Endpoint | Method | Used by | Response shape |
|----------|--------|---------|----------------|
| `/api/evaluate` | POST | Evaluation page | `{ job_id, status }` (202) |
| `/api/evaluation/{job_id}` | GET | Evaluation page, Index stats | See schema below |
| `/api/evaluations` | GET | Training history, Index | `[{ job_id, status, created_at, trainers, topologies, n_runs }]` |
| `/ws/evaluate/{job_id}` | WS | Evaluation progress | `progress` and `complete` frames |
| `/api/results/{job_id}` | GET | Training results (post-train) | `{ job_id, status, type, rewards[], trip_metrics, comm_costs }` |
| `/api/train` | POST | Training page (already wired) | `{ job_id, status }` (202) |
| `/ws/train/{job_id}` | WS | Training page (already wired) | `{ episode, mean_reward, trainer, fed_round }` |

### GET /api/evaluation/{job_id} Response Schema

```json
{
  "job_id": "eval_abc12345",
  "status": "complete",
  "created_at": "2026-03-23T...",
  "trainers": ["FedRL", "MARL", "SARL", "fixed-time", "max-pressure"],
  "topologies": ["grid-3x3", "grid-5x5", "grid-7x7"],
  "n_runs": 10,
  "campaign": [
    {
      "trainer": "FedRL",
      "topology": "grid-3x3",
      "n_completed": 10,
      "n_failed": 0,
      "aggregated": {
        "avg_waiting_time": { "mean": 27.1, "std": 2.3, "ci_95_lower": 25.7, "ci_95_upper": 28.5, "min": 22.0, "max": 31.0 },
        "avg_travel_time":  { "mean": 99.2, "std": 5.1, ... },
        "throughput":       { "mean": 0.94, ... },
        "mean_reward":      { "mean": -0.77, ... },
        "total_comm_cost":  { "mean": 2760, ... }
      },
      "individual_results": [
        { "trainer", "topology", "seed", "mean_reward", "total_reward", "comm_costs": { "EDGE2TLS_POLICY", "TLS2EDGE_OBS", "EDGE2TLS_RANK", "EDGE2TLS_ACTION", "VEH2TLS" }, "total_comm_cost", "tripinfo": { "avg_waiting_time", "avg_travel_time", "throughput", ... } }
      ]
    }
  ],
  "transfer": {
    "FedRL": [
      { "train_topology": "grid-3x3", "test_topology": "grid-3x3", "trainer": "FedRL", "metrics": { ... }, "is_transfer": false }
    ]
  }
}
```

### WebSocket Evaluate Frames

```json
// Progress frame (sent after each MC trial):
{ "type": "progress", "completed": 5, "total": 150, "trainer": "FedRL", "topology": "grid-3x3" }

// Completion frame:
{ "type": "complete", "job_id": "eval_abc12345", "status": "complete" }
```

### POST /api/evaluate Request Body

```json
{
  "trainers": ["FedRL", "MARL", "SARL", "fixed-time", "max-pressure"],
  "topologies": ["grid-3x3", "grid-5x5", "grid-7x7"],
  "n_runs": 10,
  "ranked": true,
  "horizon": 450,
  "include_transfer": false
}
```

---

## Per-Requirement Findings

### UI-01: Communication Page — Real Comm Cost Data

**Current state:** `Communication.tsx` has three hardcoded constant arrays (`barData`, `scatterData`, `rawData`) with a `// TODO: replace with fetch('/api/...')` comment.

**Real data source:** The comm cost breakdown by type (`EDGE2TLS_POLICY`, `TLS2EDGE_OBS`, `EDGE2TLS_RANK`, `EDGE2TLS_ACTION`, `VEH2TLS`) lives in `individual_results[].comm_costs` within a completed evaluation. Aggregated mean is in `aggregated.total_comm_cost`.

**Wiring approach:**
1. Call `GET /api/evaluations` to get list of completed evaluations.
2. User picks one (or page auto-selects latest).
3. Call `GET /api/evaluation/{job_id}`.
4. Derive `barData` from `campaign` entries — one row per trainer, five message-type columns from `individual_results[0].comm_costs` (or mean across all runs).
5. Derive `scatterData` from `aggregated.mean_reward.mean` vs `aggregated.total_comm_cost.mean`.

**Note:** The current page hardcodes `FedRL-pos` as a distinct trainer. In the evaluation API, trainer names are `"FedRL"`, `"MARL"`, `"SARL"`, `"fixed-time"`, `"max-pressure"`. The display names may need a mapping.

### UI-02: Training Page — Per-Client Reward Breakdown

**Current state:** The training WS stream (`useTrainStream`) returns only `mean_reward` — the mean across all agents/TLS clients. Per-client rewards are not in the current frame payload.

**Finding:** The `run_training_loop()` in `training_runner.py` extracts `episode_reward_mean` from Ray's result dict. Ray 2.x stores per-policy rewards in `result["env_runners"]["policy_reward_mean"]` (a dict keyed by policy id). This field is available but not currently forwarded.

**Options:**
1. (Preferred) Add `per_policy_rewards` to the episode_data dict in `training_runner.py` and forward it through the WS frame — small backend edit.
2. Alternatively, display per-client breakdown from an evaluation result (post-hoc, not live) — no backend change needed.

**Recommendation:** Option 1 — add `per_policy_rewards: dict[str, float]` to `episode_data` in `run_training_loop()`. Small backend change, enables live streaming.

### UI-03: Training History — List Past Runs

**Current state:** Training results survive only in-memory (no disk persistence for training jobs). Evaluation results DO persist to disk via `store.py`.

**Finding:** `GET /api/evaluations` returns all completed evaluations with metadata. This can serve as "history" if the user runs evaluations. For pure training runs (POST /api/train), there is no persistence — the job disappears on server restart.

**Approach for UI-03:** Show evaluation history (from `/api/evaluations`) as the "training history" table. This is semantically reasonable — evaluations include the trainer type and results. Add a "Load results" action that fetches the full evaluation and populates the Communication/Evaluation pages.

**Alternative:** Add disk persistence to training jobs in BackEnd (write to `results/training/` similar to `results/evaluations/`). This is a backend task and may be out of scope for a "frontend" phase. The safer approach is to use the existing `/api/evaluations` list.

### UI-04: New Evaluation Page

**Current state:** No Evaluation page exists. App.tsx has no `/evaluation` route.

**What to build:**
- Controls: trainer checkboxes, topology checkboxes, n_runs slider, ranked toggle, include_transfer toggle.
- Launch button: calls POST /api/evaluate, gets job_id.
- Progress bar: connects to WS /ws/evaluate/{job_id}, shows `completed/total`.
- Results table: after completion, fetches GET /api/evaluation/{job_id}, displays `campaign[]` as rows (trainer + topology + mean ± std for each metric).
- Grouped bar chart: `avg_waiting_time` grouped by trainer, one bar per topology (standard evaluation presentation).

### UI-05: Transfer Matrix Heatmap

**Data shape:** `results.transfer[trainerName]` is an array of objects with `train_topology`, `test_topology`, `is_transfer`, and `metrics.mean_reward.mean`.

**Rendering:** A 3×3 grid (train_topo rows, test_topo cols). Cell color encodes mean_reward (lower = worse = redder). Diagonal cells (is_transfer=false) are the baseline. Off-diagonal cells show transfer performance. Custom SVG `<rect>` approach is sufficient — no new library needed.

**Color scale:** Interpolate between HSL red (bad) and teal (good) based on normalized reward value. Use the existing `seal-teal` color token.

### UI-06: Compare Page — Trainer Cycling

**Current state:** Compare.tsx has two policy Select dropdowns (Policy A, Policy B). Enhancement: add a "Cycle →" button next to each dropdown that steps to the next policy in the `policies` array.

**Implementation:** One line of state logic — `setPolicyA(policies[(policies.indexOf(policyA) + 1) % policies.length])`.

### UI-07: Index Page — Real Stats

**Current state:** `Index.tsx` has four hardcoded stat values: comm reduction 36.24%, reward vs MARL -2.11%, travel time improvement 18.14%, active intersections 49.

**Real data source:** From a completed evaluation's `campaign` array:
- Comm reduction: `(1 - FedRL.total_comm_cost.mean / MARL.total_comm_cost.mean) * 100`
- Reward gap: `(FedRL.mean_reward.mean - MARL.mean_reward.mean) / abs(MARL.mean_reward.mean) * 100`
- Travel time improvement vs fixed-time: `(1 - FedRL.avg_travel_time.mean / fixed-time.avg_travel_time.mean) * 100`
- Active intersections: from network topology (9 for grid-3x3, or fetched from `/api/network/{topology}` node count)

**Approach:** `useQuery` for `/api/evaluations`, pick latest complete, `useQuery` for full results, derive stats. Show skeleton or last-known hardcoded values while loading.

### UI-08: Emergency Vehicles in Blue

**Current state:** SimCanvas.tsx vehicle color logic: white=moving (`#e2e8f0`), red=halted (`#ef4444`). The SimFrame type (in `useSimStream.ts`) has `vehicles` array — need to check if an `emergency` flag exists.

**Finding:** Emergency vehicle preemption is ADV-03 (Phase 5 — not yet implemented). The SimFrame vehicle objects currently have position, speed, halted fields only. A `type: "emergency"` or `is_emergency: boolean` field would need to be added by Phase 5 backend work.

**For Phase 6 (UI-08):** The SimCanvas color logic should be extended to check for an `is_emergency` (or `type === "emergency"`) field and render those vehicles blue (`#3b82f6`). Even if Phase 5 hasn't shipped yet, the canvas guard is `if (vehicle.is_emergency) color = "#3b82f6"` — it safely falls back when the field is absent. This makes UI-08 a frontend-only no-op until Phase 5 adds the backend field.

---

## Common Pitfalls

### Pitfall 1: Recharts Animation on Streaming Data

**What goes wrong:** Adding a `<Line>` without `isAnimationActive={false}` causes the chart to re-animate from scratch on every data point append, creating a jitter-heavy, unusable chart.
**Why it happens:** Recharts detects data array reference changes and triggers the enter animation.
**How to avoid:** Every `<Line>` that receives live/streaming data must have `isAnimationActive={false}`. CLAUDE.md explicitly lists this as a known pattern.
**Warning signs:** Chart "stutters" or "resets" visually as new data arrives.

### Pitfall 2: useQuery Cache Staleness on Evaluation Completion

**What goes wrong:** The Evaluation page launches a job, the WebSocket says "complete", but `GET /api/evaluation/{job_id}` returns stale/missing data because React Query cached an earlier (404) response.
**Why it happens:** React Query caches by queryKey. If the page pre-fetched the job_id before the evaluation completed, it may serve the cached 404.
**How to avoid:** Use `queryClient.invalidateQueries({ queryKey: ["evaluation", jobId] })` inside the `isDone` handler of `useEvalStream`, or set `enabled: isDone` so the query only fires after WS completion.

### Pitfall 3: Trainer Name Mismatch Between API and Display

**What goes wrong:** The evaluation API uses trainer names `"FedRL"`, `"MARL"`, `"SARL"`, `"fixed-time"`, `"max-pressure"`. The frontend Training page and Communication page use `"FedRL-naive"`, `"FedRL-pos_reward"`, `"MARL"`, `"SARL"`. Naive string comparison fails.
**How to avoid:** Create a `TRAINER_DISPLAY_NAMES` map in `api.ts` and apply it consistently. Resolve early in 06-01.

### Pitfall 4: Missing Evaluation Route Causes 404

**What goes wrong:** Evaluation.tsx is created but App.tsx is not updated → clicking the sidebar link hits the NotFound page.
**How to avoid:** Update App.tsx and AppSidebar.tsx in the same task as creating Evaluation.tsx.

### Pitfall 5: Breaking SimCanvas with a Second rAF Loop

**What goes wrong:** Adding emergency vehicle rendering by creating a second `requestAnimationFrame` loop alongside the existing one causes double-drawing artifacts and frame rate issues.
**How to avoid:** CLAUDE.md explicitly warns: "SimCanvas uses a single rAF loop with frame interpolation — do not add a second draw loop." For UI-08, only modify the vehicle color selection logic inside the existing loop in SimCanvas.tsx.

### Pitfall 6: Evaluation Page Polling Before Job Exists

**What goes wrong:** If the evaluation WS hook fires before the job is created in-memory on the backend (race condition), the WS endpoint returns close code 1008 "Job not found".
**Why it happens:** POST /api/evaluate returns 202 and the job is created synchronously, so this should not occur. But if the frontend connects the WS before the POST response is received, it can fail.
**How to avoid:** Only call `setJobId` (and thus open the WebSocket) after the POST response resolves. This is the same pattern used in Training.tsx `handleStartTraining`.

---

## Code Examples

### Adding Evaluation API Functions to api.ts

```typescript
// Source: FrontEnd/src/lib/api.ts (pattern from existing functions)
export interface EvaluationSummary {
  job_id: string;
  status: string;
  created_at: string;
  trainers?: string[];
  topologies?: string[];
  n_runs?: number;
}

export interface EvaluationResult {
  job_id: string;
  status: string;
  campaign: Array<{
    trainer: string;
    topology: string;
    n_completed: number;
    n_failed: number;
    aggregated: {
      avg_waiting_time: StatBlock;
      avg_travel_time: StatBlock;
      throughput: StatBlock;
      mean_reward: StatBlock;
      total_comm_cost: StatBlock;
    };
    individual_results: TrialResult[];
  }>;
  transfer?: Record<string, TransferEntry[]>;
}

export interface StatBlock {
  mean: number; std: number; ci_95_lower: number; ci_95_upper: number; min: number; max: number;
}

export async function getEvaluations(): Promise<EvaluationSummary[]> {
  const res = await fetch(`${BASE_URL}/api/evaluations`);
  if (!res.ok) throw new Error("Failed to fetch evaluations");
  return res.json();
}

export async function getEvaluation(job_id: string): Promise<EvaluationResult> {
  const res = await fetch(`${BASE_URL}/api/evaluation/${job_id}`);
  if (!res.ok) throw new Error("Failed to fetch evaluation");
  return res.json();
}

export async function startEvaluation(params: {
  trainers?: string[]; topologies?: string[];
  n_runs: number; ranked: boolean; horizon: number; include_transfer: boolean;
}): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${BASE_URL}/api/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Failed to start evaluation");
  return res.json();
}
```

### useEvalStream Hook

```typescript
// Source: mirrors FrontEnd/src/hooks/useTrainStream.ts
export interface EvalProgressFrame {
  type: "progress";
  completed: number;
  total: number;
  trainer: string;
  topology: string;
}

export function useEvalStream(jobId: string | null) {
  const [frames, setFrames] = useState<EvalProgressFrame[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (!jobId) return;
    const ws = new WebSocket(`ws://localhost:8000/ws/evaluate/${jobId}`);
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const frame = JSON.parse(event.data);
      if (frame.type === "progress") setFrames(prev => [...prev, frame]);
      if (frame.type === "complete") setIsDone(true);
    };
    ws.onclose = () => { setIsConnected(false); setIsDone(true); };
    ws.onerror = (e) => console.error("EvalStream WS error", e);
  }, [jobId]);

  useEffect(() => {
    if (jobId) connect();
    return () => wsRef.current?.close();
  }, [jobId, connect]);

  return { frames, isConnected, isDone };
}
```

### Transfer Matrix Heatmap (custom SVG)

```typescript
// 3x3 matrix where rows = train_topo, cols = test_topo
// Color: normalize mean_reward across all cells -> map to hsl
const TransferHeatmap = ({ data }: { data: TransferEntry[] }) => {
  const topos = ["grid-3x3", "grid-5x5", "grid-7x7"];
  const byKey: Record<string, number> = {};
  data.forEach(e => { byKey[`${e.train_topology}|${e.test_topology}`] = e.metrics.mean_reward.mean; });
  const values = Object.values(byKey);
  const minV = Math.min(...values), maxV = Math.max(...values);
  const norm = (v: number) => (v - minV) / (maxV - minV + 1e-9);
  const CELL = 80, PAD = 60;
  const W = topos.length * CELL + PAD, H = topos.length * CELL + PAD;
  return (
    <svg width={W} height={H}>
      {topos.map((train, r) => topos.map((test, c) => {
        const v = byKey[`${train}|${test}`] ?? 0;
        const n = norm(v);
        const fill = `hsl(${Math.round(n * 160)}, 70%, 45%)`; // red->green
        return (
          <g key={`${r}-${c}`}>
            <rect x={PAD + c * CELL} y={PAD + r * CELL} width={CELL - 2} height={CELL - 2} fill={fill} rx={4} />
            <text x={PAD + c * CELL + CELL/2} y={PAD + r * CELL + CELL/2 + 5} textAnchor="middle" fontSize={11} fill="#fff">
              {v.toFixed(2)}
            </text>
          </g>
        );
      }))}
    </svg>
  );
};
```

---

## State of the Art

| Old Approach | Current Approach | Status | Impact |
|--------------|------------------|--------|--------|
| Hardcoded mock arrays in every page | Real API fetch via useQuery + WS hooks | Phase 6 delivers this | Pages show research-grade data |
| Single mock training curve | Live stream from /ws/train + history from /api/evaluations | Already partially done for live; history is new | Training page becomes a real research tool |
| No Evaluation page | Full evaluation control + results table + heatmap | New in Phase 6 | Core research workflow exposed |

**Nothing deprecated to migrate.** The mock data constants can be removed entirely once real fetches are in place.

---

## Open Questions

1. **UI-02 per-client rewards: frontend-only or backend addition needed?**
   - What we know: The training WS frame currently has only `mean_reward` (aggregate). Ray 2.x stores `policy_reward_mean` per-policy in the training result dict.
   - What's unclear: Whether per-policy rewards are accessible in the current `run_training_loop()` result dict without additional extraction.
   - Recommendation: Add `per_policy_rewards` extraction to `training_runner.py` (one dict lookup). This is a small backend change justified by the UI requirement.

2. **UI-03 training history: use evaluation history or add training persistence?**
   - What we know: Only evaluations are persisted to disk. Training jobs are in-memory only.
   - Recommendation: Use `/api/evaluations` as the history source for Phase 6. Label it "Evaluation History" rather than "Training History" to be accurate. Avoid scope creep into backend persistence changes.

3. **UI-08 emergency vehicle color: Phase 5 dependency?**
   - What we know: Phase 5 (ADV-03) adds emergency vehicle preemption. The `is_emergency` field on SimFrame vehicles does not exist yet.
   - Recommendation: Add the SimCanvas color guard now (if `vehicle.is_emergency` → blue). It silently has no effect until Phase 5 populates the field. This unblocks UI-08 as a frontend-complete task.

---

## Validation Architecture

Framework: Vitest (^3.2.4) — already configured in `package.json` with `"test": "vitest run"`.

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UI-01 | Communication page renders real data (not hardcoded) | unit (mock fetch) | `npx vitest run src/pages/Communication.test.tsx` | No — Wave 0 gap |
| UI-02 | Training page shows per-client rewards when available | unit (mock WS) | `npx vitest run src/hooks/useEvalStream.test.ts` | No — Wave 0 gap |
| UI-03 | Training history list renders evaluations from API | unit (mock fetch) | `npx vitest run src/pages/Training.test.tsx` | No — Wave 0 gap |
| UI-04 | Evaluation page renders controls and launches job | unit (mock fetch) | `npx vitest run src/pages/Evaluation.test.tsx` | No — Wave 0 gap |
| UI-05 | Transfer heatmap renders correct cell count | unit (component) | `npx vitest run src/pages/Evaluation.test.tsx` | No — Wave 0 gap |
| UI-06 | Compare cycle button advances trainer selection | unit (component) | `npx vitest run src/pages/Compare.test.tsx` | No — Wave 0 gap |
| UI-07 | Index stats derived from real evaluation data | unit (mock fetch) | `npx vitest run src/pages/Index.test.tsx` | No — Wave 0 gap |
| UI-08 | SimCanvas renders emergency vehicles in blue | unit (canvas mock) | `npx vitest run src/components/SimCanvas.test.tsx` | No — Wave 0 gap |

Note: All UI tests are "manual-only" acceptable for Phase 6 since canvas and WebSocket behavior is hard to unit-test in jsdom. Smoke tests verifying renders-without-crash are sufficient for the automated suite.

### Test Runner

| Property | Value |
|----------|-------|
| Framework | Vitest 3.2.4 |
| Config file | vite.config.ts (implicit) or vitest.config.ts |
| Quick run | `cd FrontEnd && npm test` |
| Full suite | `cd FrontEnd && npm test` |
| Test env | jsdom (configured in devDependencies) |

### Sampling Rate

- Per task commit: `cd FrontEnd && npm test`
- Per wave merge: `cd FrontEnd && npm test`
- Phase gate: Full suite green + manual smoke of each page in browser

### Wave 0 Gaps

- [ ] `FrontEnd/src/hooks/useEvalStream.test.ts` — unit test for WS hook
- [ ] `FrontEnd/src/pages/Evaluation.test.tsx` — render + interaction smoke tests
- [ ] `FrontEnd/vitest.config.ts` — may be needed if jsdom environment not configured globally

---

## Sources

### Primary (HIGH confidence — live source code inspection)

- `FrontEnd/src/pages/Communication.tsx` — confirmed all data is hardcoded, has TODO comment
- `FrontEnd/src/pages/Training.tsx` — confirmed live WS wiring exists; summary table is mock
- `FrontEnd/src/pages/Index.tsx` — confirmed all stats are hardcoded constants
- `FrontEnd/src/pages/Compare.tsx` — confirmed useQuery pattern, trainer dropdown structure
- `FrontEnd/src/hooks/useTrainStream.ts` — confirmed WS hook pattern to replicate
- `FrontEnd/src/lib/api.ts` — confirmed API function signatures and base URL
- `FrontEnd/src/App.tsx` — confirmed route structure; no Evaluation route exists
- `FrontEnd/src/components/AppSidebar.tsx` — confirmed nav items; no Evaluation link
- `FrontEnd/src/components/SimCanvas.tsx` — confirmed single rAF loop; vehicle color logic
- `BackEnd/api/routes/evaluate.py` — confirmed all three endpoints (POST, GET, GET list)
- `BackEnd/api/ws/evaluate.py` — confirmed WS frame types and close behavior
- `BackEnd/api/evaluation/monte_carlo.py` — confirmed response schema (campaign[], aggregated{})
- `BackEnd/api/evaluation/metrics.py` — confirmed field names: avg_waiting_time, mean_reward, comm_costs keys
- `BackEnd/api/evaluation/transfer.py` — confirmed transfer matrix structure (train_topology, test_topology, metrics)
- `BackEnd/api/evaluation/store.py` — confirmed evaluations persisted to disk; training is NOT
- `BackEnd/api/routes/train.py` — confirmed training frames: episode, mean_reward, trainer, fed_round only
- `BackEnd/api/training_runner.py` — confirmed episode_data dict does not include per_policy_rewards
- `FrontEnd/package.json` — confirmed all needed libraries installed; no new installs needed

### Secondary (MEDIUM confidence)

- CLAUDE.md known patterns — `isAnimationActive={false}` rule, single rAF loop rule, ErrorBoundary rule — all verified against source code

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified in package.json
- Architecture patterns: HIGH — patterns extracted from live source code
- API contract: HIGH — verified from BackEnd route and WS source files
- Per-requirement findings: HIGH — all based on direct source inspection
- Pitfalls: HIGH — derived from CLAUDE.md warnings + code structure observation

**Research date:** 2026-03-23
**Valid until:** 2026-04-23 (stable dependencies; would only change if Phase 3/4/5 APIs are modified)
