import type { SimState, Topology, Signal } from "./types";

const CAR_PALETTE = [
  "#60A5FA",
  "#F97316",
  "#22C55E",
  "#EAB308",
  "#A78BFA",
  "#F43F5E",
];

export interface GridLayout {
  n: number;
  intersections: { id: string; x: number; y: number }[];
  horizontalRoads: number[]; // y positions of road center
  verticalRoads: number[]; // x positions of road center
  canvasWidth: number;
  canvasHeight: number;
  marginX: number;
  marginY: number;
  usableW: number;
  usableH: number;
}

// Compute grid layout for any NxN grid within a bounded canvas.
export function computeGridLayout(topology: Topology): GridLayout {
  const n = topology === "grid-3x3" ? 3 : 5;
  const canvasWidth = 560;
  const canvasHeight = 460;
  const marginX = 60;
  const marginY = 55;
  const usableW = canvasWidth - marginX * 2;
  const usableH = canvasHeight - marginY * 2;
  const stepX = n > 1 ? usableW / (n - 1) : 0;
  const stepY = n > 1 ? usableH / (n - 1) : 0;

  const intersections: { id: string; x: number; y: number }[] = [];
  const horizontalRoads: number[] = [];
  const verticalRoads: number[] = [];

  for (let row = 0; row < n; row++) {
    horizontalRoads.push(marginY + row * stepY);
    for (let col = 0; col < n; col++) {
      if (row === 0) verticalRoads.push(marginX + col * stepX);
      intersections.push({
        id: `${row}-${col}`,
        x: marginX + col * stepX,
        y: marginY + row * stepY,
      });
    }
  }

  return {
    n,
    intersections,
    horizontalRoads,
    verticalRoads,
    canvasWidth,
    canvasHeight,
    marginX,
    marginY,
    usableW,
    usableH,
  };
}

/** Map a SUMO grid TLS id (e.g. "A0", "B2", "E4") to the frontend layout id
 *  ("row-col"). SUMO convention for the grid nets: letter = column
 *  (A=0, B=1, …), digit = row from bottom (0=bottom). Since the frontend
 *  flips Y (top-left origin), canvas row = (n-1) - digit. */
export function sumoTlsIdToLayoutId(sumoId: string, n: number): string | null {
  const m = /^([A-Z])(\d+)$/.exec(sumoId);
  if (!m) return null;
  const col = m[1].charCodeAt(0) - "A".charCodeAt(0);
  const digit = parseInt(m[2], 10);
  if (col < 0 || col >= n || digit < 0 || digit >= n) return null;
  const row = n - 1 - digit;
  return `${row}-${col}`;
}

// Idle-state preview: empty roads with steady signals. Previously this
// emitted ~20 random cars per panel that looked chaotic and caused a visible
// "clear" the moment live data replaced them. For a seamless experience we
// show no cars until the simulator streams real ones.
export function makeMockState(
  topology: Topology,
  scenario: "flowing" | "congested"
): SimState {
  const layout = computeGridLayout(topology);
  const signals: Signal[] = layout.intersections.map((intr) => ({
    id: intr.id,
    x: intr.x,
    y: intr.y,
    // Keep a calm default pattern so intersections are still visually marked.
    state: scenario === "flowing" ? "green" : "red",
  }));

  const metrics = scenario === "flowing"
    ? { avgWait: 0, throughput: 0, completed: 0, queued: 0 }
    : { avgWait: 0, throughput: 0, completed: 0, queued: 0 };

  // Suppress unused CAR_PALETTE warning by touching it (kept for potential reuse).
  void CAR_PALETTE;

  return { vehicles: [], signals, metrics };
}
