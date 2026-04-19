import { useMemo } from "react";
import type { SimState, SignalState, Topology, Vehicle } from "@/lib/types";
import { computeGridLayout } from "@/lib/mockData";

const SIGNAL_COLORS: Record<SignalState, { fill: string; glow: string }> = {
  red: { fill: "#FF4D5A", glow: "#FF4D5A" },
  yellow: { fill: "#FFC857", glow: "#FFC857" },
  green: { fill: "#38D97A", glow: "#38D97A" },
};

const ROAD_W = 18;
const INT_SIZE = 24;
const SIGNAL_W = 8;
const SIGNAL_H = 16;
const LAMP_R = 2;
const LAMP_INACTIVE = "#334155";

interface NetworkCanvasProps {
  topology: Topology;
  state: SimState;
}

export function NetworkCanvas({ topology, state }: NetworkCanvasProps) {
  const layout = useMemo(() => computeGridLayout(topology), [topology]);
  const { canvasWidth, canvasHeight, intersections, horizontalRoads, verticalRoads } = layout;

  return (
    <svg
      width={canvasWidth}
      height={canvasHeight}
      viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
      className="rounded-[8px] border border-[var(--border)] bg-[var(--bg)]"
      style={{ shapeRendering: "geometricPrecision" }}
    >
      {/* Road surfaces */}
      {horizontalRoads.map((y) => (
        <rect
          key={`rh-${y}`}
          x={20}
          y={y - ROAD_W / 2}
          width={canvasWidth - 40}
          height={ROAD_W}
          fill="#1E293B"
        />
      ))}
      {verticalRoads.map((x) => (
        <rect
          key={`rv-${x}`}
          x={x - ROAD_W / 2}
          y={20}
          width={ROAD_W}
          height={canvasHeight - 40}
          fill="#1E293B"
        />
      ))}

      {/* Dashed center lines on horizontal roads */}
      {horizontalRoads.map((y) => (
        <line
          key={`dh-${y}`}
          x1={20}
          y1={y}
          x2={canvasWidth - 20}
          y2={y}
          stroke="#334155"
          strokeOpacity={0.5}
          strokeWidth={1.5}
          strokeDasharray="10 6"
        />
      ))}

      {/* Dashed center lines on vertical roads */}
      {verticalRoads.map((x) => (
        <line
          key={`dv-${x}`}
          x1={x}
          y1={20}
          x2={x}
          y2={canvasHeight - 20}
          stroke="#334155"
          strokeOpacity={0.5}
          strokeWidth={1.5}
          strokeDasharray="10 6"
        />
      ))}

      {/* Intersection boxes (rendered on top of roads) */}
      {intersections.map((intr) => (
        <rect
          key={`i-${intr.id}`}
          x={intr.x - INT_SIZE / 2}
          y={intr.y - INT_SIZE / 2}
          width={INT_SIZE}
          height={INT_SIZE}
          fill="#0F172A"
          stroke="#334155"
          strokeWidth={1}
          rx={2}
        />
      ))}

      {/* Vehicles. Key includes orientation+flip so a heading change (car
          turning at a junction) remounts the shape rather than sliding the
          rect anchor across lanes via the 220ms CSS transition. */}
      {state.vehicles.map((v) => (
        <VehicleShape
          key={`${v.id}:${v.orientation}:${v.flip ? 1 : 0}`}
          vehicle={v}
        />
      ))}

      {/* Traffic signals (rendered on top, at NE corner of each intersection) */}
      {state.signals.map((sig) => {
        const intr = intersections.find((i) => i.id === sig.id);
        if (!intr) return null;
        const sx = intr.x + INT_SIZE / 2 + 2;
        const sy = intr.y - INT_SIZE / 2 - 13;
        return <TrafficSignal key={`s-${sig.id}`} x={sx} y={sy} state={sig.state} />;
      })}
    </svg>
  );
}

function VehicleShape({ vehicle }: { vehicle: Vehicle }) {
  const { x, y, orientation, flip, color } = vehicle;
  const width = orientation === "h" ? 14 : 7;
  // Horizontal body capped at 6 px so opposite-direction cars don't bleed
  // across the centerline: lane-offset at our y-scale is only 2.8 px and the
  // two lane centers are 5.6 px apart. Vertical body stays 14 since that's
  // the travel direction, not the lateral one.
  const height = orientation === "h" ? 6 : 14;

  // Headlight position (front of car).
  // Horizontal car: headlight at right end by default, flip moves it to left.
  // Vertical car: headlight at bottom end by default, flip moves it to top.
  let hlX: number;
  let hlY: number;
  let hlW: number;
  let hlH: number;
  if (orientation === "h") {
    hlW = 2;
    hlH = 4; // body is 6 tall now; keep 1px margin top and bottom
    hlX = flip ? 1 : 11;
    hlY = 1;
  } else {
    hlW = 5;
    hlH = 2;
    hlX = 1;
    hlY = flip ? 1 : 11;
  }

  // (x, y) is the front-bumper position SUMO reports. We anchor the rect
  // to that point so the body extends *backward* along the travel direction.
  // Centering would put half the body past the bumper (into the intersection
  // or the oncoming car) — that was causing the ~30% overlap at stop lines.
  // Laterally we still center, so the car sits on its lane stripe.
  let tx: number;
  let ty: number;
  if (orientation === "h") {
    ty = y - height / 2;
    // eastbound (flip=false): front at east end → rect east edge at x
    // westbound (flip=true):  front at west end → rect west edge at x
    tx = flip ? x : x - width;
  } else {
    tx = x - width / 2;
    // southbound (flip=false): front at south end (larger canvas y) → rect south edge at y
    // northbound (flip=true):  front at north end (smaller canvas y) → rect north edge at y
    ty = flip ? y : y - height;
  }

  // CSS transform + transition gives smooth glide between the sim's discrete
  // frames (coming at ~5 Hz). React keeps the <g> mounted because we key by
  // vehicle.id, so when `x`/`y` update the browser interpolates.
  return (
    <g
      style={{
        transform: `translate(${tx}px, ${ty}px)`,
        transition: "transform 220ms linear",
        willChange: "transform",
      }}
    >
      <rect width={width} height={height} rx={2} fill={color} />
      <rect x={hlX} y={hlY} width={hlW} height={hlH} rx={0.5} fill="#F8FAFC" />
    </g>
  );
}

function TrafficSignal({
  x,
  y,
  state,
}: {
  x: number;
  y: number;
  state: SignalState;
}) {
  const activeColor = SIGNAL_COLORS[state];
  const fills = {
    red: state === "red" ? activeColor.fill : LAMP_INACTIVE,
    yellow: state === "yellow" ? activeColor.fill : LAMP_INACTIVE,
    green: state === "green" ? activeColor.fill : LAMP_INACTIVE,
  };

  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Housing */}
      <rect
        width={SIGNAL_W}
        height={SIGNAL_H - 4}
        rx={1}
        fill="#0F172A"
        stroke="#1F2937"
        strokeWidth={1}
      />
      {/* Pole */}
      <rect x={3} y={SIGNAL_H - 4} width={2} height={4} fill="#334155" />
      {/* Lamps */}
      <circle
        cx={4}
        cy={3}
        r={LAMP_R}
        fill={fills.red}
        style={state === "red" ? { filter: `drop-shadow(0 0 4px ${activeColor.glow})` } : undefined}
      />
      <circle
        cx={4}
        cy={7}
        r={LAMP_R}
        fill={fills.yellow}
        style={state === "yellow" ? { filter: `drop-shadow(0 0 4px ${activeColor.glow})` } : undefined}
      />
      <circle
        cx={4}
        cy={11}
        r={LAMP_R}
        fill={fills.green}
        style={state === "green" ? { filter: `drop-shadow(0 0 4px ${activeColor.glow})` } : undefined}
      />
    </g>
  );
}
