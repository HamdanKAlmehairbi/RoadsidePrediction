import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PanelConfig, SimState, Topology } from "@/lib/types";
import { computeGridLayout, sumoTlsIdToLayoutId, type GridLayout } from "@/lib/mockData";

interface RawVehicle {
  id: string;
  x: number; // normalized [0,1]
  y: number;
  angle: number; // SUMO: 0=north, 90=east
}
interface RawSignal {
  id: string;
  state: "red" | "yellow" | "green";
}
interface RawMetrics {
  avg_wait: number;
  throughput: number;
  completed: number;
  queued: number;
}
interface RawFrame {
  t: number;
  step: number;
  vehicles: RawVehicle[];
  signals: RawSignal[];
  metrics: RawMetrics;
}
interface SessionMessage {
  type: "frame" | "error";
  sim_a?: RawFrame | null;
  sim_b?: RawFrame | null;
  message?: string;
}

const CAR_PALETTE = ["#60A5FA", "#F97316", "#22C55E", "#EAB308", "#A78BFA", "#F43F5E"];

function hashColor(id: string): string {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return CAR_PALETTE[Math.abs(h) % CAR_PALETTE.length];
}

function frameToSimState(
  frame: RawFrame | null | undefined,
  layout: GridLayout,
): SimState | null {
  if (!frame) return null;

  // Drop vehicles outside the junction bbox: fringe edges extend past the
  // TLS bounds we normalize against, so their nx/ny fall outside [0,1] and
  // would place the car off-canvas or on a road segment the frontend doesn't
  // draw. Cars pop into view at the first real intersection instead of
  // appearing to float in empty space.
  const vehicles = frame.vehicles
    .filter((v) => v.x >= 0 && v.x <= 1 && v.y >= 0 && v.y <= 1)
    .map((v) => {
      // Snap SUMO angle to the nearest cardinal. SUMO's reported angle
      // jitters a fraction of a degree even for straight travel, and our
      // rect anchor depends on which 90° quadrant we're in — so we don't
      // want noise near 45/135/225/315 to flip orientation mid-block.
      const angle = ((v.angle % 360) + 360) % 360;
      const heading = Math.round(angle / 90) % 4; // 0=N, 1=E, 2=S, 3=W
      const orientation: "h" | "v" = heading % 2 === 1 ? "h" : "v";
      const flip = heading === 0 || heading === 3; // N or W → headlight at start edge
      return {
        id: v.id,
        x: layout.marginX + v.x * layout.usableW,
        y: layout.marginY + v.y * layout.usableH,
        orientation,
        flip,
        color: hashColor(v.id),
      };
    });

  const signals = frame.signals.map((s) => {
    const layoutId = sumoTlsIdToLayoutId(s.id, layout.n) ?? s.id;
    return { id: layoutId, x: 0, y: 0, state: s.state };
  });

  return {
    vehicles,
    signals,
    metrics: {
      avgWait: frame.metrics.avg_wait,
      throughput: frame.metrics.throughput,
      completed: frame.metrics.completed,
      queued: frame.metrics.queued,
    },
  };
}

export interface CompareSessionState {
  sessionId: string | null;
  /** WS handshake accepted (still open). */
  connected: boolean;
  /** True from POST /compare/start success until stop() / reset. Tells the UI
   *  "treat this as a live run, don't fall back to preview even while the WS
   *  is (re)connecting or has closed after the sim finished." */
  active: boolean;
  error: string | null;
  stateA: SimState | null;
  stateB: SimState | null;
  start: (configA: PanelConfig, configB: PanelConfig) => Promise<void>;
  stop: () => Promise<void>;
}

/** In dev, hit the backend directly instead of going through Vite's proxy —
 *  Vite's WebSocket proxy does not forward the /ws/compare upgrade reliably
 *  (observed: "did not receive a valid HTTP response"). In a prod build we
 *  assume same-origin. */
const BACKEND_DEV_HOST = "localhost:8000";

function apiBase(): string {
  if (import.meta.env.DEV) return `http://${BACKEND_DEV_HOST}`;
  return "";
}

function wsBase(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = import.meta.env.DEV ? BACKEND_DEV_HOST : window.location.host;
  return `${proto}//${host}`;
}

export function useCompareSession(
  topologyA: Topology,
  topologyB: Topology,
): CompareSessionState {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stateA, setStateA] = useState<SimState | null>(null);
  const [stateB, setStateB] = useState<SimState | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const layoutARef = useRef<GridLayout>(computeGridLayout(topologyA));
  const layoutBRef = useRef<GridLayout>(computeGridLayout(topologyB));
  // Keep refs in sync with the latest selected topologies so pre-run (and
  // between-run) conversions use the right grid. The session freezes the
  // layout at start() time anyway, but this avoids stale refs if the user
  // changes topology before hitting Run.
  useMemo(() => {
    layoutARef.current = computeGridLayout(topologyA);
  }, [topologyA]);
  useMemo(() => {
    layoutBRef.current = computeGridLayout(topologyB);
  }, [topologyB]);

  const stop = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (sid) {
      try {
        await fetch(`${apiBase()}/compare/stop/${sid}`, { method: "POST" });
      } catch {
        // ignore
      }
    }
    sessionIdRef.current = null;
    setSessionId(null);
    setConnected(false);
    setActive(false);
    setStateA(null);
    setStateB(null);
  }, []);

  const start = useCallback(
    async (configA: PanelConfig, configB: PanelConfig) => {
      setError(null);
      setActive(true);
      setStateA(null);
      setStateB(null);
      // Freeze the layout for each side at start time — the server's frames
      // are tied to these topologies, so the mapping must be too.
      layoutARef.current = computeGridLayout(configA.topology);
      layoutBRef.current = computeGridLayout(configB.topology);
      try {
        const res = await fetch(`${apiBase()}/compare/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            topology_a: configA.topology,
            strategy_a: configA.strategy,
            topology_b: configB.topology,
            strategy_b: configB.strategy,
            horizon: 3600,
            step_hz: 5.0,
          }),
        });
        if (!res.ok) {
          throw new Error(`start failed: ${res.status}`);
        }
        const data = (await res.json()) as { session_id: string; ws_url: string };
        sessionIdRef.current = data.session_id;
        setSessionId(data.session_id);

        const url = `${wsBase()}${data.ws_url}`;
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => setConnected(true);
        ws.onclose = () => setConnected(false);
        ws.onerror = () => setError("WebSocket error");
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data) as SessionMessage;
            if (msg.type === "error") {
              setError(msg.message || "unknown server error");
              return;
            }
            if (msg.sim_a) setStateA(frameToSimState(msg.sim_a, layoutARef.current));
            if (msg.sim_b) setStateB(frameToSimState(msg.sim_b, layoutBRef.current));
          } catch (e) {
            console.error("bad frame", e);
          }
        };
      } catch (e) {
        setError(e instanceof Error ? e.message : "unknown error");
        setActive(false);
      }
    },
    [],
  );

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return { sessionId, connected, active, error, stateA, stateB, start, stop };
}
