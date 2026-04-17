import { useEffect, useRef } from "react";
import type { NetworkLayout } from "@/lib/api";
import type { SimFrame } from "@/hooks/useSimStream";

interface SimCanvasProps {
  layout: NetworkLayout | null;
  frame: SimFrame | null;
  className?: string;
}

function tlsColors(state: string): [string, string] {
  if (!state || state.length < 2) return ["#6b7280", "#6b7280"];
  const mid = Math.floor(state.length / 2);
  const firstHalf = state.slice(0, mid);
  const secondHalf = state.slice(mid);
  const dominantColor = (s: string) => {
    const lower = s.toLowerCase();
    if (lower.includes("g")) return "#22c55e";
    if (lower.includes("y")) return "#eab308";
    return "#ef4444";
  };
  return [dominantColor(firstHalf), dominantColor(secondHalf)];
}

function lerpAngle(a: number, b: number, t: number): number {
  let diff = ((b - a + 540) % 360) - 180;
  return a + diff * t;
}

const FRAME_INTERVAL = 100; // expected ms between backend frames (~10fps)

export function SimCanvas({ layout, frame, className }: SimCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Frame interpolation refs
  const prevFrameRef = useRef<SimFrame | null>(null);
  const currFrameRef = useRef<SimFrame | null>(null);
  const frameArrivalRef = useRef<number>(0);
  const layoutRef = useRef<NetworkLayout | null>(null);
  const rafIdRef = useRef<number>(0);

  // Keep layout ref in sync
  layoutRef.current = layout;

  // When frame prop changes, shift frames for interpolation
  // Once a done frame is received, freeze the canvas on that frame.
  useEffect(() => {
    if (!frame) return;
    if (currFrameRef.current?.done && frame.done) return;
    prevFrameRef.current = currFrameRef.current;
    currFrameRef.current = frame;
    frameArrivalRef.current = Date.now();
  }, [frame]);

  // Single rAF loop — the ONLY thing that draws
  useEffect(() => {
    function draw() {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) {
        rafIdRef.current = requestAnimationFrame(draw);
        return;
      }

      const currentLayout = layoutRef.current;

      // Match canvas resolution to its display size
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const W = Math.round(rect.width * dpr);
      const H = Math.round(rect.height * dpr);

      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W;
        canvas.height = H;
        canvas.style.width = rect.width + "px";
        canvas.style.height = rect.height + "px";
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        rafIdRef.current = requestAnimationFrame(draw);
        return;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const cw = rect.width;
      const ch = rect.height;

      // Clear
      ctx.clearRect(0, 0, cw, ch);
      ctx.fillStyle = "#0f1117";
      ctx.fillRect(0, 0, cw, ch);

      if (!currentLayout) {
        rafIdRef.current = requestAnimationFrame(draw);
        return;
      }

      const { min_x, min_y, max_x, max_y } = currentLayout.bounds;
      const worldW = max_x - min_x || 1;
      const worldH = max_y - min_y || 1;
      const padding = 40;
      const scaleX = (cw - padding * 2) / worldW;
      const scaleY = (ch - padding * 2) / worldH;
      const scale = Math.min(scaleX, scaleY);
      const offsetX = (cw - worldW * scale) / 2;
      const offsetY = (ch - worldH * scale) / 2;

      function toCanvas(x: number, y: number): [number, number] {
        return [
          (x - min_x) * scale + offsetX,
          ch - ((y - min_y) * scale + offsetY),
        ];
      }

      // Draw roads (lane shapes as thick lines)
      ctx.lineCap = "round";
      ctx.strokeStyle = "#2a2d3e";
      ctx.lineWidth = Math.max(10, scale * 5);
      for (const edge of currentLayout.edges) {
        for (const lane of edge.lanes) {
          if (lane.shape.length < 2) continue;
          ctx.beginPath();
          const [sx, sy] = toCanvas(lane.shape[0][0], lane.shape[0][1]);
          ctx.moveTo(sx, sy);
          for (let i = 1; i < lane.shape.length; i++) {
            const [lx, ly] = toCanvas(lane.shape[i][0], lane.shape[i][1]);
            ctx.lineTo(lx, ly);
          }
          ctx.stroke();
        }
      }

      // Draw lane center lines (subtle dashes)
      ctx.strokeStyle = "#3a3d4e";
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 8]);
      for (const edge of currentLayout.edges) {
        for (const lane of edge.lanes) {
          if (lane.shape.length < 2) continue;
          ctx.beginPath();
          const [sx, sy] = toCanvas(lane.shape[0][0], lane.shape[0][1]);
          ctx.moveTo(sx, sy);
          for (let i = 1; i < lane.shape.length; i++) {
            const [lx, ly] = toCanvas(lane.shape[i][0], lane.shape[i][1]);
            ctx.lineTo(lx, ly);
          }
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);

      // Use current frame for TLS (no interpolation needed for signals)
      const currFrame = currFrameRef.current;

      // Draw intersection circles (nodes) with split semicircles for TLS
      const nodeRadius = Math.max(10, scale * 8);
      for (const node of currentLayout.nodes) {
        const [cx, cy] = toCanvas(node.x, node.y);
        const tls = currFrame?.traffic_lights?.find((t) => t.id === node.id);
        const [color1, color2] = tls ? tlsColors(tls.state) : ["#6b7280", "#6b7280"];

        // Glow
        ctx.beginPath();
        ctx.arc(cx, cy, nodeRadius + 4, 0, Math.PI * 2);
        ctx.fillStyle = color1 + "40";
        ctx.fill();

        // Left semicircle (direction 1)
        ctx.beginPath();
        ctx.arc(cx, cy, nodeRadius, Math.PI * 0.5, Math.PI * 1.5);
        ctx.closePath();
        ctx.fillStyle = color1;
        ctx.fill();

        // Right semicircle (direction 2)
        ctx.beginPath();
        ctx.arc(cx, cy, nodeRadius, Math.PI * 1.5, Math.PI * 0.5);
        ctx.closePath();
        ctx.fillStyle = color2;
        ctx.fill();
      }

      // Interpolate and draw vehicles
      if (!currFrame || !currFrame.vehicles || currFrame.vehicles.length === 0) {
        rafIdRef.current = requestAnimationFrame(draw);
        return;
      }

      const prevFrame = prevFrameRef.current;
      const now = Date.now();
      const rawT = (now - frameArrivalRef.current) / FRAME_INTERVAL;
      const t = Math.min(Math.max(rawT, 0), 1);

      // Build lookup for prev frame vehicles by id
      const prevVehicleMap = new Map<string, { x: number; y: number; speed: number; angle: number }>();
      if (prevFrame?.vehicles) {
        for (const v of prevFrame.vehicles) {
          prevVehicleMap.set(v.id, v);
        }
      }

      const vehLen = Math.max(8, scale * 3.5);
      const vehWid = Math.max(4, scale * 1.8);

      for (const v of currFrame.vehicles) {
        let drawX = v.x;
        let drawY = v.y;
        let drawAngle = v.angle;
        let drawSpeed = v.speed;

        const prev = prevVehicleMap.get(v.id);
        if (prev) {
          // Lerp position
          drawX = prev.x + (v.x - prev.x) * t;
          drawY = prev.y + (v.y - prev.y) * t;
          drawAngle = lerpAngle(prev.angle, v.angle, t);
          drawSpeed = prev.speed + (v.speed - prev.speed) * t;
        }

        const [vx, vy] = toCanvas(drawX, drawY);
        ctx.save();
        ctx.translate(vx, vy);
        ctx.rotate(((drawAngle - 90) * Math.PI) / 180);
        ctx.fillStyle = drawSpeed < 0.5 ? "#ef4444" : "#e2e8f0";
        ctx.fillRect(-vehLen / 2, -vehWid / 2, vehLen, vehWid);
        ctx.restore();
      }

      rafIdRef.current = requestAnimationFrame(draw);
    }

    rafIdRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafIdRef.current);
  }, []); // runs once — loop is self-sustaining

  return (
    <div ref={containerRef} className={className} style={{ position: "relative", minHeight: 400 }}>
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", top: 0, left: 0, background: "#0f1117" }}
        aria-label="Traffic simulation canvas"
      />
    </div>
  );
}
