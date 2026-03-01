import { useEffect, useRef, useCallback } from "react";
import type { NetworkLayout } from "@/lib/api";
import type { SimFrame } from "@/hooks/useSimStream";

interface SimCanvasProps {
  layout: NetworkLayout | null;
  frame: SimFrame | null;
  className?: string;
}

function tlsColor(state: string): string {
  if (!state) return "#6b7280";
  const s = state.toLowerCase();
  if (s.includes("g")) return "#22c55e";
  if (s.includes("y")) return "#eab308";
  return "#ef4444";
}

export function SimCanvas({ layout, frame, className }: SimCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

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
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cw = rect.width;
    const ch = rect.height;

    // Clear
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = "#0f1117";
    ctx.fillRect(0, 0, cw, ch);

    if (!layout) return;

    const { min_x, min_y, max_x, max_y } = layout.bounds;
    const padding = 40;
    const worldW = max_x - min_x || 1;
    const worldH = max_y - min_y || 1;
    const scaleX = (cw - padding * 2) / worldW;
    const scaleY = (ch - padding * 2) / worldH;
    const scale = Math.min(scaleX, scaleY);
    // Center the network in the canvas
    const offsetX = (cw - worldW * scale) / 2;
    const offsetY = (ch - worldH * scale) / 2;

    function toCanvas(x: number, y: number): [number, number] {
      return [
        (x - min_x) * scale + offsetX,
        ch - ((y - min_y) * scale + offsetY), // flip Y (SUMO is bottom-up)
      ];
    }

    // Draw roads (lane shapes as thick lines)
    ctx.lineCap = "round";
    ctx.strokeStyle = "#2a2d3e";
    ctx.lineWidth = Math.max(10, scale * 5);
    for (const edge of layout.edges) {
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
    for (const edge of layout.edges) {
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

    // Draw intersection circles (nodes)
    // Radius must cover the ~7-unit gap between lane endpoints and junction center
    const nodeRadius = Math.max(10, scale * 8);
    for (const node of layout.nodes) {
      const [cx, cy] = toCanvas(node.x, node.y);
      const tls = frame?.traffic_lights?.find((t) => t.id === node.id);
      // Glow effect
      if (tls) {
        ctx.beginPath();
        ctx.arc(cx, cy, nodeRadius + 4, 0, Math.PI * 2);
        ctx.fillStyle = tlsColor(tls.state) + "40"; // 25% opacity glow
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(cx, cy, nodeRadius, 0, Math.PI * 2);
      ctx.fillStyle = tls ? tlsColor(tls.state) : "#6b7280";
      ctx.fill();
    }

    if (!frame || frame.done || !frame.vehicles) return;

    // Draw vehicles
    const vehLen = Math.max(8, scale * 3.5);
    const vehWid = Math.max(4, scale * 1.8);
    for (const v of frame.vehicles) {
      const [vx, vy] = toCanvas(v.x, v.y);
      ctx.save();
      ctx.translate(vx, vy);
      // SUMO angles: 0°=north, 90°=east, 180°=south, 270°=west
      // Canvas: 0° = right. We need to map SUMO north to canvas up.
      // Canvas rotation: clockwise from right. So SUMO 0° (north) = -90° canvas.
      ctx.rotate(((v.angle - 90) * Math.PI) / 180);
      ctx.fillStyle = v.speed < 0.5 ? "#ef4444" : "#e2e8f0";
      ctx.fillRect(-vehLen / 2, -vehWid / 2, vehLen, vehWid);
      ctx.restore();
    }
  }, [layout, frame]);

  // Redraw on layout/frame change
  useEffect(() => {
    draw();
  }, [draw]);

  // Resize observer to handle container resizing
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(container);
    return () => observer.disconnect();
  }, [draw]);

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
