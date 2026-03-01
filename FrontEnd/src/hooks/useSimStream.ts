import { useEffect, useRef, useState, useCallback } from "react";

export interface SimFrame {
  step: number;
  done: boolean;
  vehicles: Array<{ id: string; x: number; y: number; speed: number; angle: number }>;
  traffic_lights: Array<{
    id: string; state: string; lane_occupancy: number;
    halted_lane_occupancy: number; reward: number; global_rank: number;
  }>;
  metrics: { total_halted: number; mean_speed: number; mean_reward: number };
}

export function useSimStream(jobId: string | null) {
  const [latestFrame, setLatestFrame] = useState<SimFrame | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const frameBufferRef = useRef<SimFrame[]>([]);

  const connect = useCallback(() => {
    if (!jobId) return;
    const ws = new WebSocket(`ws://localhost:8000/ws/simulate/${jobId}`);
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const frame: SimFrame = JSON.parse(event.data);
      frameBufferRef.current.push(frame);
      setLatestFrame(frame);
      if (frame.done) {
        setIsDone(true);
        ws.close();
      }
    };
    ws.onclose = () => setIsConnected(false);
    ws.onerror = (e) => console.error("SimStream WS error", e);
  }, [jobId]);

  useEffect(() => {
    if (jobId) {
      setLatestFrame(null);
      setIsDone(false);
      frameBufferRef.current = [];
      connect();
    }
    return () => wsRef.current?.close();
  }, [jobId, connect]);

  return { latestFrame, frameBuffer: frameBufferRef, isConnected, isDone };
}
