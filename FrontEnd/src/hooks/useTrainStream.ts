import { useEffect, useRef, useState, useCallback } from "react";

export interface TrainEpisode {
  episode: number;
  mean_reward: number;
  trainer: string;
  fed_round: boolean;
}

export function useTrainStream(jobId: string | null) {
  const [episodes, setEpisodes] = useState<TrainEpisode[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (!jobId) return;
    const ws = new WebSocket(`ws://localhost:8000/ws/train/${jobId}`);
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const ep: TrainEpisode = JSON.parse(event.data);
      setEpisodes(prev => [...prev, ep]);
    };
    ws.onclose = () => {
      setIsConnected(false);
      setIsDone(true);
    };
    ws.onerror = (e) => console.error("TrainStream WS error", e);
  }, [jobId]);

  useEffect(() => {
    if (jobId) connect();
    return () => wsRef.current?.close();
  }, [jobId, connect]);

  return { episodes, isConnected, isDone };
}
