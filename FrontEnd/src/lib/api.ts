const BASE_URL = "http://localhost:8000";

export interface NetworkLayout {
  topology: string;
  bounds: { min_x: number; min_y: number; max_x: number; max_y: number };
  nodes: Array<{ id: string; x: number; y: number }>;
  edges: Array<{
    id: string; from: string; to: string;
    lanes: Array<{ id: string; shape: [number, number][] }>;
  }>;
}

export interface WeightInfo {
  id: string; trainer: string; aggr: string | null;
  topology: string; ranked: boolean;
}

export interface JobResult {
  job_id: string; status: string; type: string;
  rewards: Array<{ episode: number; mean_reward: number; trainer: string; fed_round: boolean }>;
  trip_metrics: { travel_time_s: number; waiting_time_s: number; time_loss_s: number };
  comm_costs: { EDGE2TLS_POLICY: number; TLS2EDGE_OBS: number; EDGE2TLS_RANK: number; EDGE2TLS_ACTION: number; VEH2TLS: number };
}

export async function getNetworks(): Promise<{ networks: string[] }> {
  const res = await fetch(`${BASE_URL}/api/networks`);
  if (!res.ok) throw new Error("Failed to fetch networks");
  return res.json();
}

export async function getNetwork(topology: string): Promise<NetworkLayout> {
  const res = await fetch(`${BASE_URL}/api/network/${topology}`);
  if (!res.ok) throw new Error("Failed to fetch network");
  return res.json();
}

export async function getWeights(): Promise<{ weights: WeightInfo[] }> {
  const res = await fetch(`${BASE_URL}/api/weights`);
  if (!res.ok) throw new Error("Failed to fetch weights");
  return res.json();
}

export async function startSimulation(weight_id: string, topology: string, seed: number): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${BASE_URL}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weight_id, topology, seed }),
  });
  if (!res.ok) throw new Error("Failed to start simulation");
  return res.json();
}

export async function startTraining(params: {
  trainer: string; aggr: string; topology: string;
  ranked: boolean; n_episodes: number; fed_step: number;
}): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${BASE_URL}/api/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Failed to start training");
  return res.json();
}

export async function getResults(job_id: string): Promise<JobResult> {
  const res = await fetch(`${BASE_URL}/api/results/${job_id}`);
  if (!res.ok) throw new Error("Failed to fetch results");
  return res.json();
}
