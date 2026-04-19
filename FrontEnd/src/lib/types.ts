export type Topology = "grid-3x3" | "grid-5x5";

export type Strategy =
  | "MARL"
  | "SARL"
  | "FedRL"
  | "Gossip"
  | "HierFed"
  | "FedDistill"
  | "MeanField"
  | "CTDE"
  | "fixed-time"
  | "max-pressure";

export type SignalState = "red" | "yellow" | "green";

export interface Vehicle {
  id: string;
  x: number;
  y: number;
  orientation: "h" | "v";
  flip?: boolean;
  color: string;
}

export interface Signal {
  id: string;
  x: number;
  y: number;
  state: SignalState;
}

export interface Metrics {
  avgWait: number;
  throughput: number;
  completed: number;
  queued: number;
}

export interface SimState {
  vehicles: Vehicle[];
  signals: Signal[];
  metrics: Metrics;
}

export interface PanelConfig {
  topology: Topology;
  strategy: Strategy;
}
