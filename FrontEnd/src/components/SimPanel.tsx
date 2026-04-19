import type { PanelConfig, SimState } from "@/lib/types";
import { NetworkCanvas } from "./NetworkCanvas";

interface SimPanelProps {
  side: "A" | "B";
  config: PanelConfig;
  state: SimState;
  running: boolean;
  divider?: "left" | "right";
}

export function SimPanel({ side, config, state, running, divider }: SimPanelProps) {
  const accent = side === "A" ? "var(--accent-cyan)" : "var(--accent-purple)";
  const borderClass =
    divider === "right"
      ? "border-r border-[var(--border)]"
      : divider === "left"
      ? "border-l border-[var(--border)]"
      : "";

  return (
    <div className={`flex-1 flex flex-col min-w-0 ${borderClass}`}>
      {/* Panel header */}
      <div className="px-6 py-3.5 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2.5">
          <span
            className="w-[22px] h-[22px] rounded-[5px] flex items-center justify-center font-mono font-bold text-[11px] text-[var(--bg)]"
            style={{ background: accent }}
          >
            {side}
          </span>
          <span className="font-sans font-semibold text-[14px] text-[var(--text-primary)]">
            {config.strategy} · {config.topology}
          </span>
        </div>
        <div
          className="flex items-center gap-1.5 px-2 py-[3px] rounded-[4px]"
          style={{ background: running ? "#0D2817" : "#1F1F22" }}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${running ? "animate-pulse" : ""}`}
            style={{ background: running ? "var(--state-green)" : "var(--text-faint)" }}
          />
          <span
            className="font-mono font-semibold text-[10px] tracking-[0.8px]"
            style={{ color: running ? "var(--state-green)" : "var(--text-faint)" }}
          >
            {running ? "RUNNING" : "IDLE"}
          </span>
        </div>
      </div>

      {/* Canvas area */}
      <div className="flex-1 bg-[#070A13] flex items-center justify-center p-6 min-h-0">
        <NetworkCanvas topology={config.topology} state={state} />
      </div>

      {/* Metrics row */}
      <MetricsRow metrics={state.metrics} />
    </div>
  );
}

function MetricsRow({ metrics }: { metrics: SimState["metrics"] }) {
  const waitColor =
    metrics.avgWait < 25
      ? "var(--state-green)"
      : metrics.avgWait < 50
      ? "var(--state-yellow)"
      : "var(--state-red)";
  const queuedColor =
    metrics.queued < 20
      ? "var(--state-yellow)"
      : metrics.queued < 40
      ? "var(--state-yellow)"
      : "var(--state-red)";

  return (
    <div className="px-6 py-4 flex bg-[var(--surface)] border-t border-[var(--border)] shrink-0">
      <MetricCell label="AVG WAIT" value={metrics.avgWait.toFixed(1)} unit="s" valueColor={waitColor} />
      <MetricCell label="THROUGHPUT" value={Math.round(metrics.throughput * 100).toString()} unit="%" />
      <MetricCell label="COMPLETED" value={metrics.completed.toString()} unit="trips" />
      <MetricCell label="QUEUED" value={metrics.queued.toString()} unit="cars" valueColor={queuedColor} />
    </div>
  );
}

function MetricCell({
  label,
  value,
  unit,
  valueColor,
}: {
  label: string;
  value: string;
  unit: string;
  valueColor?: string;
}) {
  return (
    <div className="flex-1 flex flex-col gap-1.5">
      <span className="font-mono text-[10px] tracking-[1.2px] text-[var(--text-dim)]">
        {label}
      </span>
      <div className="flex items-end gap-1">
        <span
          className="font-sans font-semibold text-[22px]"
          style={{ color: valueColor ?? "var(--text-primary)" }}
        >
          {value}
        </span>
        <span className="font-mono text-[12px] text-[var(--text-dim)] mb-[3px]">{unit}</span>
      </div>
    </div>
  );
}
