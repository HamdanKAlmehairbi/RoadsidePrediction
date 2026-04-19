import { Zap, TrendingDown, TrendingUp, Minus, Dices } from "lucide-react";
import type { Metrics, PanelConfig } from "@/lib/types";

interface ComparisonBarProps {
  configA: PanelConfig;
  configB: PanelConfig;
  metricsA: Metrics;
  metricsB: Metrics;
  seed: number;
}

export function ComparisonBar({ configA, configB, metricsA, metricsB, seed }: ComparisonBarProps) {
  const betterSide = metricsA.avgWait <= metricsB.avgWait ? "A" : "B";
  const winner = betterSide === "A" ? configA : configB;
  const loser = betterSide === "A" ? configB : configA;
  const winMetrics = betterSide === "A" ? metricsA : metricsB;
  const lossMetrics = betterSide === "A" ? metricsB : metricsA;

  const waitPct =
    lossMetrics.avgWait > 0
      ? Math.round(((lossMetrics.avgWait - winMetrics.avgWait) / lossMetrics.avgWait) * 100)
      : 0;
  const tripsDelta = winMetrics.completed - lossMetrics.completed;
  const queueDelta = winMetrics.queued - lossMetrics.queued;

  return (
    <div className="h-[52px] px-6 flex items-center justify-between bg-[var(--surface)] border-t border-[var(--border)] shrink-0">
      <div className="flex items-center gap-2.5">
        <Zap className="w-3.5 h-3.5 text-[var(--accent-cyan)]" />
        <span className="font-sans font-medium text-[13px] text-[var(--text-primary)]">
          {winner.strategy} outperforms {loser.strategy} on {winner.topology}
        </span>
      </div>

      <div className="flex items-center gap-8">
        <Delta
          icon={<TrendingDown className="w-3 h-3" />}
          text={`${waitPct >= 0 ? "-" : "+"}${Math.abs(waitPct)}% wait time`}
        />
        <Delta
          icon={tripsDelta >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          text={`${tripsDelta >= 0 ? "+" : ""}${tripsDelta} more trips`}
        />
        <Delta
          icon={<Minus className="w-3 h-3" />}
          text={`${queueDelta >= 0 ? "+" : ""}${queueDelta} fewer queued`}
        />
      </div>

      <div className="flex items-center gap-2 px-2.5 py-1 bg-[var(--bg)] border border-[var(--border)] rounded-[6px]">
        <Dices className="w-3 h-3 text-[var(--text-dim)]" />
        <span className="font-mono text-[11px] text-[var(--text-dim)]">seed</span>
        <span className="font-mono font-semibold text-[11px] text-[var(--text-primary)]">
          {seed}
        </span>
      </div>
    </div>
  );
}

function Delta({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="flex items-center gap-1.5" style={{ color: "var(--state-green)" }}>
      {icon}
      <span className="font-mono font-semibold text-[12px]">{text}</span>
    </div>
  );
}
