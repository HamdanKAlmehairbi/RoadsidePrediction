import { Grid3x3, Brain, ChevronDown } from "lucide-react";
import type { PanelConfig, Strategy, Topology } from "@/lib/types";

const TOPOLOGIES: Topology[] = ["grid-3x3", "grid-5x5"];
const STRATEGIES: Strategy[] = [
  "MARL",
  "MeanField",
  "CTDE",
  "Gossip",
  "HierFed",
  "FedDistill",
  "FedRL",
  "SARL",
  "fixed-time",
  "max-pressure",
];

interface ConfigBarProps {
  configA: PanelConfig;
  configB: PanelConfig;
  onChangeA: (c: PanelConfig) => void;
  onChangeB: (c: PanelConfig) => void;
}

export function ConfigBar({ configA, configB, onChangeA, onChangeB }: ConfigBarProps) {
  return (
    <div className="h-[72px] px-6 flex items-center justify-between bg-[var(--surface)] border-b border-[var(--border)] shrink-0">
      <div className="flex items-center gap-3">
        <Badge label="A" color="var(--accent-cyan)" />
        <Selector
          icon={<Grid3x3 className="w-3.5 h-3.5 text-[var(--text-muted)]" />}
          label="TOPOLOGY"
          value={configA.topology}
          options={TOPOLOGIES}
          onSelect={(v) => onChangeA({ ...configA, topology: v as Topology })}
        />
        <Selector
          icon={<Brain className="w-3.5 h-3.5 text-[var(--text-muted)]" />}
          label="STRATEGY"
          value={configA.strategy}
          options={STRATEGIES}
          onSelect={(v) => onChangeA({ ...configA, strategy: v as Strategy })}
        />
      </div>

      <div className="flex flex-col items-center px-[10px] py-1">
        <span className="font-sans font-bold text-[14px] text-[var(--text-faint)] tracking-[1.5px]">
          VS
        </span>
        <span className="font-mono text-[9px] text-[var(--text-faint)]">compare</span>
      </div>

      <div className="flex items-center gap-3">
        <Selector
          icon={<Grid3x3 className="w-3.5 h-3.5 text-[var(--text-muted)]" />}
          label="TOPOLOGY"
          value={configB.topology}
          options={TOPOLOGIES}
          onSelect={(v) => onChangeB({ ...configB, topology: v as Topology })}
        />
        <Selector
          icon={<Brain className="w-3.5 h-3.5 text-[var(--text-muted)]" />}
          label="STRATEGY"
          value={configB.strategy}
          options={STRATEGIES}
          onSelect={(v) => onChangeB({ ...configB, strategy: v as Strategy })}
        />
        <Badge label="B" color="var(--accent-purple)" />
      </div>
    </div>
  );
}

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <div
      className="px-2 py-1 rounded-[4px] border flex items-center gap-1.5"
      style={{ borderColor: color, background: "var(--bg)" }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      <span className="font-mono font-semibold text-[11px]" style={{ color }}>
        {label}
      </span>
    </div>
  );
}

function Selector({
  icon,
  label,
  value,
  options,
  onSelect,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  options: string[];
  onSelect: (v: string) => void;
}) {
  return (
    <label className="relative flex items-center gap-2.5 px-3 py-2 bg-[var(--surface-2)] border border-[var(--border)] rounded-[8px] cursor-pointer hover:border-[var(--border-2)] transition-colors">
      {icon}
      <div className="flex flex-col gap-0.5">
        <span className="font-mono text-[9px] tracking-[0.8px] text-[var(--text-dim)]">
          {label}
        </span>
        <span className="font-mono font-medium text-[13px] text-[var(--text-primary)]">
          {value}
        </span>
      </div>
      <ChevronDown className="w-3.5 h-3.5 text-[var(--text-dim)]" />
      <select
        value={value}
        onChange={(e) => onSelect(e.target.value)}
        className="absolute inset-0 opacity-0 cursor-pointer"
      >
        {options.map((opt) => (
          <option key={opt} value={opt} className="bg-[var(--surface-2)]">
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}
