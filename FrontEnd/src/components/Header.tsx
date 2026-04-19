import { TrafficCone, RotateCcw, Pause, Play } from "lucide-react";

interface HeaderProps {
  clock: number;
  running: boolean;
  onToggle: () => void;
  onReset: () => void;
}

export function Header({ clock, running, onToggle, onReset }: HeaderProps) {
  return (
    <header className="h-[64px] px-6 flex items-center justify-between bg-[var(--surface)] border-b border-[var(--border)] shrink-0">
      <div className="flex items-center gap-3">
        <div
          className="w-8 h-8 rounded-[8px] flex items-center justify-center"
          style={{
            background:
              "linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%)",
          }}
        >
          <TrafficCone className="w-[18px] h-[18px] text-[var(--bg)]" />
        </div>
        <span className="font-sans font-bold text-[18px] tracking-[0.5px] text-[var(--text-primary)]">
          ATLAS
        </span>
        <span className="font-mono text-[13px] text-[var(--text-dim)]">
          / Traffic Strategy Comparison
        </span>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-[6px] bg-[var(--surface-2)] border border-[var(--border)] rounded-[6px]">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-cyan)]" />
          <span className="font-mono text-[12px] text-[var(--accent-cyan)] tracking-[0.4px]">
            t = {String(clock).padStart(4, "0")}s
          </span>
        </div>
        <button
          onClick={onReset}
          className="w-9 h-9 rounded-[8px] bg-[var(--surface-2)] border border-[var(--border)] flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          aria-label="Reset"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
        <button
          onClick={onToggle}
          className="w-9 h-9 rounded-[8px] bg-[var(--surface-2)] border border-[var(--border)] flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          aria-label={running ? "Pause" : "Play"}
        >
          {running ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
        <button
          onClick={onToggle}
          className="h-9 px-[14px] rounded-[8px] bg-[var(--accent-cyan)] text-[var(--bg)] flex items-center gap-1.5 hover:brightness-110 transition-all"
        >
          {running ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          <span className="font-sans font-semibold text-[13px]">
            {running ? "Pause" : "Run"}
          </span>
        </button>
      </div>
    </header>
  );
}
