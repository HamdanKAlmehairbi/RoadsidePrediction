import { useEffect, useMemo, useRef, useState } from "react";
import { Header } from "@/components/Header";
import { ConfigBar } from "@/components/ConfigBar";
import { SimPanel } from "@/components/SimPanel";
import { ComparisonBar } from "@/components/ComparisonBar";
import { makeMockState } from "@/lib/mockData";
import { useCompareSession } from "@/hooks/useCompareSession";
import type { PanelConfig, SimState } from "@/lib/types";

export default function App() {
  const [configA, setConfigA] = useState<PanelConfig>({
    topology: "grid-3x3",
    strategy: "Gossip",
  });
  const [configB, setConfigB] = useState<PanelConfig>({
    topology: "grid-3x3",
    strategy: "fixed-time",
  });

  const [running, setRunning] = useState(false);
  const [clock, setClock] = useState(0);
  const [mockA, setMockA] = useState<SimState>(() => makeMockState(configA.topology, "flowing"));
  const [mockB, setMockB] = useState<SimState>(() => makeMockState(configB.topology, "congested"));

  const live = useCompareSession(configA.topology, configB.topology);

  // Regenerate mock states when config changes (only when no live session is active).
  useEffect(() => {
    if (live.active) return;
    const scenario = isBaselineOrSlow(configA.strategy) ? "congested" : "flowing";
    setMockA(makeMockState(configA.topology, scenario));
  }, [configA, live.active]);

  useEffect(() => {
    if (live.active) return;
    const scenario = isBaselineOrSlow(configB.strategy) ? "congested" : "flowing";
    setMockB(makeMockState(configB.topology, scenario));
  }, [configB, live.active]);

  // Clock tick when running.
  const tickRef = useRef<number | null>(null);
  useEffect(() => {
    if (!running) {
      if (tickRef.current) window.clearInterval(tickRef.current);
      tickRef.current = null;
      return;
    }
    tickRef.current = window.setInterval(() => {
      setClock((c) => c + 1);
      // Refresh mock state periodically only when there's no active live session.
      if (!live.active && Math.random() < 0.33) {
        const scenarioA = isBaselineOrSlow(configA.strategy) ? "congested" : "flowing";
        const scenarioB = isBaselineOrSlow(configB.strategy) ? "congested" : "flowing";
        setMockA(makeMockState(configA.topology, scenarioA));
        setMockB(makeMockState(configB.topology, scenarioB));
      }
    }, 1000);
    return () => {
      if (tickRef.current) window.clearInterval(tickRef.current);
    };
  }, [running, configA, configB, live.active]);

  const toggleRun = async () => {
    if (running) {
      await live.stop();
      setRunning(false);
    } else {
      setRunning(true);
      setClock(0);
      // Kick off a live session in the background; if it fails we fall back to mock.
      live.start(configA, configB).catch(() => undefined);
    }
  };

  const resetAll = async () => {
    await live.stop();
    setRunning(false);
    setClock(0);
  };

  // Once a live session is active, prefer live frames and keep showing the last
  // live frame even if the WS has closed (workers finished). Only fall back to
  // mock when no session is active (idle or after reset).
  const stateA = live.active ? (live.stateA ?? mockA) : mockA;
  const stateB = live.active ? (live.stateB ?? mockB) : mockB;

  const seed = useMemo(() => 42, []);

  return (
    <div className="h-full w-full flex flex-col">
      <Header
        clock={clock}
        running={running}
        onToggle={toggleRun}
        onReset={resetAll}
      />
      <ConfigBar
        configA={configA}
        configB={configB}
        onChangeA={setConfigA}
        onChangeB={setConfigB}
      />
      <main className="flex-1 flex min-h-0 bg-[var(--bg)]">
        <SimPanel side="A" config={configA} state={stateA} running={running} divider="right" />
        <SimPanel side="B" config={configB} state={stateB} running={running} />
      </main>
      <ComparisonBar
        configA={configA}
        configB={configB}
        metricsA={stateA.metrics}
        metricsB={stateB.metrics}
        seed={seed}
      />
      {live.error && (
        <div className="fixed bottom-16 right-4 px-3 py-2 bg-[var(--signal-red)] text-white text-xs font-mono rounded-[6px] shadow-lg">
          Live session: {live.error} — falling back to preview mode
        </div>
      )}
    </div>
  );
}

function isBaselineOrSlow(strategy: PanelConfig["strategy"]): boolean {
  return ["fixed-time", "max-pressure", "MARL", "CTDE"].includes(strategy);
}
