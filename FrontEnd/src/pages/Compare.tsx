import { useState, useMemo, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { SimCanvas } from "@/components/SimCanvas";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { useSimStream } from "@/hooks/useSimStream";
import { getNetworks, getNetwork, getWeights, startSimulation } from "@/lib/api";

const Compare = () => {
  const [policyA, setPolicyA] = useState("MARL");
  const [policyB, setPolicyB] = useState("FedRL-naive");
  const [topology, setTopology] = useState("grid-3x3");
  const [seed, setSeed] = useState(42);
  const [jobIdA, setJobIdA] = useState<string | null>(null);
  const [jobIdB, setJobIdB] = useState<string | null>(null);
  const [chartData, setChartData] = useState<Array<{ t: number; A: number; B: number }>>([]);
  const lastStepRef = useRef(-1);

  const { data: networksData } = useQuery({ queryKey: ["networks"], queryFn: getNetworks });
  const { data: weightsData } = useQuery({ queryKey: ["weights"], queryFn: getWeights });
  const { data: networkLayout } = useQuery({
    queryKey: ["network", topology],
    queryFn: () => getNetwork(topology),
    enabled: !!topology,
  });

  const streamA = useSimStream(jobIdA);
  const streamB = useSimStream(jobIdB);

  const topologies = networksData?.networks ?? ["grid-3x3", "grid-5x5", "grid-7x7"];
  const policies = useMemo(() => {
    if (!weightsData?.weights) return ["FedRL-naive", "MARL", "SARL", "Fixed Timing"];
    const unique = [...new Set(weightsData.weights.map(w => w.id))];
    return [...unique, "Fixed Timing"];
  }, [weightsData]);

  const frameA = streamA.latestFrame;
  const frameB = streamB.latestFrame;

  // Update chart data from both streams
  useEffect(() => {
    const stepToAdd = Math.max(frameA?.step ?? -1, frameB?.step ?? -1);
    if (stepToAdd > lastStepRef.current && (frameA || frameB)) {
      lastStepRef.current = stepToAdd;
      setChartData(prev => [...prev, {
        t: stepToAdd,
        A: frameA?.metrics?.mean_reward ?? (prev.length > 0 ? prev[prev.length - 1].A : 0),
        B: frameB?.metrics?.mean_reward ?? (prev.length > 0 ? prev[prev.length - 1].B : 0),
      }]);
    }
  }, [frameA, frameB]);

  const handleRunBoth = async () => {
    setChartData([]);
    lastStepRef.current = -1;
    try {
      const weightA = policyA === "Fixed Timing" ? "fixed-timing" : policyA;
      const weightB = policyB === "Fixed Timing" ? "fixed-timing" : policyB;
      const [resA, resB] = await Promise.all([
        startSimulation(weightA, topology, seed),
        startSimulation(weightB, topology, seed),
      ]);
      setJobIdA(resA.job_id);
      setJobIdB(resB.job_id);
    } catch (err) {
      console.error("Failed to start simulations:", err);
    }
  };

  const metrics = [
    {
      label: "Halted",
      a: frameA?.metrics?.total_halted ?? 0,
      b: frameB?.metrics?.total_halted ?? 0,
      unit: "",
      better: (frameA?.metrics?.total_halted ?? 0) < (frameB?.metrics?.total_halted ?? 0) ? "a" as const : "b" as const,
    },
    {
      label: "Speed",
      a: Number((frameA?.metrics?.mean_speed ?? 0).toFixed(1)),
      b: Number((frameB?.metrics?.mean_speed ?? 0).toFixed(1)),
      unit: " m/s",
      better: (frameA?.metrics?.mean_speed ?? 0) > (frameB?.metrics?.mean_speed ?? 0) ? "a" as const : "b" as const,
    },
    {
      label: "Reward",
      a: Number((frameA?.metrics?.mean_reward ?? 0).toFixed(2)),
      b: Number((frameB?.metrics?.mean_reward ?? 0).toFixed(2)),
      unit: "",
      better: (frameA?.metrics?.mean_reward ?? 0) > (frameB?.metrics?.mean_reward ?? 0) ? "a" as const : "b" as const,
    },
  ];

  const betterPolicy = metrics.filter(m => m.better === "b").length > metrics.filter(m => m.better === "a").length ? "B" : "A";

  return (
    <ErrorBoundary>
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 bg-card border border-border rounded-lg p-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Policy A</span>
          <Select value={policyA} onValueChange={setPolicyA}>
            <SelectTrigger className="w-40 bg-muted border-border"><SelectValue /></SelectTrigger>
            <SelectContent>{policies.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <Select value={topology} onValueChange={setTopology}>
          <SelectTrigger className="w-28 bg-muted border-border"><SelectValue /></SelectTrigger>
          <SelectContent>{topologies.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
        </Select>
        <Input
          type="number"
          placeholder="Seed"
          value={seed}
          onChange={e => setSeed(Number(e.target.value))}
          className="w-24 bg-muted border-border"
        />
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Policy B</span>
          <Select value={policyB} onValueChange={setPolicyB}>
            <SelectTrigger className="w-40 bg-muted border-border"><SelectValue /></SelectTrigger>
            <SelectContent>{policies.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <Button className="bg-seal-blue hover:bg-seal-blue/90 text-white ml-auto" onClick={handleRunBoth}>
          Run Both
        </Button>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-muted-foreground px-1">
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-2 rounded-sm" style={{ background: "#e2e8f0" }} />
          <span>Moving</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-2 rounded-sm" style={{ background: "#ef4444" }} />
          <span>Halted</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ background: "#22c55e" }} />
          <span>Green</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ background: "#eab308" }} />
          <span>Yellow</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ background: "#ef4444" }} />
          <span>Red</span>
        </div>
      </div>

      {/* Split View */}
      <div className="flex gap-2">
        {/* Canvas A */}
        <div className="flex-[45] flex flex-col gap-2">
          <Badge className="bg-seal-amber/20 text-seal-amber border-seal-amber/30 w-fit">{policyA}</Badge>
          <div className="bg-card border border-border rounded-lg h-64 flex items-center justify-center overflow-hidden">
            <SimCanvas layout={networkLayout ?? null} frame={streamA.latestFrame} className="w-full h-full" />
          </div>
        </div>

        {/* Diff Panel */}
        <div className="flex-[10] flex flex-col gap-2 justify-center">
          {metrics.map(m => (
            <Card key={m.label} className="bg-card border-border">
              <CardContent className="p-3 text-center">
                <p className="text-xs text-muted-foreground mb-1">{m.label}</p>
                <div className="flex justify-between text-sm font-mono">
                  <span className={m.better === "a" ? "text-green-400" : ""}>{m.a}{m.unit}</span>
                  <span className={m.better === "b" ? "text-green-400" : ""}>{m.b}{m.unit}</span>
                </div>
              </CardContent>
            </Card>
          ))}
          <Badge className="bg-seal-teal/20 text-seal-teal border-seal-teal/30 text-xs text-center">
            Policy {betterPolicy} better
          </Badge>
        </div>

        {/* Canvas B */}
        <div className="flex-[45] flex flex-col gap-2">
          <Badge className="bg-seal-teal/20 text-seal-teal border-seal-teal/30 w-fit">{policyB}</Badge>
          <div className="bg-card border border-border rounded-lg h-64 flex items-center justify-center overflow-hidden">
            <SimCanvas layout={networkLayout ?? null} frame={streamB.latestFrame} className="w-full h-full" />
          </div>
        </div>
      </div>

      {/* Comparison Chart */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Cumulative Mean Reward Comparison</CardTitle>
        </CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData.length > 0 ? chartData : [{ t: 0, A: 0, B: 0 }]}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(234, 14%, 20%)" />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} label={{ value: "Timestep", position: "insideBottom", offset: -5, fill: "hsl(215, 16%, 57%)" }} />
              <YAxis tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
              <Tooltip contentStyle={{ background: "hsl(230, 18%, 13%)", border: "1px solid hsl(234, 14%, 20%)", borderRadius: 8 }} />
              <Legend />
              <Line type="monotone" dataKey="A" name={policyA} stroke="#f59e0b" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="B" name={policyB} stroke="#14b8a6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
    </ErrorBoundary>
  );
};

export default Compare;
