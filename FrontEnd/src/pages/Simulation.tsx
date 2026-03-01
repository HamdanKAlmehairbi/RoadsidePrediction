import { useState, useMemo, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Play, Pause } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { SimCanvas } from "@/components/SimCanvas";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { useSimStream } from "@/hooks/useSimStream";
import { getNetworks, getNetwork, getWeights, startSimulation } from "@/lib/api";

const Simulation = () => {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState("1");
  const [policy, setPolicy] = useState("FedRL-naive");
  const [topology, setTopology] = useState("grid-3x3");
  const [seed, setSeed] = useState(42);
  const [jobId, setJobId] = useState<string | null>(null);
  const [rewardData, setRewardData] = useState<Array<{ t: number; reward: number }>>([]);
  const lastStepRef = useRef(-1);

  const { data: networksData } = useQuery({ queryKey: ["networks"], queryFn: getNetworks });
  const { data: weightsData } = useQuery({ queryKey: ["weights"], queryFn: getWeights });
  const { data: networkLayout } = useQuery({
    queryKey: ["network", topology],
    queryFn: () => getNetwork(topology),
    enabled: !!topology,
  });

  const { latestFrame, isConnected, isDone } = useSimStream(jobId);

  const topologies = networksData?.networks ?? ["grid-3x3", "grid-5x5", "grid-7x7"];

  const policies = useMemo(() => {
    if (!weightsData?.weights) return ["FedRL-naive", "MARL", "SARL", "Fixed Timing"];
    const unique = [...new Set(weightsData.weights.map(w => w.id))];
    return [...unique, "Fixed Timing"];
  }, [weightsData]);

  // Accumulate reward data when new frames arrive
  useEffect(() => {
    if (latestFrame && latestFrame.step !== lastStepRef.current) {
      lastStepRef.current = latestFrame.step;
      setRewardData(prev => [...prev, { t: latestFrame.step, reward: latestFrame?.metrics?.mean_reward ?? 0 }]);
    }
  }, [latestFrame]);

  const handleRun = async () => {
    setRewardData([]);
    lastStepRef.current = -1;
    try {
      const weightId = policy === "Fixed Timing" ? "fixed-timing" : policy;
      const result = await startSimulation(weightId, topology, seed);
      setJobId(result.job_id);
      setPlaying(true);
    } catch (err) {
      console.error("Failed to start simulation:", err);
    }
  };

  const sparkData = useMemo(() => {
    if (rewardData.length === 0) return Array.from({ length: 20 }, () => ({ v: -0.4 + Math.random() * 0.3 }));
    return rewardData.slice(-20).map(d => ({ v: d.reward }));
  }, [rewardData]);

  return (
    <ErrorBoundary>
    <div className="flex gap-4 h-[calc(100vh-5rem)]">
      {/* Main Area */}
      <div className="flex-1 flex flex-col gap-4">
        {/* Controls Bar */}
        <div className="flex flex-wrap items-center gap-3 bg-card border border-border rounded-lg p-3">
          <Select value={policy} onValueChange={setPolicy}>
            <SelectTrigger className="w-40 bg-muted border-border"><SelectValue placeholder="Policy" /></SelectTrigger>
            <SelectContent>{policies.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
          </Select>
          <Select value={topology} onValueChange={setTopology}>
            <SelectTrigger className="w-28 bg-muted border-border"><SelectValue placeholder="Topology" /></SelectTrigger>
            <SelectContent>{topologies.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
          </Select>
          <Input
            type="number"
            placeholder="Seed"
            value={seed}
            onChange={e => setSeed(Number(e.target.value))}
            className="w-24 bg-muted border-border"
          />
          <Button className="bg-seal-blue hover:bg-seal-blue/90 text-white" onClick={handleRun}>
            Run Simulation
          </Button>
          <Button variant="outline" size="icon" onClick={() => setPlaying(!playing)}>
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          {["1", "2", "5"].map(s => (
            <Button key={s} variant={speed === s ? "default" : "outline"} size="sm" onClick={() => setSpeed(s)}>
              {s}x
            </Button>
          ))}
          {isConnected && <span className="text-xs text-seal-teal ml-2">Connected</span>}
          {isDone && <span className="text-xs text-muted-foreground ml-2">Done</span>}
        </div>

        {/* Canvas */}
        <div className="flex-1 bg-card border border-border rounded-lg relative overflow-hidden flex items-center justify-center">
          <SimCanvas
            layout={networkLayout ?? null}
            frame={latestFrame}
            className="w-full h-full"
          />
        </div>
      </div>

      {/* Right Sidebar */}
      <div className="w-[280px] flex flex-col gap-3 shrink-0">
        {[
          { label: "Halted Vehicles", value: latestFrame ? String(latestFrame?.metrics?.total_halted ?? 0) : "—" },
          { label: "Mean Speed", value: latestFrame ? `${(latestFrame?.metrics?.mean_speed ?? 0).toFixed(1)} m/s` : "— m/s" },
        ].map(m => (
          <Card key={m.label} className="bg-card border-border">
            <CardContent className="p-4">
              <p className="text-xs text-muted-foreground">{m.label}</p>
              <p className="text-2xl font-mono font-bold">{m.value}</p>
            </CardContent>
          </Card>
        ))}

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Mean Reward</p>
            <p className="text-2xl font-mono font-bold">
              {latestFrame ? (latestFrame?.metrics?.mean_reward ?? 0).toFixed(2) : "—"}
            </p>
            <div className="h-8 mt-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sparkData}>
                  <Line type="monotone" dataKey="v" stroke="hsl(168, 76%, 40%)" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Timestep</p>
            <p className="text-2xl font-mono font-bold">
              {latestFrame ? latestFrame.step : "—"} <span className="text-sm text-muted-foreground">/ 360</span>
            </p>
          </CardContent>
        </Card>

        {/* Cumulative Reward Chart */}
        <Card className="bg-card border-border flex-1">
          <CardHeader className="p-4 pb-0">
            <CardTitle className="text-sm">Cumulative Reward</CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-2 h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rewardData.length > 0 ? rewardData : [{ t: 0, reward: 0 }]}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(234, 14%, 20%)" />
                <XAxis dataKey="t" tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
                <Tooltip contentStyle={{ background: "hsl(230, 18%, 13%)", border: "1px solid hsl(234, 14%, 20%)", borderRadius: 8 }} />
                <Line type="monotone" dataKey="reward" stroke="hsl(168, 76%, 40%)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground mb-2">Legend</p>
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center gap-2">
                <span className="inline-block w-3 h-2 rounded-sm" style={{ background: "#e2e8f0" }} />
                <span>Moving vehicle</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-block w-3 h-2 rounded-sm" style={{ background: "#ef4444" }} />
                <span>Halted vehicle</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-block w-3 h-3 rounded-full" style={{ background: "#22c55e" }} />
                <span>Green light</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-block w-3 h-3 rounded-full" style={{ background: "#eab308" }} />
                <span>Yellow light</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-block w-3 h-3 rounded-full" style={{ background: "#ef4444" }} />
                <span>Red light</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
    </ErrorBoundary>
  );
};

export default Simulation;
