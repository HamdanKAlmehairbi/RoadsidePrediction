import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Play, Pause } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

// TODO: replace with fetch('/api/...') for live data
const mockRewardData = Array.from({ length: 360 }, (_, i) => ({
  t: i,
  reward: -0.8 + 0.6 * (1 - Math.exp(-i / 120)) + (Math.random() - 0.5) * 0.05,
}));

const sparkData = Array.from({ length: 20 }, (_, i) => ({ v: -0.4 + Math.random() * 0.3 }));

const policies = ["FedRL-naive", "MARL", "SARL", "Fixed Timing"];
const topologies = ["3×3", "5×5", "7×7"];

const Simulation = () => {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState("1");

  return (
    <div className="flex gap-4 h-[calc(100vh-5rem)]">
      {/* Main Area */}
      <div className="flex-1 flex flex-col gap-4">
        {/* Controls Bar */}
        <div className="flex flex-wrap items-center gap-3 bg-card border border-border rounded-lg p-3">
          <Select defaultValue="FedRL-naive">
            <SelectTrigger className="w-40 bg-muted border-border"><SelectValue placeholder="Policy" /></SelectTrigger>
            <SelectContent>{policies.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
          </Select>
          <Select defaultValue="3×3">
            <SelectTrigger className="w-28 bg-muted border-border"><SelectValue placeholder="Topology" /></SelectTrigger>
            <SelectContent>{topologies.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
          </Select>
          <Input type="number" placeholder="Seed" defaultValue={42} className="w-24 bg-muted border-border" />
          <Button className="bg-seal-blue hover:bg-seal-blue/90 text-white">Run Simulation</Button>
          <Button variant="outline" size="icon" onClick={() => setPlaying(!playing)}>
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          {["1", "2", "5"].map(s => (
            <Button key={s} variant={speed === s ? "default" : "outline"} size="sm" onClick={() => setSpeed(s)}>
              {s}×
            </Button>
          ))}
        </div>

        {/* Canvas Placeholder */}
        <div className="flex-1 bg-card border border-border rounded-lg relative overflow-hidden">
          {/* TODO: SimCanvas component renders roads + vehicles + TLS here via requestAnimationFrame from WS /ws/simulate/{job_id} */}
          <svg viewBox="0 0 600 600" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
            {/* Roads */}
            {[0, 1, 2].map(r => (
              <g key={`row-${r}`}>
                <rect x={0} y={180 * r + 85} width={600} height={30} className="fill-muted/30" />
                <rect x={180 * r + 85} y={0} width={30} height={600} className="fill-muted/30" />
              </g>
            ))}
            {/* Traffic lights */}
            {[0, 1, 2].flatMap(r => [0, 1, 2].map(c => {
              const colors = ["#22c55e", "#f59e0b", "#ef4444"];
              const color = colors[(r * 3 + c) % 3];
              return <circle key={`tl-${r}-${c}`} cx={200 * r + 100} cy={200 * c + 100} r={8} fill={color} opacity={0.8} />;
            }))}
            {/* Vehicles */}
            {Array.from({ length: 20 }, (_, i) => (
              <rect key={`v-${i}`} x={50 + (i * 137) % 500} y={90 + (i * 89) % 420} width={6} height={4} rx={1} className="fill-foreground/60" />
            ))}
          </svg>
        </div>
      </div>

      {/* Right Sidebar */}
      <div className="w-[280px] flex flex-col gap-3 shrink-0">
        {[
          { label: "Halted Vehicles", value: "14" },
          { label: "Mean Speed", value: "8.3 m/s" },
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
            <p className="text-2xl font-mono font-bold">−0.24</p>
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
            <p className="text-2xl font-mono font-bold">42 <span className="text-sm text-muted-foreground">/ 360</span></p>
          </CardContent>
        </Card>

        {/* Cumulative Reward Chart */}
        <Card className="bg-card border-border flex-1">
          <CardHeader className="p-4 pb-0">
            <CardTitle className="text-sm">Cumulative Reward</CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-2 h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockRewardData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(234, 14%, 20%)" />
                <XAxis dataKey="t" tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
                <Tooltip contentStyle={{ background: "hsl(230, 18%, 13%)", border: "1px solid hsl(234, 14%, 20%)", borderRadius: 8 }} />
                <Line type="monotone" dataKey="reward" stroke="hsl(168, 76%, 40%)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Simulation;
