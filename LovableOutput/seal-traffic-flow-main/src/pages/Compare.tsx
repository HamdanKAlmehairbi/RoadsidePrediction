import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

// TODO: replace with fetch('/api/...') for live data
const policies = ["FedRL-naive", "MARL", "SARL", "Fixed Timing"];
const topologies = ["3×3", "5×5", "7×7"];

const policyColor: Record<string, string> = {
  "FedRL-naive": "hsl(168, 76%, 40%)",
  "MARL": "hsl(38, 92%, 50%)",
  "SARL": "hsl(271, 81%, 56%)",
  "Fixed Timing": "hsl(215, 16%, 57%)",
};

const mockCompareData = Array.from({ length: 360 }, (_, i) => ({
  t: i,
  A: -0.8 + 0.42 * (1 - Math.exp(-i / 140)) + (Math.random() - 0.5) * 0.03,
  B: -0.8 + 0.62 * (1 - Math.exp(-i / 100)) + (Math.random() - 0.5) * 0.03,
}));

const metrics = [
  { label: "Halted", a: 14, b: 9, unit: "", better: "b" },
  { label: "Speed", a: 7.2, b: 9.8, unit: " m/s", better: "b" },
  { label: "Reward", a: -0.41, b: -0.22, unit: "", better: "b" },
];

const Compare = () => {
  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 bg-card border border-border rounded-lg p-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Policy A</span>
          <Select defaultValue="MARL">
            <SelectTrigger className="w-40 bg-muted border-border"><SelectValue /></SelectTrigger>
            <SelectContent>{policies.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <Select defaultValue="3×3">
          <SelectTrigger className="w-28 bg-muted border-border"><SelectValue /></SelectTrigger>
          <SelectContent>{topologies.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
        </Select>
        <Input type="number" placeholder="Seed" defaultValue={42} className="w-24 bg-muted border-border" />
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Policy B</span>
          <Select defaultValue="FedRL-naive">
            <SelectTrigger className="w-40 bg-muted border-border"><SelectValue /></SelectTrigger>
            <SelectContent>{policies.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <Button className="bg-seal-blue hover:bg-seal-blue/90 text-white ml-auto">Run Both</Button>
      </div>

      {/* TODO: two useSimStream hooks run in parallel with same seed */}

      {/* Split View */}
      <div className="flex gap-2">
        {/* Canvas A */}
        <div className="flex-[45] flex flex-col gap-2">
          <Badge className="bg-seal-amber/20 text-seal-amber border-seal-amber/30 w-fit">MARL</Badge>
          <div className="bg-card border border-border rounded-lg h-64 flex items-center justify-center text-muted-foreground text-sm">
            Simulation Canvas A
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
            Policy B better
          </Badge>
        </div>

        {/* Canvas B */}
        <div className="flex-[45] flex flex-col gap-2">
          <Badge className="bg-seal-teal/20 text-seal-teal border-seal-teal/30 w-fit">FedRL-naive</Badge>
          <div className="bg-card border border-border rounded-lg h-64 flex items-center justify-center text-muted-foreground text-sm">
            Simulation Canvas B
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
            <LineChart data={mockCompareData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(234, 14%, 20%)" />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} label={{ value: "Timestep", position: "insideBottom", offset: -5, fill: "hsl(215, 16%, 57%)" }} />
              <YAxis tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
              <Tooltip contentStyle={{ background: "hsl(230, 18%, 13%)", border: "1px solid hsl(234, 14%, 20%)", borderRadius: 8 }} />
              <Legend />
              <Line type="monotone" dataKey="A" name="MARL" stroke="hsl(38, 92%, 50%)" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="B" name="FedRL-naive" stroke="hsl(168, 76%, 40%)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
};

export default Compare;
