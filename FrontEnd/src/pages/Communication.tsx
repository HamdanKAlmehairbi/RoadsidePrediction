import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ScatterChart, Scatter, ZAxis, Cell, ReferenceLine } from "recharts";

// TODO: replace with fetch('/api/...') for live data
const barData = [
  { trainer: "SARL", policy: 0, observation: 0, rank: 0, action: 360, vehicle: 1200 },
  { trainer: "MARL", policy: 450, observation: 360, rank: 0, action: 360, vehicle: 1200 },
  { trainer: "FedRL-naive", policy: 120, observation: 360, rank: 720, action: 360, vehicle: 1200 },
  { trainer: "FedRL-pos", policy: 120, observation: 360, rank: 720, action: 360, vehicle: 1200 },
];

const scatterData = [
  { trainer: "SARL", messages: 1560, reward: -1.2, color: "hsl(271, 81%, 56%)" },
  { trainer: "MARL", messages: 2370, reward: -0.75, color: "hsl(38, 92%, 50%)" },
  { trainer: "FedRL-naive", messages: 2760, reward: -0.77, color: "hsl(168, 76%, 40%)", star: true },
  { trainer: "FedRL-pos", messages: 2760, reward: -0.79, color: "hsl(168, 76%, 40%)" },
];

const rawData = [
  { trainer: "SARL", policy: 0, obs: 0, rank: 0, action: 360, vehicle: 1200, total: 1560, reward: -1.20 },
  { trainer: "MARL", policy: 450, obs: 360, rank: 0, action: 360, vehicle: 1200, total: 2370, reward: -0.75 },
  { trainer: "FedRL-naive", policy: 120, obs: 360, rank: 720, action: 360, vehicle: 1200, total: 2760, reward: -0.77 },
  { trainer: "FedRL-pos_reward", policy: 120, obs: 360, rank: 720, action: 360, vehicle: 1200, total: 2760, reward: -0.79 },
];

const messageTypes = [
  { type: "EDGE→TLS Policy", desc: "Edge server sends updated policy weights to traffic lights.", cost: "~50KB per message" },
  { type: "TLS→EDGE Observation", desc: "Each traffic light sends local state observations to the edge.", cost: "~2KB per message" },
  { type: "EDGE→TLS Rank", desc: "Edge broadcasts ranking info to coordinate federated rounds.", cost: "~1KB per message" },
  { type: "EDGE→TLS Action", desc: "Edge sends computed actions back to traffic lights.", cost: "~0.5KB per message" },
  { type: "Vehicle→TLS Count", desc: "Vehicles report their presence to the nearest traffic light.", cost: "~0.1KB per message" },
];

const CustomScatterLabel = (props: any) => {
  const { cx, cy, payload } = props;
  if (!payload) return null;
  return (
    <text x={cx} y={(cy ?? 0) - 12} textAnchor="middle" fill="hsl(210, 40%, 92%)" fontSize={11}>
      {payload.star ? "★ " : ""}{payload.trainer}
    </text>
  );
};

const Communication = () => {
  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="rounded-lg bg-gradient-to-r from-seal-teal/20 to-seal-teal/5 border border-seal-teal/30 p-6">
        <p className="text-lg font-medium text-seal-teal">
          FedRL reduces communication overhead by 36.24% with only a 2.11% reward trade-off compared to centralised MARL.
        </p>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Stacked Bar */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Messages per Episode by Type</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(234, 14%, 20%)" />
                <XAxis dataKey="trainer" tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
                <Tooltip contentStyle={{ background: "hsl(230, 18%, 13%)", border: "1px solid hsl(234, 14%, 20%)", borderRadius: 8 }} />
                <Legend />
                <Bar dataKey="policy" name="Policy" stackId="a" fill="hsl(217, 91%, 60%)" />
                <Bar dataKey="observation" name="Observation" stackId="a" fill="hsl(168, 76%, 40%)" />
                <Bar dataKey="rank" name="Rank" stackId="a" fill="hsl(142, 71%, 45%)" />
                <Bar dataKey="action" name="Action" stackId="a" fill="hsl(38, 92%, 50%)" />
                <Bar dataKey="vehicle" name="Vehicle" stackId="a" fill="hsl(271, 81%, 56%)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Scatter */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Reward vs Communication Cost</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(234, 14%, 20%)" />
                <XAxis type="number" dataKey="messages" name="Messages" tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} label={{ value: "Total Messages", position: "insideBottom", offset: -5, fill: "hsl(215, 16%, 57%)" }} />
                <YAxis type="number" dataKey="reward" name="Reward" tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
                <ZAxis range={[100, 100]} />
                <Tooltip contentStyle={{ background: "hsl(230, 18%, 13%)", border: "1px solid hsl(234, 14%, 20%)", borderRadius: 8 }} />
                {/* Dashed Pareto frontier */}
                <ReferenceLine segment={[{ x: 1560, y: -1.2 }, { x: 2370, y: -0.75 }]} stroke="hsl(168, 76%, 40%)" strokeDasharray="6 4" opacity={0.5} />
                <ReferenceLine segment={[{ x: 2370, y: -0.75 }, { x: 2760, y: -0.77 }]} stroke="hsl(168, 76%, 40%)" strokeDasharray="6 4" opacity={0.5} />
                <Scatter data={scatterData} label={<CustomScatterLabel />}>
                  {scatterData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Explainer */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Message Types Explained</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {messageTypes.map(m => (
              <li key={m.type} className="text-sm">
                <span className="font-medium text-foreground">{m.type}</span>
                <span className="text-muted-foreground"> — {m.desc} </span>
                <span className="font-mono text-xs text-seal-teal">{m.cost}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Raw Data Table */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Raw Communication Data</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-border">
                <TableHead>Trainer</TableHead>
                <TableHead>Policy</TableHead>
                <TableHead>Obs</TableHead>
                <TableHead>Rank</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Vehicle</TableHead>
                <TableHead>Total</TableHead>
                <TableHead>Mean Reward</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rawData.map(row => (
                <TableRow key={row.trainer} className="border-border">
                  <TableCell className="font-medium">{row.trainer}</TableCell>
                  <TableCell className="font-mono">{row.policy}</TableCell>
                  <TableCell className="font-mono">{row.obs}</TableCell>
                  <TableCell className="font-mono">{row.rank}</TableCell>
                  <TableCell className="font-mono">{row.action}</TableCell>
                  <TableCell className="font-mono">{row.vehicle}</TableCell>
                  <TableCell className="font-mono font-bold">{row.total}</TableCell>
                  <TableCell className="font-mono">{row.reward.toFixed(2)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};

export default Communication;
