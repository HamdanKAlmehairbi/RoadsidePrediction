import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from "recharts";
import { Link } from "react-router-dom";

// TODO: replace with fetch('/api/...') for live data
const trainers = ["FedRL-naive", "FedRL-pos_reward", "MARL", "SARL"];
const topologies = ["3×3", "5×5", "7×7"];

const mockTrainingData = Array.from({ length: 50 }, (_, i) => ({
  episode: i + 1,
  "FedRL-naive": -3.5 + 2.7 * (1 - Math.exp(-i / 15)) + (Math.random() - 0.5) * 0.15,
  "MARL": -3.5 + 2.75 * (1 - Math.exp(-i / 18)) + (Math.random() - 0.5) * 0.15,
  "SARL": -3.5 + 2.3 * (1 - Math.exp(-i / 20)) + (Math.random() - 0.5) * 0.15,
}));

const summaryData = [
  { trainer: "FedRL-naive", final: -0.80, best: -0.72, episodes: 50, agg: "FedAvg" },
  { trainer: "MARL", final: -0.75, best: -0.68, episodes: 50, agg: "N/A" },
  { trainer: "SARL", final: -1.20, best: -1.05, episodes: 50, agg: "N/A" },
];

const Training = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [showExisting, setShowExisting] = useState(true);
  const [episodes, setEpisodes] = useState([50]);

  // TODO: useTrainStream connects to WS /ws/train/{job_id} and appends to chart data

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 bg-card border border-border rounded-lg p-3">
        <Select defaultValue="FedRL-naive">
          <SelectTrigger className="w-44 bg-muted border-border"><SelectValue placeholder="Trainer" /></SelectTrigger>
          <SelectContent>{trainers.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
        </Select>
        <Select defaultValue="3×3">
          <SelectTrigger className="w-28 bg-muted border-border"><SelectValue placeholder="Topology" /></SelectTrigger>
          <SelectContent>{topologies.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
        </Select>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Ranked</span>
          <Switch />
        </div>
        <div className="flex items-center gap-2 w-40">
          <span className="text-xs text-muted-foreground whitespace-nowrap">Episodes: {episodes[0]}</span>
          <Slider min={10} max={100} step={5} value={episodes} onValueChange={setEpisodes} />
        </div>
        <Button className="bg-seal-blue hover:bg-seal-blue/90 text-white" onClick={() => setIsTraining(!isTraining)}>
          {isTraining ? "Stop Training" : "Start Training"}
        </Button>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-xs text-muted-foreground">Show existing</span>
          <Switch checked={showExisting} onCheckedChange={setShowExisting} />
        </div>
      </div>

      {/* Main Chart */}
      {showExisting && (
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Training Progress — Mean Episode Reward</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockTrainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(234, 14%, 20%)" />
                <XAxis dataKey="episode" tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} label={{ value: "Episode", position: "insideBottom", offset: -5, fill: "hsl(215, 16%, 57%)" }} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
                <Tooltip contentStyle={{ background: "hsl(230, 18%, 13%)", border: "1px solid hsl(234, 14%, 20%)", borderRadius: 8 }} />
                <Legend />
                {[10, 20, 30, 40, 50].map(ep => (
                  <ReferenceLine key={ep} x={ep} stroke="hsl(168, 76%, 40%)" strokeDasharray="4 4" opacity={0.4} label={{ value: ep === 10 ? "Fed round" : "", position: "top", fill: "hsl(168, 76%, 40%)", fontSize: 9 }} />
                ))}
                <Line type="monotone" dataKey="FedRL-naive" stroke="hsl(168, 76%, 40%)" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="MARL" stroke="hsl(38, 92%, 50%)" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="SARL" stroke="hsl(271, 81%, 56%)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Status Bar */}
      {isTraining && (
        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="h-3 w-3 rounded-full bg-seal-teal animate-pulse" />
            <span className="text-sm font-mono">Episode 12 / 50</span>
            <span className="text-sm text-muted-foreground">Mean reward: <span className="font-mono">−1.24</span></span>
            <span className="text-sm text-muted-foreground">FedRL · Fed round: yes</span>
            <Progress value={24} className="flex-1 h-2" />
          </CardContent>
        </Card>
      )}

      {/* Summary Table */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Training Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-border">
                <TableHead>Trainer</TableHead>
                <TableHead>Final Reward</TableHead>
                <TableHead>Best Reward</TableHead>
                <TableHead>Episodes</TableHead>
                <TableHead>Aggregation Fn</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {summaryData.map(row => (
                <TableRow key={row.trainer} className="border-border">
                  <TableCell className="font-medium">{row.trainer}</TableCell>
                  <TableCell className="font-mono">{row.final.toFixed(2)}</TableCell>
                  <TableCell className="font-mono">{row.best.toFixed(2)}</TableCell>
                  <TableCell className="font-mono">{row.episodes}</TableCell>
                  <TableCell>{row.agg}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Post-training CTA */}
      {!isTraining && (
        <div className="flex justify-center">
          <Button asChild className="bg-seal-teal hover:bg-seal-teal/90 text-white">
            <Link to="/simulation">Simulate this policy →</Link>
          </Button>
        </div>
      )}
    </div>
  );
};

export default Training;
