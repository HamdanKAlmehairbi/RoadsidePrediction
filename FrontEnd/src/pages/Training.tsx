import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from "recharts";
import { Link } from "react-router-dom";
import { useTrainStream } from "@/hooks/useTrainStream";
import { startTraining } from "@/lib/api";

const trainers = ["FedRL-naive", "FedRL-pos_reward", "MARL", "SARL"];
const topologyOptions = ["grid-3x3", "grid-5x5", "grid-7x7"];

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
  const [trainer, setTrainer] = useState("FedRL-naive");
  const [topology, setTopology] = useState("grid-3x3");
  const [ranked, setRanked] = useState(false);
  const [numEpisodes, setNumEpisodes] = useState([50]);
  const [showExisting, setShowExisting] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);

  const { episodes, isConnected, isDone } = useTrainStream(jobId);
  const isTraining = isConnected && !isDone;

  const handleStartTraining = async () => {
    try {
      let trainerType = trainer;
      let aggr = "naive";

      if (trainer.startsWith("FedRL")) {
        trainerType = "FedRL";
        aggr = trainer.replace("FedRL-", "");
      }

      const result = await startTraining({
        trainer: trainerType,
        aggr,
        topology,
        ranked,
        n_episodes: numEpisodes[0],
        fed_step: 1,
      });
      setJobId(result.job_id);
    } catch (err) {
      console.error("Failed to start training:", err);
    }
  };

  const liveChartData = useMemo(() => {
    return episodes.map(ep => ({
      episode: ep.episode,
      reward: ep.mean_reward,
    }));
  }, [episodes]);

  const fedRoundEpisodes = useMemo(() => {
    return episodes.filter(ep => ep.fed_round).map(ep => ep.episode);
  }, [episodes]);

  const latestEpisode = episodes.length > 0 ? episodes[episodes.length - 1] : null;
  const progress = latestEpisode ? (latestEpisode.episode / numEpisodes[0]) * 100 : 0;

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 bg-card border border-border rounded-lg p-3">
        <Select value={trainer} onValueChange={setTrainer}>
          <SelectTrigger className="w-44 bg-muted border-border"><SelectValue placeholder="Trainer" /></SelectTrigger>
          <SelectContent>{trainers.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
        </Select>
        <Select value={topology} onValueChange={setTopology}>
          <SelectTrigger className="w-28 bg-muted border-border"><SelectValue placeholder="Topology" /></SelectTrigger>
          <SelectContent>{topologyOptions.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent>
        </Select>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Ranked</span>
          <Switch checked={ranked} onCheckedChange={setRanked} />
        </div>
        <div className="flex items-center gap-2 w-40">
          <span className="text-xs text-muted-foreground whitespace-nowrap">Episodes: {numEpisodes[0]}</span>
          <Slider min={10} max={100} step={5} value={numEpisodes} onValueChange={setNumEpisodes} />
        </div>
        <Button
          className="bg-seal-blue hover:bg-seal-blue/90 text-white"
          onClick={handleStartTraining}
          disabled={isTraining}
        >
          {isTraining ? "Training..." : "Start Training"}
        </Button>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-xs text-muted-foreground">Show existing</span>
          <Switch checked={showExisting} onCheckedChange={setShowExisting} />
        </div>
      </div>

      {/* Existing Results Chart */}
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

      {/* Live Training Chart */}
      {liveChartData.length > 0 && (
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Live Training — {trainer}</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={liveChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(234, 14%, 20%)" />
                <XAxis dataKey="episode" tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} label={{ value: "Episode", position: "insideBottom", offset: -5, fill: "hsl(215, 16%, 57%)" }} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(215, 16%, 57%)" }} />
                <Tooltip contentStyle={{ background: "hsl(230, 18%, 13%)", border: "1px solid hsl(234, 14%, 20%)", borderRadius: 8 }} />
                {fedRoundEpisodes.map(ep => (
                  <ReferenceLine key={ep} x={ep} stroke="hsl(168, 76%, 40%)" strokeDasharray="4 4" opacity={0.4} />
                ))}
                <Line type="monotone" dataKey="reward" name={trainer} stroke="hsl(168, 76%, 40%)" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Status Bar */}
      {(isTraining || (episodes.length > 0 && !isDone)) && (
        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="h-3 w-3 rounded-full bg-seal-teal animate-pulse" />
            <span className="text-sm font-mono">
              Episode {latestEpisode?.episode ?? 0} / {numEpisodes[0]}
            </span>
            <span className="text-sm text-muted-foreground">
              Mean reward: <span className="font-mono">{latestEpisode?.mean_reward.toFixed(2) ?? "—"}</span>
            </span>
            <span className="text-sm text-muted-foreground">
              {trainer} {latestEpisode?.fed_round ? "· Fed round: yes" : ""}
            </span>
            <Progress value={progress} className="flex-1 h-2" />
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
      {isDone && (
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
