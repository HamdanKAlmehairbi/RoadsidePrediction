import { Link } from "react-router-dom";
import { ArrowDown, Equal, ArrowUp, TrafficCone, Cpu, Brain, Share2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const stats = [
  { label: "Communication Reduction", value: "36.24%", color: "text-seal-teal", bg: "bg-seal-teal/10", icon: ArrowDown },
  { label: "Reward vs MARL", value: "−2.11%", color: "text-seal-amber", bg: "bg-seal-amber/10", icon: Equal },
  { label: "Travel Time Improvement", value: "18.14%", color: "text-seal-blue", bg: "bg-seal-blue/10", icon: ArrowUp },
  { label: "Active Intersections", value: "49", color: "text-seal-purple", bg: "bg-seal-purple/10", icon: TrafficCone },
];

const howItWorks = [
  { step: "1. Simulate", icon: Cpu, desc: "SUMO runs a grid city with real traffic patterns and signal controllers." },
  { step: "2. Learn", icon: Brain, desc: "Each traffic light trains its own PPO policy to minimise waiting time." },
  { step: "3. Federate", icon: Share2, desc: "Lights periodically share their model weights via an edge server." },
];

const Index = () => {
  return (
    <div className="max-w-6xl mx-auto space-y-10">
      {/* Hero */}
      <section className="text-center pt-8">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
          Smart Edge-Enabled Adaptive Traffic Lights
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          AI-controlled traffic lights that learn to reduce congestion — with 36% less communication than centralised methods.
        </p>
      </section>

      {/* Stat Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <Card key={s.label} className="border-border bg-card">
            <CardContent className="p-5 flex items-center gap-4">
              <div className={`p-2.5 rounded-lg ${s.bg}`}>
                <s.icon className={`h-5 w-5 ${s.color}`} />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">{s.label}</p>
                <p className={`text-2xl font-mono font-bold ${s.color}`}>{s.value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </section>

      {/* How It Works */}
      <section>
        <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {howItWorks.map((item) => (
            <Card key={item.step} className="border-border bg-card">
              <CardContent className="p-6">
                <div className="p-2 rounded-lg bg-seal-teal/10 w-fit mb-3">
                  <item.icon className="h-5 w-5 text-seal-teal" />
                </div>
                <h3 className="font-semibold mb-1">{item.step}</h3>
                <p className="text-sm text-muted-foreground">{item.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="flex flex-wrap gap-4 justify-center pb-8">
        <Button asChild size="lg" className="bg-seal-teal hover:bg-seal-teal/90 text-white">
          <Link to="/simulation">Watch a Simulation</Link>
        </Button>
        <Button asChild size="lg" className="bg-seal-blue hover:bg-seal-blue/90 text-white">
          <Link to="/training">Start Training</Link>
        </Button>
      </section>
    </div>
  );
};

export default Index;
