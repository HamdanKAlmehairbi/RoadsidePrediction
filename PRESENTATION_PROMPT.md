# Pencil.dev Prompt — 15-Minute Presentation

## PROMPT (paste this into Pencil.dev):

---

Create a **15-minute research presentation** (approximately 18-22 slides) with a modern, clean, professional design. Use a dark navy/teal theme with white text, subtle gradients, and accent colors (amber for highlights, green for good results, red for bad results). Use images only where they add meaning — context photos for problem/intro slides, network diagrams for topology slides. Do NOT add decorative images to data slides (charts, tables, methodology). Keep chart slides clean with the visualization taking full width. The tone should be **professional and clear** — accessible to a non-specialist audience but with academic confidence. Avoid informal or playful language (no pop culture references, no "rule them all" style phrasing). Strategy cards and content blocks should be compact — no excess whitespace below cards.

**Title:** "When Does Federation Help? Comparing 10 Training Strategies for Smart Traffic Lights"

**Authors:** Hamdan Almehairbi, Majid Ibrahim, Abdullah Alfalasi, Mohammed Almulla, Mohammed Albloushi

**Target venue:** Transportation Research Part C: Emerging Technologies (Impact Factor 7.9) / IEEE ITSC 2026

---

### SLIDE 1: Title Slide
**Title:** "When Does Federation Help?"
**Subtitle:** A Controlled Comparison of 10 Training Strategies for Multi-Intersection Traffic Signal Control
**Authors:** Hamdan Almehairbi, Majid Ibrahim, Abdullah Alfalasi, Mohammed Almulla, Mohammed Albloushi
**Visual:** Dramatic aerial photo of a complex urban intersection at night with traffic light trails. Modern, cinematic feel.

---

### SLIDE 2: The Problem
**Title:** "Traffic Congestion Costs $87 Billion Per Year in the US Alone"
**Content:**
- Traditional traffic lights run on fixed timers set decades ago
- They can't adapt to rush hour, accidents, or events
- AI can train traffic lights to respond in real-time
- But HOW should multiple intersections coordinate?

**Visual:** Split image — left side: congested highway/intersection with red taillights. Right side: flowing, green-lit smart intersection.

---

### SLIDE 3: The Research Question
**Title:** "If Every Intersection Has Its Own AI Brain... Should They Talk to Each Other?"
**Content:**
- Each intersection is an independent AI agent controlling its traffic light
- They can learn completely alone (independent)
- They can share everything (one brain for all)
- Or something in between — share with neighbors, share partially, share smartly
- **Our question: Which approach works best, and when?**

**Visual:** Illustration of a grid of intersections, some connected by glowing lines, some isolated. Like a neural network overlaid on a city map.

---

### SLIDE 4: Why This Is Hard
**Title:** "Why Nobody Has Answered This Fairly Before"
**Content:**
- Existing papers each use different simulators, different rewards, different algorithms
- Comparing Paper A's method (trained in CityFlow for 100 episodes) with Paper B's method (trained in SUMO for 500 episodes) is meaningless
- We built the **first standardized benchmark** where the ONLY variable is the training strategy
- Same simulator (SUMO), same reward, same algorithm (PPO), same everything

**Visual:** Messy comparison chart with crossed-out arrows (representing unfair comparisons) vs. a clean controlled experiment diagram.

---

### SLIDE 5: Our 10 Strategies — The Spectrum
**Title:** "10 Strategies: From Fully Independent to Fully Shared"
**Content:** Show this spectrum visually as a horizontal gradient bar:

```
Fully Independent ←————————————————————————→ Fully Shared
MARL   MeanField   CTDE   Gossip   HierFed   FedDistill   FedRL   SARL
```

Plus two non-AI baselines: Fixed-Time and Max-Pressure

**Visual:** Colorful spectrum/gradient bar with icons at each position. Left side is dark (isolated nodes), right side is bright (connected nodes).

---

### SLIDE 6: Strategy Explainer — Independent Strategies
**Title:** "Independent Strategies: No Weight Sharing"
**Content:**
- **MARL (Multi-Agent):** Each intersection trains its own policy independently. No communication between agents.
- **MeanField:** Independent policies, but each agent's observation includes the mean action of its neighbors from the previous step. Implicit coordination without communication.
- **CTDE (Centralized Training, Decentralized Execution):** During training, the value function sees all agents' observations for better credit assignment. At deployment, each agent acts on local observation only.

**Visual:** Three compact diagrams: (1) isolated nodes, (2) nodes with small arrows showing neighbor observation, (3) nodes connected to a shared critic during training, independent during execution.

---

### SLIDE 7: Strategy Explainer — Collaborative Strategies
**Title:** "Collaborative Strategies: Partial Sharing"
**Content:**
- **Gossip:** Each agent averages its policy weights with direct road-topology neighbors. Decentralized — no central server. Communication follows the physical road network.
- **HierFed (Hierarchical Federated):** Agents form spatially local clusters, average within each cluster, then cluster representatives coordinate globally. Two-tier aggregation respecting spatial locality.
- **FedDistill (Federated Distillation):** Agents share only action probabilities (logits), not full model weights. 100x less data transmitted, near-equivalent performance.

**Visual:** Three compact network diagrams: (1) mesh with bidirectional arrows between neighbors, (2) tree structure with clusters and a global coordinator, (3) small data packets (logits) flowing between nodes.

---

### SLIDE 8: Strategy Explainer — Centralized Strategies
**Title:** "Centralized Strategies: Full Sharing"
**Content:**
- **FedRL (Federated Averaging):** A central server collects all agent weights, computes a weighted average, and broadcasts the global model back. Classic star-topology federated learning.
- **SARL (Single-Agent):** All intersections share a single policy. One model maps observations to actions identically across the entire network. Maximum sharing, no per-intersection specialization.

**Visual:** (1) Star topology — central server connected to all nodes. (2) Single policy icon connected to all intersections.

---

### SLIDE 9: Our Test Networks
**Title:** "3 Network Topologies: Synthetic to Real-World"
**Content:**
- **Grid 3x3** (9 intersections): Controlled, symmetric. The "lab experiment."
- **Grid 5x5** (25 intersections): Bigger. Tests whether strategies scale.
- **Cologne-8** (8 intersections, REAL roads from Cologne, Germany): Irregular, asymmetric. The reality check.

**Visual:** Three side-by-side network visualizations — clean grid, larger grid, and the irregular Cologne network overlay on a satellite map of Cologne.

---

### SLIDE 10: Experimental Setup — Simulation
**Title:** "Simulation Environment"
**Content:** (Clean layout with labeled sections, no numbered list)

- **Simulator:** SUMO (Simulation of Urban Mobility) — open-source microscopic traffic simulator used across 500+ transportation research papers. Models individual vehicles, lane changes, signal phases, and queue dynamics at sub-second resolution.
- **Traffic Demand:** 360 vehicles per lane per hour (VPLPH). On grid-3x3 with 112 lanes, this generates ~40,000 vehicles per episode. Routes generated via SUMO's randomTrips with fringe-factor weighting (vehicles enter/exit at boundary nodes). Deterministic seeds for reproducibility.
- **Action Space:** Discrete — each intersection chooses {hold current phase, advance to next phase} at each step. Minimum green time: 4 steps. Maximum green time: 120 steps (forced switch if no action taken).
- **Observation Space:** 14 continuous features per intersection [0, 1]:
  - Lane occupancy (total vehicle length / total lane length)
  - Halted occupancy (stopped vehicle length / lane length, speed < 0.1 m/s)
  - Speed ratio (actual / limit)
  - 7 signal phase indicators
  - 4 network-wide occupancy rankings (local and global)

**Visual:** Diagram-only. Show a SUMO intersection with labeled observation features pointing to lanes, queues, and the signal head. No decorative image.

---

### SLIDE 10b: Experimental Setup — Training & Evaluation
**Title:** "Training & Evaluation Protocol"
**Content:**

- **Algorithm:** PPO (Proximal Policy Optimization) with identical hyperparameters for all 10 strategies — learning rate 0.001, discount factor 0.95, PyTorch backend, 4 rollout workers per job. No strategy gets a tuning advantage.
- **Reward Function:** r = -(lane_occupancy + halted_occupancy)^2. Quadratic penalty: small congestion gets a small penalty, heavy congestion gets a disproportionately large one. This incentivizes clearing the worst bottlenecks first.
- **Training Budget:** 50 episodes per strategy-topology pair. Same random seed (54321) for all training runs so demand patterns are identical.
- **Evaluation:** 10 independent Monte Carlo runs per configuration (seeds 42-51). Metrics extracted from SUMO's tripinfo output: average waiting time, average travel time, throughput (fraction of vehicles completing trips), and communication cost.
- **Statistical Rigor:** 95% confidence intervals via bootstrap resampling (10,000 resamples). All reported means include CI bounds.

**Visual:** Clean two-column layout or labeled diagram showing training pipeline (left) flowing into evaluation pipeline (right). No decorative image.

---

### SLIDE 11: Results — Grid 3x3
**Title:** "Small Network: All AI Crushes Traditional Methods"
**Content:** Bar chart showing average waiting time (seconds):

| Strategy | Wait Time |
|----------|:---------:|
| HierFed | 11.3s |
| Gossip | 11.4s |
| FedDistill | 11.8s |
| MeanField | 12.8s |
| MARL | 14.2s |
| SARL | 14.9s |
| FedRL | 15.6s |
| CTDE | 16.0s |
| fixed-time | 73.6s |
| max-pressure | 170.7s |

**Key callout:** "AI reduces wait times by 80-85% vs traditional fixed-time signals"

**Visual:** Horizontal bar chart. RL strategies in green/teal (short bars). Baselines in red (very long bars). Dramatic visual contrast.

---

### SLIDE 12: Results — Grid 5x5
**Title:** "Bigger Network: The Best Strategy Changes"
**Content:** Same bar chart format:

| Strategy | Wait Time |
|----------|:---------:|
| Gossip | 17.6s |
| SARL | 19.9s |
| HierFed | 21.0s |
| FedDistill | 21.1s |
| FedRL | 21.8s |
| CTDE | 21.9s |
| MeanField | 23.9s |
| MARL | 25.0s |
| fixed-time | 70.6s |
| max-pressure | 152.3s |

**Key callout:** "Gossip (neighbor-only sharing) now wins. At scale, who you share with matters more than how much."

**Visual:** Bar chart with Gossip highlighted in gold/amber. Arrow annotation showing "MARL dropped to last place among AI."

---

### SLIDE 13: Results — Cologne (Real World)
**Title:** "Real-World Networks Require Extended Training"
**Content:**

| Strategy | Wait Time |
|----------|:---------:|
| fixed-time | 44.4s |
| max-pressure | 51.1s |
| SARL | 54.7s |
| HierFed | 59.3s |
| Other RL | 63-72s |

- Real-world networks have irregular geometry, asymmetric intersections, and complex phase structures
- 50 training episodes is sufficient for synthetic grids but not for this level of complexity
- **Currently running 200-episode training on HPC** (4x baseline) to close this gap
- This is consistent with the literature — real-world deployment requires domain adaptation

**Visual:** Bar chart showing the results. No decorative image — let the data and the forward-looking message carry the slide.

---

### SLIDE 14: Cross-Topology Rankings
**Title:** "Who Wins Overall? The Robustness Ranking"
**Content:** Table showing rankings across all topologies:

| Strategy | 3x3 | 5x5 | Cologne | Avg Rank |
|----------|:---:|:---:|:-------:|:--------:|
| HierFed | 1st | 3rd | 4th | **2.7** |
| Gossip | 2nd | 1st | 7th | **3.3** |
| SARL | 6th | 2nd | 3rd | **3.7** |

**Key callout:** "HierFed is the most consistent performer. Gossip is the best on large networks."

**Visual:** Podium-style visualization with HierFed on gold, Gossip on silver, SARL on bronze. Or a heatmap table with color coding.

---

### SLIDE 15: The Key Insight
**Title:** "The Goldilocks Zone of Coordination"
**Content:**
- Too little sharing (MARL): Each intersection reinvents the wheel
- Too much sharing (FedRL, SARL): Loses local specialization
- **Just right: Share with your road neighbors (Gossip, HierFed)**
- The road topology itself is the optimal communication graph

**Visual:** A U-shaped curve or spectrum diagram showing performance vs. sharing level, with the optimal zone highlighted in the middle (Gossip/HierFed region).

---

### SLIDE 16: Communication Efficiency
**Title:** "You Don't Need to Share Everything"
**Content:**
- FedDistill shares only action decisions, not full brain weights
- 100x less data transmitted than FedRL
- Yet ranks 3rd on grid-3x3 and 4th on grid-5x5
- Implication: Bandwidth-constrained deployments (edge computing, cellular) can still coordinate effectively

**Visual:** Comparison graphic — giant file icon (FedRL: full weights) vs tiny envelope (FedDistill: just logits). Both achieving similar results.

---

### SLIDE 17: Ongoing HPC Campaign
**Title:** "What We're Running Right Now"
**Content:**
- **grid-7x7** (49 intersections): Does the ranking hold at even larger scale?
- **Cologne-8 extended** (200 episodes): Can AI catch up to baselines with more training?
- **Demand sensitivity** (150/360/600 vehicles): Which strategies handle rush hour?
- **Hyperparameter ablations**: How sensitive are results to aggregation frequency, cooperative reward, regularization?

**Running on 4 GPUs, ~12 hours total. Results incoming.**

**Visual:** HPC rack or GPU cluster photo. Progress indicators or a Gantt chart showing the 7 phases.

---

### SLIDE 18: Publication Target
**Title:** "Where This Research Is Going"
**Content:**
- **Primary target:** Transportation Research Part C: Emerging Technologies (Impact Factor 7.9)
  - Rolling submission, transportation-first journal that values benchmarking
- **Conference path:** IEEE ITSC 2026 (Intelligent Transportation Systems Conference)
  - Closest community match for multi-intersection signal control
- **Alternative:** IEEE T-ITS (Impact Factor 8.4), AAMAS 2027 (multi-agent systems)

**Visual:** Journal logos or academic venue badges. A roadmap arrow showing Conference -> Journal pipeline.

---

### SLIDE 19: Technical Contributions
**Title:** "What We Contribute"
**Content:**
1. **First 10-way controlled comparison** of RL training strategies for traffic signals
2. **Topology-aware coordination insight**: road network structure = optimal communication graph
3. **Communication efficiency finding**: logit distillation achieves near-best performance at 1% bandwidth
4. **Real-world gap quantification**: 50 episodes insufficient for cologne-8
5. **Open-source benchmark framework** with SUMO integration, HPC scripts, and reproducible evaluation

**Visual:** Five icons/badges representing each contribution, arranged in a clean grid.

---

### SLIDE 20: Future Directions
**Title:** "What's Next"
**Content:**
- Multi-simulator validation (CityFlow, VISSIM) for generalizability
- Transfer learning: train on grids, deploy on real networks
- Multi-algorithm comparison (DQN, A2C alongside PPO)
- Larger networks (16x16 grids, full Cologne network)
- Dynamic demand: time-of-day curriculum training
- Fairness metrics: do some intersections get worse so others improve?

**Visual:** Roadmap or branching tree diagram showing future research directions.

---

### SLIDE 21: Thank You / Q&A
**Title:** "Thank You"
**Content:**
- Authors: Hamdan Almehairbi, Majid Ibrahim, Abdullah Alfalasi, Mohammed Almulla, Mohammed Albloushi
- GitHub: github.com/HamdanKAlmehairbi/RoadsidePrediction
- Framework: ATLAS (Adaptive Traffic Light Assessment of Strategies)
- Built with: SUMO + Ray RLlib + PyTorch + FastAPI

**Visual:** Clean closing slide with project logo/icon, QR code to GitHub, and contact info. Show all author names.

---

## DESIGN NOTES FOR PENCIL:
- Use a consistent dark navy (#0a1628) or dark teal (#0d2137) background throughout
- Accent color: amber/gold (#f59e0b) for highlights and key numbers
- Good results in green (#10b981), bad results in red (#ef4444)
- Use large, bold numbers for key statistics (e.g., "85% reduction", "$87B")
- Every slide MUST have a relevant high-quality image or diagram
- Use icons (network nodes, traffic lights, brains, arrows) to make strategies visual
- Transitions should feel smooth and modern, not flashy
- Charts should be clean with minimal gridlines — let the data speak
- Speaker notes are provided below as a separate script

---

# SPEAKER SCRIPT (15 minutes)

## Slide 1 — Title (30 seconds)
"Good morning/afternoon. My name is Hamdan, and today I'm going to talk about a question that seems simple but turns out to be surprisingly deep: when you train AI to control traffic lights across a city, how should those AI agents coordinate with each other?"

## Slide 2 — The Problem (45 seconds)
"Let's start with why this matters. Traffic congestion costs the US alone $87 billion every year. That's wasted fuel, wasted time, and increased emissions. Most traffic lights in cities today still run on fixed timers — schedules that were set years or even decades ago. They can't react to a sudden traffic jam, a closed lane, or a football game letting out. Reinforcement learning — the same technology behind AlphaGo and ChatGPT — can train traffic lights to adapt in real time. And it works. But here's the catch..."

## Slide 3 — The Research Question (45 seconds)
"If you have a city with 25 intersections, each one controlled by its own AI agent, a natural question arises: should these agents talk to each other? If yes, how? Should every intersection share everything with a central server? Should they only whisper to their immediate neighbors on the road? Should they share their full knowledge or just their final decisions? There's a whole spectrum of options, and the existing research has no clear answer because every paper tests their method differently."

## Slide 4 — Why This Is Hard (45 seconds)
"That's the gap we address. Previous work is essentially incomparable. Paper A uses SUMO with a 3x3 grid. Paper B uses CityFlow with a single intersection. Paper C uses a different reward function entirely. Comparing them is like comparing a runner's marathon time to a cyclist's sprint — different sports. We built the first framework where we test 10 different strategies under absolutely identical conditions. Same simulator, same roads, same traffic, same AI algorithm. The ONLY thing that changes is how the agents coordinate."

## Slide 5 — The Spectrum (30 seconds)
"Here's our lineup. Ten strategies arranged from fully independent on the left — where each intersection learns completely alone — to fully shared on the right — where all intersections share one brain. Plus two traditional baselines that don't use AI at all."

## Slide 6 — Independent Strategies (1 minute)
"Let me walk through the strategies. On the independent end: MARL gives each intersection its own policy — no communication at all. MeanField is similar but augments each agent's observation with the mean action of its neighbors, so there's implicit coordination without any explicit data sharing. CTDE takes a different approach — during training, the value function has access to global state across all intersections for better credit assignment. But at deployment, each agent acts only on its local observation."

## Slide 7 — Collaborative Strategies (1 minute)
"The middle of the spectrum is where things get interesting. Gossip is elegantly simple: each intersection periodically averages its policy weights with its direct road neighbors. No central server — communication follows the physical road topology. HierFed adds structure: agents form spatially local clusters, aggregate within each cluster first, then cluster representatives coordinate globally. And FedDistill takes a bandwidth-efficient approach — instead of sharing full model weights, agents share only their action probabilities. A hundred times less data transmitted, with comparable performance."

## Slide 8 — Centralized Strategies (30 seconds)
"On the fully shared end: FedRL uses a central server that collects all agent weights, computes a weighted average, and broadcasts the global model back — standard federated averaging. And SARL is the simplest approach — a single shared policy for every intersection. Maximum coordination, but no per-intersection specialization."

## Slide 9 — Test Networks (30 seconds)
"We tested on three networks. A controlled 3x3 grid — our lab experiment with 9 intersections. A larger 5x5 grid with 25 intersections to test scalability. And Cologne-8 — eight real-world intersections from the city of Cologne, Germany, with irregular roads and realistic configurations."

## Slide 10 — Simulation Environment (45 seconds)
"Let me walk through the experimental setup. We use SUMO — the most widely-used microscopic traffic simulator in the field, with over 500 publications. It models individual vehicles, lane changes, and signal phase transitions at sub-second resolution. Each intersection observes 14 features: how full its lanes are, how many vehicles are stopped, current speed ratios, signal phase state, and where it ranks in the network by congestion. The action is simple — hold the current signal phase or advance to the next one. Traffic demand is 360 vehicles per lane per hour, which on a 3x3 grid with 112 lanes means roughly 40,000 vehicles per episode."

## Slide 10b — Training & Evaluation Protocol (45 seconds)
"For training, every strategy uses the exact same PPO algorithm with identical hyperparameters — learning rate point-zero-zero-one, gamma point-nine-five, PyTorch, four workers. No strategy gets a tuning advantage. The reward function penalizes congestion quadratically: a little congestion gets a small penalty, heavy congestion gets a disproportionately large one. This pushes agents to fix the worst bottlenecks first. We train each configuration for 50 episodes with the same random seed, then evaluate with 10 independent Monte Carlo runs using different seeds. All metrics — waiting time, travel time, throughput — come directly from SUMO's tripinfo output, with 95% confidence intervals from bootstrap resampling."

## Slide 11 — Results Grid 3x3 (45 seconds)
"Here are the results, and they're striking. On the small grid, every AI strategy crushes the traditional methods. The best AI — HierFed — achieves an average waiting time of 11 seconds per vehicle. Fixed-time signals? 74 seconds. Max-pressure? 171 seconds. That's an 85% reduction in waiting time. But notice something interesting — the top performers aren't the extremes. It's not the fully independent MARL or the fully centralized FedRL. It's the middle-ground strategies: HierFed, Gossip, FedDistill."

## Slide 12 — Results Grid 5x5 (45 seconds)
"Now things get really interesting at scale. On the 5x5 grid with 25 intersections, the ranking shifts. Gossip — the strategy where intersections only talk to their direct road neighbors — takes first place. And MARL, the fully independent approach, drops to dead last among AI methods. The message is clear: at scale, some form of coordination is essential. But here's the insight — it's not about coordinating with everyone. It's about coordinating with the RIGHT neighbors. Gossip and HierFed respect the road topology. FedRL ignores it."

## Slide 13 — Results Cologne (45 seconds)
"On the real-world Cologne network, we see a different picture. The RL strategies don't yet outperform the baselines at 50 training episodes. This is expected — real-world networks have irregular geometry, asymmetric intersections, and complex phase structures that require significantly more training to learn. This is consistent with the broader literature on sim-to-real transfer. We're currently running extended training at 200 episodes on HPC to close this gap, and early indicators are promising."

## Slide 14 — Cross-Topology Rankings (30 seconds)
"Looking across all three topologies, the most robust strategy overall is HierFed with an average rank of 2.7 — consistently good everywhere. Gossip is second at 3.3 — the best on large networks but weaker on real-world ones. SARL rounds out the top 3."

## Slide 15 — Key Insight (45 seconds)
"So what's the takeaway? There's an optimal level of coordination. Too little sharing — as in MARL — and each intersection wastes training time relearning what its neighbors already know. Too much sharing — as in FedRL with its central server — and you lose local specialization while ignoring the spatial structure of the network. The optimal point is topology-aware coordination: share with the intersections you're physically connected to. The road network itself turns out to be the best communication graph for learning."

## Slide 16 — Communication Efficiency (30 seconds)
"One more finding worth highlighting: you don't need to share everything. FedDistill transmits 100 times less data than full weight sharing, yet still ranks 3rd and 4th. For real-world deployments on cellular networks or edge devices with limited bandwidth, this is very attractive."

## Slide 17 — Ongoing HPC Campaign (30 seconds)
"We're currently running an expanded campaign on HPC with 4 GPUs. Testing a 49-intersection grid to see if the rankings hold at even larger scale. Extended training on Cologne to close the real-world gap. And ablation studies on demand levels, aggregation frequency, and cooperative reward to understand sensitivity."

## Slide 18 — Publication Target (20 seconds)
"We're targeting Transportation Research Part C, a top-tier transportation journal with an impact factor of 7.9, and IEEE ITSC as a conference venue. The work fits naturally at the intersection of transportation engineering and multi-agent AI."

## Slide 19 — Contributions (30 seconds)
"To summarize our contributions: the first controlled 10-way comparison of training strategies for traffic signals. The insight that road topology equals optimal communication topology. Evidence that logit distillation achieves near-best performance at 1% of the bandwidth. Quantification of the real-world gap. And ATLAS — an open-source benchmark framework for reproducible evaluation."

## Slide 20 — Future Directions (20 seconds)
"Looking ahead, we plan to validate across multiple simulators, test transfer learning from synthetic to real networks, expand to larger city-scale networks, and investigate fairness — whether coordination helps some intersections at the expense of others."

## Slide 21 — Thank You (10 seconds)
"The ATLAS framework is open-source on GitHub. I'm happy to take questions. Thank you."

---

**Total speaking time: ~14 minutes, leaving 1 minute buffer for transitions and pauses.**
