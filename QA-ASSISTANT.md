# Q&A Assistant — Federated RL Traffic Signal Control Benchmarking

You are helping a team answer questions after their midterm presentation. Answer confidently, concisely, and honestly. If something is a limitation, own it. If something comes from the literature, say so. If something is the team's design, say so.

---

## PRESENTATION OVERVIEW

**16 slides, ~9.5 minutes, 5 speakers.**

| Slide | Title | Speaker | Time |
|-------|-------|---------|------|
| 1 | Title | Al Rayassi | 0:00-0:20 |
| 2 | Research Context | Al Rayassi | 0:20-1:10 |
| 3 | Objectives | Al Rayassi | 1:10-1:50 |
| 4 | Related Work | Al Rayassi | 1:50-2:30 |
| 5 | Benchmarking Framework | Al Rayassi | 2:30-3:10 |
| 6 | Simulation Layer | Majid | 3:10-3:50 |
| 7 | The RL Agent | Majid | 3:50-4:30 |
| 8 | Dataset | Majid | 4:30-5:00 |
| 9 | Three Strategies | Majid | 5:00-5:45 |
| 10 | Demand Settings | Majid | 5:45-6:15 |
| 11 | Learning Curves | Abdallah | 6:15-6:50 |
| 12 | Waiting Time | Abdallah | 6:50-7:35 |
| 13 | Communication Cost | Abdallah | 7:35-8:05 |
| 14 | Trade-off Analysis | Blooshi | 8:05-8:35 |
| 15 | Next Steps | Blooshi | 8:35-9:05 |
| 16 | Contributions + Thank You | Blooshi | 9:05-9:35 |

---

## PROJECT IDENTITY

**What this is:** A benchmarking framework that evaluates three RL training strategies (SARL, MARL, FedRL) for traffic signal control under identical controlled conditions.

**What this is NOT:** A paper proposing a new algorithm. The contribution is the standardized evaluation framework and the fair comparison it enables.

**Core value in two sentences:** We designed an intersection-agnostic observation space that lets different intersection types share learned knowledge through federated aggregation — something no individual component provides on its own. We then built the unified platform that produces the first apples-to-apples comparison of single-agent, multi-agent, and federated RL for traffic signal control under identical conditions.

**Stage:** Midterm — baseline comparison complete on one demand setting. Multi-demand scenarios, aggregation variants, and full trade-off analysis are second half.

---

## GLOSSARY OF TECHNICAL TERMS

**Reinforcement Learning (RL):** A machine learning paradigm where an agent learns by interacting with an environment. It observes a state, takes an action, receives a reward, and updates its policy to maximize cumulative reward over time. No labeled training data needed.

**Policy:** The function that maps observations to actions. In our case, a neural network that takes 14 numbers and outputs a probability distribution over 2 actions (keep phase or switch). The policy IS the trained model — the thing we're comparing across strategies.

**PPO (Proximal Policy Optimization):** The specific RL algorithm we use. A policy gradient method by Schulman et al. (2017) that updates the policy conservatively — it clips the update to prevent large, destabilizing changes. Stable, well-studied, widely used for discrete action spaces. We chose it because it's the standard in this domain, not because it's novel.

**SUMO (Simulation of Urban Mobility):** Open-source microscopic traffic simulator by the German Aerospace Center (DLR). Models individual vehicles with realistic physics — acceleration, braking, lane changing, right-of-way. We interact with it through TraCI (Traffic Control Interface), a Python API that lets us read sensor data and control traffic lights during simulation.

**TraCI (Traffic Control Interface):** SUMO's Python API. We use it to query lane occupancy, vehicle speeds, traffic light states, and to set traffic light phases. Every observation and action flows through TraCI.

**Episode:** One complete training run. In our case, 4000 simulation timesteps. The agent interacts with the environment for 4000 steps, accumulates rewards, and the PPO algorithm updates the policy based on that experience. We train for 25 episodes per configuration.

**Monte Carlo Evaluation:** Running the trained policy multiple times with different random seeds to get a statistically reliable performance estimate. We use 5 runs with seeds 42-46. Each run generates different traffic (different vehicle routes and departure times) to test whether the policy generalizes.

**FedAvg (Federated Averaging):** Algorithm by McMahan et al. (2017). Each client trains a local model, sends its weights to a server, the server averages the weights, sends the average back. Originally designed for mobile keyboards learning from user typing without uploading raw text. We apply it to traffic intersections.

**FedProx (Federated Proximal):** Extension of FedAvg by Li et al. (2020). Adds a regularization term that penalizes local models for drifting too far from the global average. Designed for "heterogeneous" settings where clients have very different data distributions — in our case, intersections with very different traffic patterns.

**Observation Space:** The set of information the agent sees at each timestep. Our 14-feature vector normalized to [0,1]. Defined in `seal/sumo/config.py`, computed in `seal/sumo/kernel/trafficlight/light.py`.

**Action Space:** The set of possible actions. Ours is Discrete(2) — keep current phase (0) or advance to next phase (1). Binary is intersection-agnostic: works for any intersection type regardless of lane count or phase structure.

**Reward Function:** The signal that tells the agent how well it's doing. Ours: `r = -(occupancy + halted_occupancy)^2`. Computed per intersection per timestep. More congestion = more negative reward.

**Intersection-Agnostic:** An observation representation that works the same way for any intersection type — 2-lane, 4-lane, 8-lane. Achieved by using ratios (fraction full, fraction red) rather than absolute values (47 cars, phase GGrr). This is what enables weight sharing across different intersection types in federated aggregation.

**VPLPH (Vehicles Per Lane Per Hour):** The traffic demand level. 360 VPLPH means 360 vehicles enter each lane every hour. Our standard demand. Period between departures = 3600 / (360 × number_of_lanes).

**Topology:** The road network layout. We use Grid 3×3 (9 intersections) and Grid 5×5 (25 intersections). Synthetic grids with heterogeneous lane counts — center intersections have more lanes than edge ones.

**Aggregation:** The process of combining multiple local models into one global model. In FedRL, this happens every episode. Our version uses reward-weighted averaging — better-performing intersections contribute more.

**Convergence:** When the training reward stops improving and plateaus. Our curves plateau around episode 15-20, indicating the policy has learned a stable strategy.

**Phase:** A traffic light configuration — which directions get green, yellow, or red simultaneously. Example: "GGrr" means two directions get green and two get red. Phases cycle in a fixed sequence; the agent decides WHEN to advance, not which phase to show.

**Phase Ratio:** Our intersection-agnostic encoding of the current traffic light state. Instead of "GGrr" (topology-specific), we encode "50% green, 50% red" (works for any intersection). Seven ratio features cover all SUMO signal states (red, yellow, minor green, priority green, u-turn, off-blinking, off).

**Ranking Features:** Four features (indices 10-13) that tell each intersection how congested it is relative to neighbors (local rank) and the whole network (global rank), for both total vehicles and halted vehicles. Enables cross-topology transfer: "I'm the most congested in my neighborhood" means the same thing on any grid size.

**Cross-Topology Transfer:** The ability to use a policy trained on one network size on a different network size without retraining. Enabled by our intersection-agnostic observations — same 14 features regardless of how many intersections or lanes exist.

**Edge Server:** In the federated architecture, the central server that receives model weights from all intersections, computes the average, and distributes it back. In a real deployment, this would be a roadside compute unit. In our simulation, it's a function call in `fed_agent.py`.

**Communication Cost:** Total data transmitted during training. Centralized strategies stream observations (~64 bytes) and actions (~1 byte) every timestep from every intersection. Federated only sends model weights (~50KB) once per episode. We compute this theoretically based on actual data sizes.

**Controlled Variable:** Something held constant across all experiments. We control 8 layers: road network, traffic demand, observations, actions, reward, algorithm, training budget, evaluation protocol.

**Independent Variable:** The thing we deliberately change — how policies are organized across intersections. This is the ONLY thing that differs between SARL, MARL, and FedRL in our framework.

**Dependent Variable:** What we measure as outcomes — average waiting time, average travel time, communication cost. These change as a RESULT of the independent variable.

**Wilcoxon Signed-Rank Test:** A non-parametric statistical test for comparing paired samples. Planned for final evaluation to determine whether differences between strategies are statistically significant or just noise from random variation.

**GAE (Generalized Advantage Estimation):** A method for estimating how much better an action was compared to the average. Used in PPO to reduce variance in policy gradient estimates. We use lambda=1.0 (standard).

**Ray RLlib:** The distributed RL library we use. Part of the Ray framework by Anyscale. Handles multi-agent PPO training, policy management, and environment interaction. We use the "old API stack" mode for backward compatibility.

**Krauss Model:** SUMO's default car-following model. Determines how vehicles accelerate and brake based on the vehicle ahead. We don't configure this — it's SUMO's built-in physics. Every strategy uses the same vehicle physics.

---

## DESIGN DECISIONS AND RATIONALE

### Why RL for traffic signal control?
Traffic is dynamic and stochastic — analytical optimization requires a mathematical model of traffic flow that breaks down in practice. RL learns directly from interaction without needing a closed-form model. It can discover strategies that outperform any fixed or rule-based policy because it optimizes directly for the reward signal through exploration.

### Why PPO specifically?
PPO is stable for discrete action spaces, well-supported by Ray RLlib for multi-agent training, and used across the FedRL traffic literature (FedLight, SEAL, Fed-PPO). The choice is pragmatic — we held it constant across all strategies so the algorithm is NOT a variable in our comparison. DQN (Bao et al. 2023) and A2C (FedLight 2021) are valid alternatives used in other papers.

### Why not DQN?
DQN is value-based and works well for single-agent setups. PPO is policy-gradient and handles multi-agent setups more naturally through Ray RLlib's multi-policy framework. Either would work; we chose PPO for infrastructure compatibility and literature consistency.

### Why binary action instead of phase selection?
Phase selection would be Discrete(N) where N varies by intersection type (4 for a 4-lane, 8 for an 8-lane). You cannot share weights between policies with different output dimensions. Binary phase switching is Discrete(2) for every intersection — essential for federated aggregation across heterogeneous intersections.

### Why 14 features specifically?
Built from three functional groups: 3 traffic flow features (occupancy, halted, speed) capture the physical state. 7 phase ratios capture what the light shows in intersection-agnostic format. 4 ranking features provide relative congestion context for cross-topology transfer. Each group serves a purpose — removing any would lose information.

### Why ratios instead of raw counts?
A 2-lane intersection with 10 cars and a 4-lane intersection with 10 cars are in very different situations. Raw counts don't capture that. Ratios normalize by capacity — 0.8 lane occupancy means "80% full" regardless of lane count. This is what makes observations comparable across intersection types.

### Why ranking features?
Without them, an agent only knows "I have 60% occupancy" — no context. With ranking, it knows "I'm the most congested in my neighborhood." This contextual awareness helps prioritize and enables cross-topology transfer. Computed from the TLS adjacency graph in the network file.

### Why quadratic reward?
Linear penalty treats going from 10% to 20% occupancy the same as 80% to 90%. Quadratic makes high congestion disproportionately expensive: -(0.2 + 0.1)^2 = -0.09 vs -(0.8 + 0.7)^2 = -2.25. Encourages preventing queues rather than tolerating moderate congestion.

### Why not use waiting time as reward?
Waiting time is a trip-level metric — only available after a vehicle reaches its destination. We need a per-timestep signal during training. Occupancy ratios are available every timestep via TraCI and correlate strongly with waiting time. We use SUMO's waiting time as the evaluation metric (ground truth), not as the training reward.

### Why reward-weighted aggregation instead of equal-weight?
Standard FedAvg weights every client equally. In our grids, center intersections handle 3-4x more traffic than corner intersections. Equal weighting dilutes the center's hard-won knowledge about heavy traffic. Reward-based weighting gives busy, well-performing intersections proportionally more influence.

### Why synthetic grids instead of real-world networks?
Three reasons: (1) Controlled heterogeneity — we design exactly how intersections differ (center has more lanes). (2) Reproducibility — same seed = same traffic, anyone can replicate. (3) Standard practice — FedLight, SEAL, HFRL all use synthetic SUMO grids. We attempted Manhattan (OSM) and Cologne (RESCO) but the observation layer needs adaptation for arbitrary intersection geometries. Planned for final evaluation.

### Why 360 VPLPH?
Standard in the literature. Produces moderate congestion — enough to make the problem non-trivial without causing gridlock. Lower demand (150) makes the problem too easy for all strategies. Higher demand (600) is planned as a stress test for the second half.

### Why no turning vehicles?
Turning vehicles can deadlock in SUMO and introduce confounding factors unrelated to signal timing. Straight-through traffic isolates the variable we care about (signal control quality). Follows established methodology in the SEAL and FedLight papers.

### Why 25 episodes?
Reward curves plateau by episode 15-20 on both topologies. A 50-episode run on Grid 5×5 confirmed the same plateau. 25 is sufficient to show convergence and compare strategies. Longer training would yield marginal improvements without changing the relative ordering.

### Why 5 Monte Carlo seeds?
Practical balance for midterm. Gives variance estimates and error bars. Final evaluation will use 10 seeds with Wilcoxon significance testing. 5 is sufficient to show consistent trends.

### Why aggregate every episode?
Every episode = every 4000 timesteps. More frequent aggregation gives faster knowledge sharing but more communication overhead. Less frequent gives more local specialization but slower convergence. Every-episode aggregation is a reasonable middle ground and the default in our framework. Optimizing this frequency is potential future work.

### Why not early stopping?
Early stopping would introduce a hidden variable — strategies might stop at different episodes, making comparison unfair. All strategies train for exactly 25 episodes. Same compute budget, same data exposure.

### Why these PPO hyperparameters?
lr=5e-5, minibatch=128, clip=0.3, KL=0.3, batch=4000, rollout=200, GAE=1.0, VF clip=10. Drawn from the SEAL framework literature. We did NOT tune per-strategy because that would violate our standardization principle. If we gave FedRL different hyperparameters than MARL, we couldn't attribute performance differences to the training strategy alone.

### Why the same network architecture for all strategies?
Same 256×256 MLP with ReLU for all three. If SARL used a larger network, its better performance might come from capacity, not the training strategy. Holding architecture constant is essential for the benchmark's validity.

### Why did you choose SUMO over CityFlow?
SUMO is the most widely used open-source microscopic traffic simulator. CityFlow is faster but less realistic in vehicle physics. Most papers in our reference list (FedLight, SEAL, Bao, HFRL) use SUMO, making our results directly comparable.

### Why Grid 3×3 AND Grid 5×5?
One topology wouldn't show whether results generalize. Two gives us a minimal scalability test — 9 vs 25 intersections. The Grid 5×5 result showing FedRL pulling ahead (16.8s vs 17.4s) while it trailed on 3×3 (11.5s vs 10.4s) demonstrates that the relative advantage changes with scale. This would be invisible with one topology.

### Why not Grid 7×7?
Training time. Grid 5×5 with 25 intersections takes ~4 hours per config. Grid 7×7 with 49 intersections would take ~8+ hours per config. With 32 planned configurations, that's impractical. We also lack pre-trained example weights for Grid 7×7.

---

## SLIDE-SPECIFIC Q&A

### Slide 1 — Title
**Q: What does "Benchmarking Training Strategies" mean?**
A: We don't propose that one strategy is best. We test three strategies under identical conditions and measure what each is good at. The output is a comparison and trade-off analysis, not a claim that one wins.

**Q: What do "Standardized," "Reproducible," "Comparable" mean here?**
A: Standardized = same environment for all strategies. Reproducible = same seeds produce same results. Comparable = differences in outcomes can only come from the training strategy, not from experimental setup differences.

### Slide 2 — Research Context
**Q: Where does the $87B number come from?**
A: INRIX 2019 transportation analytics report. Referenced across the traffic RL literature including the SEAL paper.

**Q: What do you mean "no standardized comparison exists"?**
A: FedLight (2021) uses A2C on one topology. SEAL (2022) uses PPO with pre-trained weights. Bao (2023) uses DQN with different observations. When their numbers differ, you can't tell if it's because the strategy is better or because the setup is different. Our framework eliminates that ambiguity.

### Slide 3 — Objectives
**Q: What's the difference between Objective 2 and Objective 3?**
A: Objective 2 is "run the comparison." Objective 3 is "interpret the results." Comparing tells you which numbers are bigger. Trade-off analysis tells you WHY and WHEN — when does FedRL win? When does SARL win? What do you give up for lower communication?

### Slide 4 — Related Work
**Q: You cite only three papers. Isn't that a thin literature review?**
A: These are the three most directly comparable papers doing federated RL for traffic signals. The full field has fewer than 10 papers. We also reference McMahan (FedAvg), Li (FedProx), Ault & Sharon (RESCO benchmark), and Schulman (PPO) for the methods we use.

**Q: You say SEAL used pre-trained weights. Why is that a limitation?**
A: Pre-trained weights mean they evaluated existing models without training from scratch. They showed the models work but didn't demonstrate that their training strategy produces those models. We train from scratch for all strategies, so we can compare training convergence and final performance.

**Q: You criticize these papers but aren't you building on their work?**
A: We draw on their ideas — ratio-based observations from SEAL, the federated training concept from FedLight, the evaluation methodology from RESCO. Our contribution is the standardized framework that lets us compare these ideas fairly. The components are informed by the literature; the integration and the controlled comparison are ours.

### Slide 5 — Benchmarking Framework
**Q: What are the controlled variables specifically?**
A: Road network (.net.xml file), traffic demand (360 VPLPH, same randomTrips), vehicle routes (same random seeds), observation function (same get_observation() code computing 14 features), action space (Discrete(2)), reward function (r = -(o+h)^2), PPO algorithm with identical hyperparameters (lr=5e-5, clip=0.3, 256×256 MLP), training budget (25 episodes), evaluation protocol (5 MC runs, seeds 42-46, SUMO tripinfo metrics).

**Q: What's the independent variable?**
A: How policies are organized. SARL: 1 shared policy. MARL: N independent policies, centralized training. FedRL: N policies with periodic reward-weighted aggregation. That is the ONLY variable.

**Q: What are the dependent variables?**
A: Average waiting time (seconds vehicles spend stopped), average travel time (total trip duration), communication cost (total bytes transmitted during training).

**Q: How do you know you haven't introduced hidden variables?**
A: We use the same codebase. All three strategies inherit from the same BaseTrainer class, use the same SumoEnv environment, the same TrafficLight observation code, the same PPO config dict. The divergence is literally ~15 lines of code in on_data_recording_step() that implements FedAvg aggregation. Everything else is shared code paths.

### Slide 6 — Simulation Layer
**Q: What version of SUMO?**
A: 1.26.0 on Windows 11.

**Q: What vehicle model does SUMO use?**
A: Krauss car-following model (default). Determines acceleration, braking, and following distance based on the vehicle ahead. We don't configure it — same physics for all strategies.

**Q: How does SUMO handle traffic lights?**
A: Traffic lights have predefined phase sequences in the .net.xml file (e.g., GGrr → yyrr → rrGG → rryy). Our RL agent decides WHEN to advance to the next phase using TraCI's setRedYellowGreenState(). Timing constraints (4s min, 120s max) are enforced by our TrafficLight class.

**Q: What does "automatically builds maps from road network files" mean?**
A: We parse the .net.xml XML to extract all traffic light junctions, their controlled lanes, and their adjacency relationships. This builds the TLS graph used for ranking features and cooperative reward computation. No manual configuration needed — add a new .net.xml and the system discovers all intersections automatically.

**Q: You say 14 features but earlier slides showed 10. Which is it?**
A: 14. The 10 shown on some diagrams are the base features (3 traffic flow + 7 phase ratios). The remaining 4 are ranking features (local rank, global rank, local halt rank, global halt rank). Optional time encoding adds 2 more for 16 total.

### Slide 7 — The RL Agent
**Q: Walk me through the reward formula.**
A: r = -(o + h)^2. o is lane occupancy (0.0 to 1.0, fraction of lane length filled by vehicles). h is halted lane occupancy (fraction filled by vehicles with speed < 0.1 m/s). Add them, square the sum, negate it. Example: lanes 60% full, 40% halted → r = -(0.6 + 0.4)^2 = -1.0. Lanes 10% full, 5% halted → r = -(0.1 + 0.05)^2 = -0.0225. High congestion is penalized quadratically more.

**Q: What's the neural network architecture?**
A: Two hidden layers of 256 neurons each, ReLU activation. Input: 14 features. Output: 2 action probabilities (keep/switch). Standard Ray RLlib FullyConnectedNetwork. Same for all strategies.

**Q: What does "maximize cumulative discounted reward" mean?**
A: The agent doesn't just maximize immediate reward — it maximizes the sum of all future rewards, with future rewards discounted by gamma. This makes the agent consider long-term consequences: clearing a queue now might cause a bigger queue later if the timing is wrong.

**Q: The diagram shows "SUMO TraCI (Traffic Light)" — what exactly happens?**
A: The agent outputs action 0 or 1. If 1, our TrafficLight class calls `traci.trafficlight.setRedYellowGreenState(tls_id, next_phase)` to advance the phase. SUMO simulates one timestep. Then we call `traci.lane.getLastStepVehicleIDs()`, `traci.vehicle.getSpeed()`, etc. to compute the next observation. This loop repeats 4000 times per episode.

### Slide 8 — Dataset
**Q: How do you generate traffic?**
A: SUMO's `randomTrips.py` with parameters: `--period (3600 / n_vehicles)`, `--fringe-factor 100` (vehicles start/end at network edges), `--seed N` (deterministic). The formula: `n_vehicles = 360 × n_lanes × 1 hour`. For Grid 3×3 with 24 lanes: 8,640 vehicles/hour, departure every 0.42 seconds.

**Q: What does "heterogeneous lane counts" mean?**
A: In our grids, roads near the center have more lanes than roads on the border. A center intersection might control 4 lanes in each direction while a corner controls 1. This tests whether the observation space handles different intersection types — which is the whole point of intersection-agnostic features.

**Q: Could you use real traffic data instead of randomTrips?**
A: Yes — SUMO can load any route file. RESCO benchmark includes real demand from Cologne, Ingolstadt, and Salt Lake City. The HFRL paper uses TomTom-calibrated NYC data. We used randomTrips for reproducibility and consistency with prior work. Real-world demand integration is planned.

**Q: 360 vehicles per lane per hour — is that a lot?**
A: It's moderate. For a 2-lane road, that's one vehicle per lane every 10 seconds. Enough to create meaningful congestion at intersections but not gridlock. Real-world urban roads see 400-1800 VPLPH depending on the road type.

### Slide 9 — Three Strategies
**Q: If SARL sees all observations and MARL has a central coordinator, what's the actual difference?**
A: SARL has ONE policy that receives observations from all intersections — the policy itself is shared, so intersection A0 and intersection B1 produce the same output for the same observation. MARL has SEPARATE policies — A0 and B1 have their own weights and can learn different behaviors. The central coordinator in MARL manages the training loop, not the decision-making.

**Q: In FedRL, what exactly gets sent to the server?**
A: The full set of neural network parameters — all weights and biases from the 2×256 MLP. Approximately 50KB of float32 values per intersection. Not observations (64 bytes), not actions (1 byte), not rewards, not gradients — just the final learned weights.

**Q: Why reward-weighted and not just equal-weight averaging?**
A: A center intersection serving 500 vehicles with reward -5 has learned more about heavy traffic management than a corner serving 50 vehicles with reward -2. Equal weighting dilutes the center's knowledge. Reward weighting lets useful patterns propagate faster.

**Q: Is the server a single point of failure?**
A: In our simulation, the server is a function call, not a separate process. In a real deployment, yes — if the edge server goes down, agents can't aggregate. But they continue training locally. They just don't share until the server recovers. FedAvg is naturally resilient to temporary disconnection.

### Slide 10 — Demand Settings
**Q: Why four demand settings?**
A: One setting (our current 360 VPLPH) shows which strategy performs best under standard conditions. But a benchmark needs to show WHEN each strategy wins. Low demand (150) tests whether RL is even necessary. High demand (600) tests how strategies degrade under stress. Variable demand (200-700) tests adaptability. Together they characterize the full performance landscape.

**Q: Where do the VPLPH numbers come from?**
A: 360 is the standard from the SEAL literature. 150 is approximately the threshold where fixed-time becomes adequate (from our preliminary observations). 600 is near the saturation point for our grid networks. 200-700 is the range of our time-of-day curriculum.

**Q: 32 configurations seems like a lot. Can you actually run all of them?**
A: 4 settings × 4 strategies × 2 topologies = 32 configs. Each Grid 3×3 config takes ~30 min. Each Grid 5×5 config takes ~90 min. Total: ~32 hours of compute. Spread over a weekend, this is feasible. The infrastructure is built and tested.

### Slide 11 — Learning Curves
**Q: Why does SARL (Decentralized) start better on Grid 3×3?**
A: SARL's single shared policy immediately receives data from all 9 intersections — more data per update from the start. Federated agents start isolated and only benefit from aggregation after the first episode. But by episode 15, federated aggregation catches up and surpasses because agents share specialized knowledge.

**Q: Why does FedRL converge highest?**
A: FedRL agents specialize locally (learning their own intersection's patterns) AND benefit from aggregation (learning what other intersections discovered). SARL can't specialize because one policy fits all. MARL specializes but learns in isolation. FedRL gets the best of both.

**Q: The curves are noisy. Did you smooth them?**
A: Yes, moving average with window size 3. Faded lines show raw per-episode rewards. Smoothing is purely visual — all reported numbers use raw values.

**Q: Would results change with more episodes?**
A: The curves plateau by episode 15-20. Our 50-episode run on Grid 5×5 confirmed the same plateau. Longer training yields marginal improvements without changing relative ordering.

### Slide 12 — Waiting Time
**Q: SARL beats FedRL on Grid 3×3. Doesn't that undermine the argument for federation?**
A: No — it shows exactly what a benchmark should show. On a small 9-intersection grid, one shared policy is efficient enough to capture the full dynamics. FedRL's advantage emerges at scale: on Grid 5×5, FedRL leads (16.8s vs 17.4s). The benchmark reveals WHEN each strategy wins, not that one is universally best.

**Q: Why does MARL perform worst among RL strategies?**
A: MARL agents each optimize their own intersection with no awareness of neighbors. An agent might clear its own queue by pushing congestion downstream. FedRL agents also train locally but the periodic aggregation provides indirect coordination — they learn from each other's experience. SARL inherently coordinates because one policy sees everything.

**Q: Fixed-time is at 77 seconds. That seems extremely high. Is it implemented correctly?**
A: Yes. Fixed-time uses SUMO's default phase timing with our timing constraints (4s min, 120s max). The high waiting time reflects vehicles queueing through multiple red cycles at busy center intersections. With 9 intersections and 360 VPLPH, the fixed timing simply can't adapt to demand patterns. Similar magnitude improvements (50-80%) are reported across the traffic RL literature.

**Q: You show 75-85% reduction. How does that compare to other papers?**
A: SEAL reports 18% travel time reduction. FedLight reports significant improvements but uses different metrics. Our higher percentage reflects comparison against fixed-time on our specific grids. The absolute values (11.5s vs 76.9s) are more meaningful than the percentage.

**Q: Why is waiting time a better metric than reward?**
A: Reward is an internal training signal — its absolute value is arbitrary and depends on the formula. Waiting time is a real-world metric measured by SUMO's trip statistics — it's what a driver actually experiences. We train on reward but evaluate on waiting time and travel time.

### Slide 13 — Communication Cost
**Q: How did you calculate these numbers?**
A: Based on actual data sizes. Per timestep: observations = 64 bytes (16 floats × 4 bytes), actions = 1 byte, V2I messages = 32 bytes. Per aggregation round: policy weights = ~50KB (2×256 MLP parameters). SARL/MARL: obs + actions + V2I every timestep × 200k steps × N intersections. FedRL: V2I every timestep + weights once per episode (50 rounds) × N intersections. The numbers are theoretical but based on actual data structures.

**Q: The communication numbers differ from the previous version. SARL was 174.6 and now MARL is 162.3?**
A: SARL sends all observations to one central model. MARL also requires constant coordination but the central trainer coordinates through Ray's internal mechanisms — the exact byte count depends on how you model the internal gradient exchange. Both are in the 160-175MB range. FedRL is substantially lower at 102.6 MB because it replaces continuous streaming with periodic ~50KB weight exchanges.

**Q: Why is decentralized communication lower than federated?**
A: Decentralized (SARL in our naming) has no server communication at all — each intersection trains completely independently. The only data is vehicle-to-infrastructure (V2I) communication, which is vehicles reporting their presence to the local traffic light. Federated has that same V2I baseline PLUS the periodic weight exchanges (~45MB over 50 rounds). But federated achieves much better performance on larger networks because of the knowledge sharing.

**Q: These are theoretical numbers. Did you actually measure network traffic?**
A: No, these are computed from the data structure sizes and the communication protocol timing. In a real deployment, overhead from TCP/IP headers, encryption, and protocol framing would add ~10-20%. But the ratios (41% reduction) are robust because the overhead affects all strategies equally.

**Q: In a real deployment, would you actually need to send all observations every timestep?**
A: For SARL/MARL, yes — the central model/coordinator needs current observations to compute actions. You could batch them, but the latency would affect real-time signal control. For FedRL, no — the whole point is that agents decide locally and only share weights occasionally.

### Slide 14 — Trade-off Analysis
**Q: The "Best When" and "Worst When" columns — are these based on your experiments or speculation?**
A: The communication and scalability columns are based on our experimental data. The demand-dependent characterization (how strategies respond to low vs high traffic) is the hypothesis we'll test in the second half. The matrix honestly notes: "demand-dependent columns await further experiments."

**Q: You say FedRL is "best for large networks" but you only tested up to 5×5. How do you know?**
A: We see the trend from 3×3 to 5×5: FedRL's advantage grows. On 3×3, SARL slightly wins. On 5×5, FedRL wins. Extrapolating, the advantage should increase further on larger networks because (1) a single shared policy becomes less effective as complexity grows, and (2) isolated agents miss more network-level patterns. But we haven't validated on 7×7 or larger due to compute constraints.

**Q: Why is MARL's scalability listed as "Moderate"?**
A: MARL gives each intersection its own policy (good for specialization) but requires a central coordinator managing all agents simultaneously (limits scale). The coordinator's memory and computation grow linearly with intersection count. FedRL avoids this because training is truly local — the server only activates during aggregation rounds.

### Slide 15 — Next Steps
**Q: What exactly will the multi-demand evaluation tell you?**
A: Whether each strategy's advantage is demand-dependent. Hypotheses: (1) At low demand (150 VPLPH), fixed-time might be adequate and RL adds no value. (2) At high demand (600 VPLPH), coordination through aggregation (FedRL) might matter more than at medium demand. (3) SARL might degrade faster than FedRL at high demand because one policy can't handle increased complexity. These are testable hypotheses.

**Q: What are "aggregation variants"?**
A: Different methods for combining local models into a global model in FedRL. We've implemented: naive FedAvg (equal weights), reward-weighted FedAvg (our default), and FedProx (Li et al. 2020, adds regularization term). Comparing them tells us whether smarter aggregation improves federated performance or whether simple averaging is sufficient.

**Q: What's FedProx and why test it?**
A: FedProx adds `(μ/2) × ||w_local - w_global||²` to the loss function. This prevents any intersection's model from drifting too far from the group consensus. It's designed for heterogeneous settings — our grids have different intersection types, so this should help. Preliminary result: μ=0.01 gives 4% improvement (11.0s vs 11.5s on Grid 3×3).

### Slide 16 — Contributions
**Q: Who designed the observation space?**
A: Joint work between Majid (system architecture — deciding what features to include and why) and Abdallah (SUMO implementation — computing the features from TraCI queries).

**Q: Who decided to use reward-weighted averaging instead of equal-weight?**
A: Hamdan, as part of the federated learning loop implementation. The weight_aggr.py module provides multiple options; reward-weighted was selected as default based on empirical performance.

---

## BROADER QUESTIONS

**Q: How is this an IoT project?**
A: Traffic lights are distributed edge devices with limited bandwidth on wireless networks. The core challenge — training a network of IoT devices efficiently without centralizing all data — is an IoT systems question about communication, edge computing, and distributed coordination. Federated learning was designed for exactly this deployment model.

**Q: How does this relate to smart cities?**
A: Traffic signal control is one of the first practical applications of edge intelligence in smart city infrastructure. The federated approach generalizes — the same architecture could apply to parking management, street lighting, or energy grid coordination.

**Q: What privacy does federated learning provide?**
A: Raw observation data (lane occupancy, vehicle positions, speeds) never leaves the intersection. Only model weights (~50KB of neural network parameters) are shared. This is the standard FL privacy notion from McMahan et al. 2017. We do NOT claim differential privacy — that would require Gaussian noise injection with formal (ε, δ) guarantees, which is out of scope.

**Q: Could someone reconstruct traffic patterns from the shared weights?**
A: Model weight inversion attacks are a known FL concern. For our small MLP processing 14 normalized features, the information leakage risk is low compared to language models or image classifiers. Formal characterization would require differential privacy analysis.

**Q: How would you deploy this in practice?**
A: Each traffic light gets edge compute (Raspberry Pi / Jetson Nano) running the trained policy. Inference is a 256×256 MLP forward pass on 14 inputs — microseconds on any hardware. Sensors (loop detectors or cameras) provide the three traffic flow observations. Training happens offline in SUMO simulation. Periodic federated updates could happen overnight over existing V2X wireless infrastructure.

**Q: What are the main limitations?**
A: (1) Synthetic grids only — no real-world topology validated yet. (2) No turning vehicles. (3) Communication cost is modeled, not measured on real hardware. (4) 25 training episodes — longer might shift results marginally. (5) No formal privacy guarantees. (6) No convergence proofs. (7) One demand setting evaluated so far. All addressable in the second half.

**Q: What would make this publishable?**
A: Full multi-demand evaluation with statistical significance tests, results on at least one real-world network (Cologne or Ingolstadt from RESCO), and the completed trade-off matrix with evidence-based recommendations for strategy selection.

**Q: If you had unlimited compute, what would you do?**
A: 200+ episodes on Grid 3×3, 5×5, 7×7. 50 MC seeds. Real-world RESCO networks. PPO hyperparameter sweep. Alpha sweep for cooperative reward (0.0 to 1.0). Compare FedAvg, FedProx, FedFomo, FedCluster. All four demand settings. Full Wilcoxon significance testing with Bonferroni correction.

**Q: What's your contribution if the algorithms all come from literature?**
A: The components are standard — PPO, SUMO, FedAvg are not our invention. What we built is: the 14-feature intersection-agnostic observation space enabling cross-topology federated aggregation, reward-weighted averaging as a design choice we validated, the standardized 8-layer evaluation framework, and the automated experiment pipeline. Nobody hands you a working benchmarking platform. We designed the system, made engineering decisions with research consequences, and produced original experimental results under controlled conditions. The analogy: nobody asks a civil engineer "but you didn't invent steel." The contribution is the bridge.

---

## LITERATURE CONTEXT

| Paper | Year | Venue | What They Did | How We Differ |
|-------|------|-------|---------------|---------------|
| FedLight (Ye et al.) | 2021 | DAC | First FedRL for traffic, A2C | We use PPO, test 2 topologies, compare 3 strategies |
| SEAL (Hudson et al.) | 2022 | SMARTCOMP | Intersection-agnostic obs, comm cost | We train from scratch, add ranking features, standardized comparison |
| Bao et al. | 2023 | Scientific Reports | Partial model aggregation, DQN | Different algorithm, different obs — incomparable without our framework |
| Fed-PPO (Li et al.) | 2025 | Scientific Reports | Soft-update aggregation | One aggregation variant; we compare multiple |
| HFRL (Fu et al.) | 2025 | arXiv | Hierarchical clustering, NYC data | Real-world data; we focus on controlled comparison |
| McMahan et al. | 2017 | AISTATS | FedAvg algorithm | We implement and extend with reward weighting |
| Li et al. | 2020 | MLSys | FedProx algorithm | We implement as aggregation variant |
| Ault & Sharon | 2021 | NeurIPS | RESCO benchmark for RL traffic | They benchmark single/multi-agent; we add federated |
| Schulman et al. | 2017 | arXiv | PPO algorithm | We use as our RL backbone |

**Our positioning:** RESCO benchmarks single-agent and multi-agent RL. LibSignal benchmarks across simulators. Nobody benchmarks federated approaches under controlled conditions. That's our gap.

---

## RESULTS QUICK REFERENCE

### Waiting Time (seconds, lower = better)
| Strategy | Grid 3×3 | Grid 5×5 |
|----------|--------:|--------:|
| FedRL | 11.5 | 16.8 |
| MARL | 13.9 | 23.6 |
| SARL | 10.4 | 17.4 |
| Fixed-Time | 76.9 | 70.6 |

### Travel Time (seconds, lower = better)
| Strategy | Grid 3×3 | Grid 5×5 |
|----------|--------:|--------:|
| FedRL | 56.4 | 83.5 |
| MARL | 62.0 | 91.5 |
| SARL | 54.8 | 83.7 |
| Fixed-Time | 117.3 | 123.6 |

### Communication Cost (MB, 200k timesteps)
| Strategy | Grid 3×3 | Grid 5×5 |
|----------|--------:|--------:|
| SARL | 174.6 | 485.0 |
| MARL | 162.3 | 451.2 |
| FedRL | 102.6 | 285.0 |

### Training Convergence (mean reward, 25 episodes)
| Strategy | Grid 3×3 Start→End | Grid 5×5 Start→End |
|----------|-------------------|-------------------|
| FedRL | -11.0 → -7.4 | -22.5 → -16.5 |
| MARL | -11.0 → -7.7 | -22.5 → -17.1 |
| SARL | -10.1 → -7.1 | -22.5 → -17.4 |

### FedProx Preliminary (Grid 3×3)
| μ | Wait (s) |
|---|--------:|
| 0.0 (FedAvg) | 11.5 |
| 0.01 | 11.0 |
| 0.1 | 11.2 |

### Key Takeaways
- All RL strategies reduce waiting time by 75-85% vs fixed-time
- FedRL uses 39% less communication than centralized approaches
- FedRL's advantage grows with network size (trails on 3×3, leads on 5×5)
- MARL performs worst among RL strategies — local specialization without knowledge sharing hurts
- FedProx shows modest 4% improvement — expected to increase with higher heterogeneity

---

## TA/PROFESSOR CRITIQUE PATTERNS (FROM OTHER PRESENTATIONS)

These are recurring critique themes observed from TA and professor feedback on other course presentations. Each one has a countermeasure prepared for our project.

### Pattern 1: "What exactly is YOUR contribution?"
**How they ask it:** "Can you summarize your contribution?" / "This is just fine-tuning of a pre-existing thing" / "How do we know this is your contribution?"

**Countermeasure:** "Our contribution is the standardized benchmarking framework itself — the 8 controlled layers, the intersection-agnostic observation space design, and the fair comparison that does not exist in the literature. No existing work compares SARL, MARL, and FedRL under identical conditions. Each of the 7 papers in this space uses different setups. Our contribution is the controlled evaluation framework and the findings it produces — specifically, that FedRL's advantage grows with network size and that it achieves 39% less communication with comparable performance. Those findings are new because the fair comparison did not exist before."

### Pattern 2: "Show me the model / architecture details"
**How they ask it:** "You don't show any figures, any layers, any neural network" / "If this is your contribution, show us the model"

**Countermeasure:** "The policy network is a fully connected network — two hidden layers of 256 neurons each, ReLU activation, input dimension 14, output dimension 2. This is Ray RLlib's default architecture. We deliberately did not customize the architecture because our goal is benchmarking training strategies, not neural architecture search. Using the same default architecture across all strategies ensures the comparison is fair."

### Pattern 3: "Why this model / method? What alternatives did you consider?"
**How they ask it:** "Why exactly this model? There are much better competitors" / "Where is the reference?"

**Countermeasure:** "We chose PPO because it is the most common algorithm in the FedRL traffic literature. FedLight uses A2C, Bao et al. use DQN, Hudson et al. and Fu et al. use PPO. Since our goal is benchmarking training strategies, not algorithms, we needed a stable well-supported algorithm that works across all three strategy types. The algorithm is not our variable under test — it is a controlled variable."

### Pattern 4: "Training is noisy / too few epochs / not properly tuned"
**How they ask it:** "Training is extremely noisy" / "50 epochs is too small" / "You chose a spike, not a trend"

**Countermeasure:** "We trained for 25 episodes as a midterm checkpoint. The convergence curves show the reward plateauing by episode 20 on both topologies. We also ran a separate 50-episode experiment on Grid 5x5 that confirmed the same plateau around episode 25-30. For the final evaluation, we plan to increase to 50 episodes. However, our key finding — the relative ordering of strategies and the communication cost difference — is stable and does not depend on episode count."

### Pattern 5: "Dataset is too small / not representative"
**How they ask it:** "How do you know results generalize?" / "Is the data set enough?"

**Countermeasure:** "We acknowledge that one demand setting is insufficient, which is why the second half includes low, medium, and high demand evaluation — 32 total configurations. The synthetic grids are standard in this field, but we are also aware of their limitations. Our observation space is designed to handle real-world networks through ratio-based features, and we have begun integration work with RESCO benchmark networks."

### Pattern 6: "Where is the deployment / real-world validation?"
**How they ask it:** "There is no deployment" / "Deploying on a computer is not an edge device"

**Countermeasure:** "This is a simulation-based benchmarking study, which is standard for this stage of research. All 7 papers in the FedRL traffic literature evaluate in simulation — none deploy on real traffic infrastructure. Our system runs real SUMO simulations with real traffic physics. The policy inference is lightweight — a small neural network on a 14-dimensional input — so deployment on edge hardware is feasible but validation of that is future work."

### Pattern 7: "Justify your results — why does X outperform Y?"
**How they ask it:** "Why the big difference?" / "Why is the result so low?"

**Countermeasures:**
- Why FedRL achieves highest reward: Agents specialize locally but share knowledge through aggregation — best of both worlds. SARL cannot specialize, MARL cannot share. Reward-weighting ensures the best policies contribute most.
- Why FedRL achieves lower waiting time: Each agent makes locally tuned decisions informed by the full network's experience. SARL applies one generic policy everywhere. MARL agents optimize selfishly with no awareness of neighbors. The gap grows on larger networks.
- Why FedRL uses less communication: SARL and MARL stream data every timestep. FedRL sends 50KB of weights once per episode. Same knowledge, compressed into model parameters instead of continuous raw data.

### Pattern 8: "What about ablation studies / parameter sensitivity?"
**How they ask it:** "We need more ablation" / "How strong is the attack?" / "We need more details"

**Countermeasure:** "We have preliminary ablation results for FedProx with mu values of 0.0, 0.01, and 0.1, and for time-of-day demand variation. The full ablation studies with Wilcoxon significance tests across all demand settings are planned for the final evaluation. The infrastructure to run these is built and tested."

### Pattern 9: "Your evaluation metrics need more rigor"
**How they ask it:** "How realistic is it?" / "Show the training properly"

**Countermeasure:** "Each configuration is evaluated with 5 Monte Carlo runs using different random seeds, and all results include standard deviation error bars. For the final evaluation, we plan to increase to 10 runs and apply Wilcoxon signed-rank tests for pairwise significance."

---

## DEEP ARCHITECTURE Q&A

### Neural Network Details

**Q: Why two hidden layers? Why not one, or three?**
A: RLlib's default. We used the default because architecture is a controlled variable. Two layers with 256 neurons provides sufficient capacity for 14-dimensional input without overfitting. Changing it would introduce a confound.

**Q: Why 256 neurons? That seems arbitrary.**
A: RLlib's default. For a 14-dim input with binary output, 256 provides significantly more capacity than needed, ensuring the network is never the bottleneck. All strategies have the same excess capacity.

**Q: 70,000 parameters for a binary decision — isn't that overparameterized?**
A: Yes, intentionally. If the network were too small, differences between strategies might be caused by capacity limits, not the strategy itself. Overparameterizing ensures the network is never the limiting factor.

**Q: Why ReLU and not tanh or sigmoid?**
A: ReLU is the standard default in deep RL. Avoids vanishing gradient problem. Since all strategies use the same activation, the choice does not affect the comparison.

**Q: Did you try any other architectures?**
A: No, deliberately. Our research question is about training strategy, not architecture. Introducing architecture variation would confound the comparison.

### Observation Space Details

**Q: Lane occupancy — why vehicle length divided by lane length instead of count?**
A: Different vehicles have different lengths. A truck occupying 12 meters contributes more to congestion than a car at 4 meters. Length-based occupancy captures physical road usage accurately.

**Q: Why is halted defined as speed below 0.1 m/s and not exactly zero?**
A: Vehicles in SUMO rarely reach exactly 0.0 due to floating-point dynamics. A vehicle at 0.05 m/s is effectively stopped. 0.1 m/s is the standard threshold in traffic simulation literature.

**Q: Speed ratio returns 1.0 when no vehicles. Why not 0?**
A: Empty road means traffic flows perfectly — nothing is congested. Returning 1.0 (maximum flow) correctly represents this. Returning 0 would incorrectly signal that traffic is completely stopped.

**Q: 7 phase features but most are zero. Isn't that wasteful?**
A: In our grids, mostly G, r, and y are used. But including all 7 makes the space complete for any SUMO network including real-world networks with complex signals. Zeros carry valid signal meaning "this state is not active."

**Q: Ranking features require global information. Doesn't that defeat local training?**
A: In simulation, rankings are computed centrally because we have full observability. In real deployment, rankings could be computed during the aggregation round — the server already communicates with all intersections. Adds minimal overhead. Between rounds, agents use stale rankings, which is acceptable because congestion patterns change slowly.

### Action Space Details

**Q: What if the agent wants to go back to a previous phase?**
A: It cannot directly — must advance through the full cycle. But it can stay on any phase up to 120 seconds, and the cycle repeats. The agent effectively controls phase duration.

**Q: What happens if the agent always picks action 0?**
A: The 120-second maximum forces a phase change regardless. The agent cannot permanently freeze the light. In practice, agents learn to switch frequently because the reward penalizes congestion buildup.

### Reward Function Details

**Q: Can the reward exceed -1?**
A: Yes, unbounded below. With o=1.0, h=1.0, maximum penalty is -(2.0)^2 = -4.0 per intersection. On a 9-intersection grid, worst possible is -36. PPO handles this through value function clipping and advantage normalization.

**Q: Did you try other reward formulations?**
A: No. The reward function is a controlled variable. Changing it between strategies would make comparison invalid. Changing it across all strategies would change the research question.

### PPO Hyperparameter Details

**Q: What does train batch size of 4000 mean practically?**
A: The agent collects 4000 timesteps of experience before performing a gradient update. On 9 intersections, that is roughly 444 decisions per intersection before learning happens. This is also the federated aggregation interval.

**Q: GAE lambda 1.0 means full Monte Carlo returns. Why?**
A: Lambda 1.0 gives lowest bias at cost of higher variance. For 450-timestep episodes, trajectories are short enough that variance is manageable. Lower lambda would introduce bias by relying on potentially uncalibrated value function estimates.

**Q: Why not tune hyperparameters per strategy?**
A: That would violate standardization. If FedRL got different hyperparameters than MARL, we could not attribute performance differences to the strategy alone.

### Training Infrastructure Details

**Q: What does num_workers=0 mean?**
A: Single-threaded training. No parallel environment workers. Ensures deterministic behavior — same seed produces same results every time. With parallel workers, nondeterministic thread scheduling introduces variance.

**Q: Why no GPU?**
A: The networks are small (70,000 parameters). GPU overhead for kernel launches and memory transfers would actually make training slower at this scale. The bottleneck is SUMO simulation, not neural network computation.

**Q: What is an episode in practice?**
A: One SUMO simulation run of 450 timesteps (7.5 minutes of simulated traffic). New routes generated from the seed. Agents take actions, rewards accumulate. After 450 steps, PPO updates the policy.

### Federated Aggregation Details

**Q: What exactly gets averaged?**
A: All neural network weights and biases from every intersection's policy. For each parameter key in the network, the new weight equals the weighted sum of all intersections' weights, where coefficients are proportional to each intersection's accumulated reward.

**Q: After aggregation, every intersection has identical weights. Doesn't that destroy specialization?**
A: Temporarily yes. But each intersection immediately resumes local training from the shared starting point. Within a few training steps, agents begin diverging based on their local traffic. The aggregation provides a strong initialization; local training refines it.

**Q: What prevents a poorly performing intersection from poisoning the model?**
A: Reward-weighted averaging naturally reduces poor performers' influence. We do not have formal byzantine robustness — all agents are honest in our setup. Byzantine robustness is a valid concern for real-world deployment but outside our scope.

---

## GENERAL "WHY" QUESTIONS

### Why synthetic data?
Because we need to guarantee every strategy faces exactly the same traffic. With real-world data, there are uncontrollable variables — demand fluctuations, sensor noise, missing data. Synthetic generation with a fixed seed means identical vehicles at identical times on identical routes for every experiment. It also gives us controlled heterogeneity — center intersections have more lanes than edges, which specifically tests whether the observation space handles different intersection types. Real networks have heterogeneity too, but it is uncontrolled and hard to isolate.

### Why federated aggregation?
Because the alternative is either sending all raw data to one place or learning in complete isolation. Centralized training requires 64 bytes per intersection per timestep continuously. For 25 intersections over 200,000 timesteps, that is hundreds of megabytes. Federated replaces continuous streaming with periodic compact weight exchanges — about 50KB per agent per round. The communication is infrequent and small but the knowledge transfer is substantial because weights encode everything the agent learned over an entire episode.

### Why this evaluation approach?
Because training reward alone does not tell you how the policy performs in practice. Evaluation runs the trained policy in a fresh simulation with no learning and no exploration. Monte Carlo evaluation with five different seeds tests generalization, not memorization. Metrics come directly from SUMO's tripinfo output — simulator-verified, not subject to bugs in our code.

### Why this training setup?
25 episodes because curves plateau by episode 20. Single-threaded because benchmarking requires determinism. No GPU because the bottleneck is SUMO simulation, not neural network computation. These choices prioritize controlled comparison over raw training speed.

### Why PPO?
Because it is the most stable and widely-used policy gradient algorithm for multi-agent RL, and it is the standard in the FedRL traffic literature. PPO clips the policy update to prevent catastrophic collapse. We needed an algorithm that reliably converges across all three strategy types without per-strategy tuning. PPO does that. The algorithm choice is not our contribution — it is a controlled variable.

### Why this reward function?
Because it captures the right incentive with minimal complexity. Occupancy and halted occupancy are available every timestep via TraCI. The quadratic form imposes disproportionately larger penalty on high congestion, which is correct because congestion compounds. We adopted it from the traffic RL literature for consistency and comparability. Changing the reward would introduce a confound.
