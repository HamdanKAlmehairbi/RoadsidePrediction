# Phase 4: Core Extensions - Research

**Researched:** 2026-03-22
**Domain:** FedRL extensions — FedProx, cooperative reward shaping, time-of-day demand curriculum, observation space encoding
**Confidence:** MEDIUM-HIGH (codebase read directly; FedProx hook strategy verified via Ray source + PyTorch forum; SUMO route generation understood from existing code)

---

## Summary

Phase 4 adds four research-depth extensions to the working FedRL system built in Phases 1-3. All extensions are additive: they do not replace existing code but layer on top of it via new parameters. The codebase is well-structured for this — FedPolicyTrainer inherits from BaseTrainer which already builds PPO via the old API stack, routes/train.py already has a clean TrainRequest model, and training_runner.py has a thin create_trainer() shim.

The central implementation challenge is EXT-01 (FedProx). Adding a proximal term to PPO loss requires subclassing PPOTorchPolicy and overriding loss(). The old API stack supports this pattern; the loss() method on PPOTorchPolicy can be overridden by calling super().loss() to get the base loss and adding the proximal term on top. Global weights needed for the proximal term must be stored on the policy instance (set at each FedAvg aggregation step). This is a workable pattern with no library changes required.

EXT-02 (cooperative reward) is straightforward: the reward is computed per-TLS in SumoEnv._get_reward(), and tls_graph (road adjacency) is already computed in TrafficLightHub. The alpha-weighted neighbor average just needs the obs dict passed to _get_reward() alongside the neighbor IDs.

EXT-03 (time-of-day demand) is implemented entirely in route generation. generate_random_routes() already accepts vehicles_per_lane_per_hour. The curriculum splits each episode's horizon into time segments and calls generate_random_routes() with different vplph values at reset, or triggers mid-episode route updates.

EXT-04 (time encoding) requires expanding the observation space from 14 to 16 features by appending sin/cos encoding of normalized episode step. Both the observation space Box and get_observation() method need updating, and all downstream code that hard-codes N_RANKED_FEATURES must be verified.

**Primary recommendation:** Implement all four extensions as opt-in via new API parameters (fedprox_mu, alpha, time_of_day, use_time_encoding). Default values of 0/1.0/False/False preserve exact backward compatibility with existing experiments.

---

## Standard Stack

No new libraries are required. All extensions use existing dependencies.

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | FedProx proximal term computation | Already used by RLLib PPO; autograd handles gradient |
| Ray RLLib | 2.x (old API stack) | PPOTorchPolicy subclassing for custom loss | Already in use; old stack supports loss() override |
| numpy | 1.x | Weight tensor arithmetic for proximal term | Already used throughout seal/trainer |
| SUMO / randomTrips | existing | Time-of-day route generation | generate_random_routes() already wraps randomTrips.py |

### No New Dependencies
All extensions are implemented with existing stack. Do NOT add:
- Flower (flwr) — overkill, existing FedAvg is custom
- Any FL framework — this is a custom implementation atop RLLib

---

## Architecture Patterns

### Recommended Project Structure (new files only)

```
BackEnd/
├── seal/
│   ├── trainer/
│   │   ├── fed_agent.py            # MODIFY: add FedProx support
│   │   ├── fedprox_policy.py       # NEW: FedProxPPOTorchPolicy subclass
│   │   └── weight_aggr.py          # no change needed
│   └── sumo/
│       ├── env.py                  # MODIFY: cooperative reward, time encoding
│       ├── config.py               # MODIFY: N_RANKED_FEATURES 14->16
│       └── kernel/trafficlight/
│           ├── light.py            # MODIFY: get_observation() append time encoding
│           └── space.py            # MODIFY: Box shape from N_RANKED_FEATURES
├── api/
│   ├── routes/
│   │   └── train.py                # MODIFY: add fedprox_mu, alpha, time_of_day, use_time_encoding params
│   └── training_runner.py          # MODIFY: create_trainer() passes new params
```

### Pattern 1: FedProx via PPOTorchPolicy Subclass (EXT-01)

**What:** Create FedProxPPOTorchPolicy that overrides loss() to add the proximal term. Store global weights on the policy instance so they're accessible during forward pass.

**When to use:** When fedprox_mu > 0 (otherwise use vanilla PPOTorchPolicy — zero cost fallback).

**Implementation logic:**

```python
# Source: Ray RLLib source (verified) + FedProx paper pattern
# BackEnd/seal/trainer/fedprox_policy.py

import torch
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.utils.annotations import override
from typing import Union, List, Type
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType


class FedProxPPOTorchPolicy(PPOTorchPolicy):
    """PPO policy with FedProx proximal term in loss.

    The proximal term mu/2 * ||w - w_global||^2 keeps local weights
    close to the last global model, improving convergence on heterogeneous clients.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_weights = None   # set by FedPolicyTrainer after each FedAvg
        self._fedprox_mu = 0.0        # set via set_fedprox_mu()

    def set_fedprox_mu(self, mu: float) -> None:
        self._fedprox_mu = mu

    def store_global_weights(self) -> None:
        """Snapshot current weights as the global reference point."""
        self._global_weights = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        # Get base PPO loss
        base_loss = super().loss(model, dist_class, train_batch)

        # Add proximal term only if mu > 0 and global weights are set
        if self._fedprox_mu > 0.0 and self._global_weights is not None:
            proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
            for name, param in model.named_parameters():
                if name in self._global_weights:
                    global_w = self._global_weights[name].to(param.device)
                    proximal_term += (self._fedprox_mu / 2.0) * torch.sum(
                        (param - global_w) ** 2
                    )
            return base_loss + proximal_term

        return base_loss
```

**Wiring into FedPolicyTrainer:**

```python
# In FedPolicyTrainer.__init__(), when fedprox_mu > 0:
self.policy_type = FedProxPPOTorchPolicy   # replaces PPOTorchPolicy
self.fedprox_mu = fedprox_mu

# In FedPolicyTrainer.on_data_recording_step(), after aggregation:
if aggregate_this_round and self.fedprox_mu > 0:
    for policy_id in self.policies:
        policy = self.ray_trainer.get_policy(policy_id)
        policy.set_fedprox_mu(self.fedprox_mu)
        policy.store_global_weights()  # snapshot new global as reference
```

**Critical detail:** `store_global_weights()` must be called AFTER `set_weights(new_params)` so the reference point is the fresh global model, not the stale local one. This matches FedProx paper semantics (proximal to w^t, the global at round t).

### Pattern 2: Cooperative Reward with Neighbor Graph (EXT-02)

**What:** reward = alpha * r_local + (1 - alpha) * mean(r_neighbors). Neighbor graph already exists in tls_hub.tls_graph.

**When to use:** When alpha < 1.0.

**Key insight:** `_get_reward()` in SumoEnv currently receives only the single TLS's obs. To compute cooperative reward, we need the full obs dict and the TLS id. Change the call signature.

```python
# In SumoEnv.step(), current code:
reward = {tls.id: self._get_reward(obs[tls.id])
          for tls in self.kernel.tls_hub}

# New code for cooperative reward:
reward = self._get_all_rewards(obs)

# New method:
def _get_all_rewards(self, obs: Dict) -> Dict:
    local_rewards = {
        tls.id: self._get_local_reward(obs[tls.id])
        for tls in self.kernel.tls_hub
    }
    if self.alpha >= 1.0:
        return local_rewards
    graph = self.kernel.tls_hub.tls_graph
    cooperative_rewards = {}
    for tls in self.kernel.tls_hub:
        neighbors = graph.get(tls.id, [])
        if neighbors:
            neighbor_mean = sum(local_rewards[n] for n in neighbors) / len(neighbors)
        else:
            neighbor_mean = local_rewards[tls.id]
        cooperative_rewards[tls.id] = (
            self.alpha * local_rewards[tls.id] +
            (1.0 - self.alpha) * neighbor_mean
        )
    return cooperative_rewards

def _get_local_reward(self, obs: np.ndarray) -> float:
    """Original per-TLS reward (rename from _get_reward)."""
    return -1 * (obs[LANE_OCCUPANCY] + obs[HALTED_LANE_OCCUPANCY])**2
```

**Backward compatibility:** alpha defaults to 1.0 in env_config, so the existing reward function is unchanged when alpha=1.0.

**Alpha is stored in env_config** and consumed at env __init__ time, just like `ranked` is today. Pass it via `env_config_fn()` in BaseTrainer.

### Pattern 3: Time-of-Day Demand Curriculum (EXT-03)

**What:** During each episode reset, generate routes with varying vehicles_per_lane_per_hour across time segments (AM rush=600, midday=240, PM rush=540). This is a single-call-per-episode approach (generate varied route file at reset), not a mid-episode SUMO re-route.

**Why single call:** generate_random_routes() generates a complete .rou.xml before the episode begins. SUMO randomTrips does not support time-varying demand natively at the point the tool is invoked; however, the tool does support --begin/--end offsets so multiple passes can be concatenated. The simplest correct approach is: generate separate trip sets for AM/midday/PM time windows and concatenate them into one .rou.xml.

**Implementation approach:**

```python
# In AbstractSumoEnv.rand_routes() (or override in SumoEnv):
def rand_routes_time_of_day(self) -> None:
    """Generate a time-of-day curriculum route file."""
    # Three demand periods within the episode horizon
    horizon = self.horizon or 3600
    period_len = horizon // 3
    segments = [
        (0,          period_len,     600),   # AM rush
        (period_len, 2*period_len,   240),   # Midday
        (2*period_len, horizon,      540),   # PM rush
    ]
    # Generate trips for each segment separately, then merge XML
    # ... write merged traffic.rou.xml
```

**Alternative (simpler, slightly less accurate):** Call `generate_random_routes()` once per episode with a randomly sampled vplph from [240, 600] (uniform over the demand range). This gives variation without the multi-segment complexity. Reserve multi-segment for a second task if needed.

**Recommended approach for planning:** Use two sub-tasks — Task A generates a simple variable-vplph env, Task B implements proper AM/midday/PM segments with merged route XML.

**Passed via env_config:** `"time_of_day": True` in env_config_fn(). SumoEnv checks this flag in rand_routes() to switch between fixed and curriculum generation.

### Pattern 4: Sine/Cosine Time Encoding (EXT-04)

**What:** Append two features to each observation: sin(2π * t/T) and cos(2π * t/T) where t = step_counter, T = horizon. This allows the policy to learn time-of-day behavior patterns.

**Observation space change:** N_RANKED_FEATURES goes from 14 to 16. N_UNRANKED_FEATURES goes from 10 to 12.

**Files that hardcode the feature count:**
- `seal/sumo/config.py` — defines N_RANKED_FEATURES, N_UNRANKED_FEATURES, and all index constants
- `seal/sumo/kernel/trafficlight/space.py` — Box shape uses N_RANKED_FEATURES
- `seal/sumo/kernel/trafficlight/light.py` — get_observation() initializes array of n_features length
- Evaluation runner loads pickled weights — changing obs space breaks backward compatibility with existing weights

**Backward compatibility warning:** Adding 2 features changes the policy network input layer shape. Weights trained without time encoding are INCOMPATIBLE with policies that have time encoding. This must be a flag-gated additive feature:
- When `use_time_encoding=False` (default): N_RANKED_FEATURES remains 14
- When `use_time_encoding=True`: N_RANKED_FEATURES becomes 16

The cleanest implementation: instead of modifying config.py constants, the time encoding features are appended in SumoEnv._observe() after get_observation() returns, and the observation_space property in SumoEnv overrides the hub-level space to add 2 when the flag is active.

```python
# In SumoEnv._observe():
obs = {tls.id: tls.get_observation() for tls in self.kernel.tls_hub}
if self.use_time_encoding:
    horizon = self.horizon or 3600
    t = self.step_counter
    sin_t = math.sin(2 * math.pi * t / horizon)
    cos_t = math.cos(2 * math.pi * t / horizon)
    for tls_id in obs:
        obs[tls_id] = np.append(obs[tls_id], [sin_t, cos_t]).astype(np.float32)
# ... existing ranked/cleanup logic ...
```

```python
# In SumoEnv.observation_space property:
@property
def observation_space(self):
    base = self.kernel.tls_hub[first].observation_space  # Box(14,) or Box(10,)
    if self.use_time_encoding:
        n = base.shape[0] + 2
        return Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)
    return base
```

**Note:** sin/cos time features can fall outside [0,1] range. The Box bounds should be [-1,1] or the features should be scaled to [0,1] using (sin+1)/2. Recommend [-1, 1] bounds for the 2 appended features only, or use a broader Box(-inf, inf) for the time component. Cleanest: use np.float32 Box with low=np.array([0]*base_n + [-1,-1]) and high=np.array([1]*base_n + [1,1]).

### Pattern 5: API Extension (EXT-01 through EXT-04)

**TrainRequest additions:**

```python
class TrainRequest(BaseModel):
    trainer: str
    aggr: Optional[str] = None
    topology: str = "grid-3x3"
    ranked: bool = True
    n_episodes: int = 50
    fed_step: int = 1
    # New in Phase 4:
    fedprox_mu: float = 0.0          # EXT-01: 0.0 = pure FedAvg
    alpha: float = 1.0               # EXT-02: 1.0 = fully selfish
    time_of_day: bool = False         # EXT-03: False = fixed demand
    use_time_encoding: bool = False   # EXT-04: False = original 14-feature obs
```

**create_trainer() additions:**

```python
def create_trainer(trainer_type, topology, ranked, fed_step=1, aggr=None,
                   n_episodes=50, fedprox_mu=0.0, alpha=1.0,
                   time_of_day=False, use_time_encoding=False):
    ...
    common_kwargs["env_config_extra"] = {
        "alpha": alpha,
        "time_of_day": time_of_day,
        "use_time_encoding": use_time_encoding,
    }
    if trainer_upper == "FEDRL":
        trainer = FedPolicyTrainer(
            fed_step=fed_step,
            weight_fn=weight_fn,
            fedprox_mu=fedprox_mu,
            **common_kwargs,
        )
```

### Anti-Patterns to Avoid

- **Modifying N_RANKED_FEATURES as a global constant for time encoding:** This would break all existing weight files and all evaluation runner inference. Use a flag-gated per-env approach instead.
- **Applying FedProx proximal term on every step instead of only after aggregation:** The proximal term should reference the last global weights (set after FedAvg), not a moving target. store_global_weights() must be called once post-aggregation.
- **Computing cooperative reward after observations but before obs cleaning:** NaN/Inf cleanup in _observe() happens after reward computation. Ensure neighbor obs used in cooperative reward are also cleaned. Solution: compute cooperative reward from local_rewards (scalar floats), not raw obs.
- **Passing env_config extras through trainer_kwargs:** The existing env_config_fn() in BaseTrainer builds env_config from self.* attributes. New parameters should be stored as self.alpha, self.time_of_day, etc. — not buried in trainer_kwargs dict which is for RLLib algorithm-level config.
- **Mid-episode route file changes:** SUMO does not support swapping the route file via TraCI mid-episode without restarting. Time-of-day demand must be pre-generated at episode reset via rand_routes().

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FedProx gradient computation | Custom backprop/optimizer | PyTorch autograd via loss() override | Adding proximal_term to total_loss is sufficient; autograd handles it |
| Neighbor graph construction | Custom adjacency from TraCI | `tls_hub.tls_graph` (already built from net.xml) | Already implemented in TrafficLightHub.get_tls_graph() |
| Route file XML merging | Custom XML writer | Python stdlib xml.etree.ElementTree | Already imported in kernel.py; extend generate_random_routes() |
| Time encoding math | Custom trig | Python math.sin/math.cos + numpy | Stdlib; no new deps needed |
| Policy weight access | Direct model state dict introspection | `policy.get_weights()` / `policy.set_weights()` | RLLib API; already used in fedavg() |

**Key insight:** The FedProx proximal term sounds complex but reduces to five lines of PyTorch: iterate named_parameters(), compute squared diff against stored globals, accumulate, add to base loss. Do not over-engineer this.

---

## Common Pitfalls

### Pitfall 1: Policy Type Not Used in on_policy_setup()
**What goes wrong:** FedPolicyTrainer.on_policy_setup() builds the policies dict using `self.policy_type`, but `self.policy_type` is set by `BaseTrainer.__load_policy_type()` which always sets it to PPOTorchPolicy. Even if you set `self.policy_type = FedProxPPOTorchPolicy` after super().__init__(), the policies dict passed to PPOConfig uses the type from `on_policy_setup()` which runs at train time.
**How to avoid:** Set `self.policy_type = FedProxPPOTorchPolicy` in FedPolicyTrainer.__init__() AFTER calling super().__init__(). Verify: on_policy_setup() is called in train() (not __init__), so the override timing works.
**Warning signs:** Training runs without error but FedProxPPOTorchPolicy.loss() is never called (log a debug message in it to verify).

### Pitfall 2: Global Weights Stored Before set_weights() Completes
**What goes wrong:** If store_global_weights() is called before the new aggregated params are fully applied via set_weights(), the "global" reference point is the old local weights — proximal term pulls toward wrong target.
**How to avoid:** Call in order: (1) new_params = self.fedavg(policy_dict), (2) policy.set_weights(new_params), (3) policy.store_global_weights(). This sequence is in on_data_recording_step().
**Warning signs:** FedProx with high mu shows no difference vs FedAvg — proximal term is effectively zero because global and local weights are the same.

### Pitfall 3: Observation Space Mismatch Crashes Ray
**What goes wrong:** If SumoEnv.observation_space returns a Box of shape (16,) but on_policy_setup() is called first (which calls dummy_env._observe() returning shape-14 arrays), Ray will silently use the wrong space or crash with shape errors.
**How to avoid:** Ensure dummy_env in on_policy_setup() has the same env_config (including use_time_encoding flag) as the training env. The env_config_fn() must include the flag.
**Warning signs:** `ValueError: observation has wrong shape` or silent policy learning failures with time encoding on.

### Pitfall 4: Existing Trained Weights Broken by Feature Count Change
**What goes wrong:** The evaluation runner loads .pkl weight files and calls policy.set_weights(). If weights were trained with 14 features and the policy now has 16 inputs, the linear layer weight tensor shapes don't match.
**How to avoid:** use_time_encoding is always opt-in (default False). New weights trained with time encoding must be stored under a different filename (e.g., ranked_timeenc.pkl). The evaluation runner must know which feature count was used to load correctly.
**Warning signs:** `RuntimeError: size mismatch for model.0.weight: copying a param with shape [64, 16] from checkpoint, the shape in current model is [64, 14]`.

### Pitfall 5: Alpha Cooperative Reward Breaks Existing Evaluation
**What goes wrong:** If alpha is embedded in env_config and the evaluation runner doesn't pass it, evaluation episodes run with default alpha (which may differ from training alpha). This is not a crash but produces misleading metrics.
**How to avoid:** Store alpha in the evaluation request alongside the other training params. Or: make alpha=1.0 (pure local) the default everywhere, so unannotated existing evaluations are unaffected.
**Warning signs:** Evaluation metrics for alpha=0.5 training are indistinguishable from alpha=1.0 results.

### Pitfall 6: Time-of-Day Routes Generated Only at First Reset
**What goes wrong:** AbstractSumoEnv.rand_routes() is called from reset() only when `rand_routes_on_reset=True` OR on the first reset. If time_of_day is enabled but rand_routes_on_reset defaults to False, the time-of-day route file is only generated once and all subsequent episodes use the same file.
**How to avoid:** When time_of_day=True, force rand_routes_on_reset=True in env_config. Add an assertion or log warning in SumoEnv.__init__() when time_of_day is True but rand_routes_on_reset is False.

---

## Code Examples

### EXT-01: Accessing Policy Weights in RLLib Old API Stack

```python
# Source: ray/rllib/algorithms/ppo/ppo_torch_policy.py + verified by reading existing fedavg() code
# Existing pattern in FedPolicyTrainer.fedavg():
policy = self.ray_trainer.get_policy(policy_id)
weights = policy.get_weights()  # returns OrderedDict of numpy arrays

# For FedProx, access PyTorch model parameters directly:
policy = self.ray_trainer.get_policy(policy_id)
# policy.model is the ModelV2 instance with .named_parameters()
global_snapshot = {
    k: v.detach().clone()
    for k, v in policy.model.state_dict().items()
}
```

### EXT-01: Proximal Term Computation

```python
# Source: FedProx paper (Li et al., MLSys 2020) + PyTorch forum verification
proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
for name, param in model.named_parameters():
    if name in self._global_weights:
        g = self._global_weights[name].to(param.device)
        proximal_term = proximal_term + (self._fedprox_mu / 2.0) * torch.sum((param - g) ** 2)
total_loss = base_loss + proximal_term
```

### EXT-02: tls_graph Usage

```python
# Source: read from BackEnd/seal/sumo/kernel/trafficlight/hub.py
# tls_graph is Dict[str, List[str]]: maps TLS id -> list of neighbor TLS ids
graph = self.kernel.tls_hub.tls_graph
neighbors_of_A = graph.get("A", [])  # ["B", "D"] for grid-3x3
```

### EXT-03: Route Generation Call Site

```python
# Source: BackEnd/seal/sumo/utils/random_routes.py (verified)
# Existing call in AbstractSumoEnv.rand_routes():
generate_random_routes(
    netfile=netfile,
    path=self.path,
    number_of_lanes=self.num_of_lanes,
    vehicles_per_lane_per_hour=360,   # <-- this is the param to vary
    end_time=self.horizon or DEFAULT_END_TIME,
    seed=self.env_seed,
)
# Time-of-day: pass vplph from a schedule dict based on episode number or time segment
```

### EXT-04: Observation Space Override in SumoEnv

```python
# Source: observation_space property pattern in env.py (verified)
@property
def observation_space(self) -> spaces.Space:
    first = self.kernel.tls_hub.index2id[0]
    base = self.kernel.tls_hub[first].observation_space   # Box(14,) or Box(10,)
    if getattr(self, 'use_time_encoding', False):
        n = base.shape[0] + 2
        low = np.concatenate([base.low, [-1.0, -1.0]]).astype(np.float32)
        high = np.concatenate([base.high, [1.0, 1.0]]).astype(np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    return base
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| FedAvg only (naive equal-weight aggregation) | FedAvg with pos_reward / traffic weighting (existing) | Original SEAL paper | Phase 4 adds FedProx as third option |
| Fixed-demand route generation | Configurable vplph via tuple range | Already in random_routes.py | Time-of-day extends this to structured segments |
| Flat 14-feature observation | 14-feature + optional time encoding (16) | Phase 4 EXT-04 | Allows learning of temporal patterns; breaks weight compatibility |

**Current FedAvg aggregation in codebase:** The existing fedavg() in FedPolicyTrainer performs reward-weighted model averaging. FedProx operates at the LOCAL training loss level (not at aggregation) — these are orthogonal. FedProx + naive FedAvg is the expected combination for the experiment.

---

## Open Questions

1. **FedProx mu sweep values for experiment**
   - What we know: FedProx paper uses mu in {0.001, 0.01, 0.1, 1.0}; small values (0.01) work best for most settings
   - What's unclear: Appropriate mu range for traffic signal control with PPO reward scale ~[-4, 0]
   - Recommendation: Default mu=0.01 for initial experiments; log proximal_term magnitude to verify scale relative to policy loss

2. **Time-of-day: per-episode or mid-episode?**
   - What we know: SUMO cannot swap route files mid-episode without restart; rand_routes() runs at reset
   - What's unclear: Whether the research goal requires within-episode demand variation or episode-level variation
   - Recommendation: Implement as episode-level variation first (different vplph per episode in curriculum); document that mid-episode would require SUMO restart which defeats the purpose

3. **Cooperative reward alpha and convergence stability**
   - What we know: Alpha=0 (fully cooperative) can cause instability if neighbor rewards are noisy; alpha=0.5 is common in literature
   - What's unclear: Whether the existing reward scale (negative squared occupancy) averages well across neighbors
   - Recommendation: Default alpha=0.5 for initial experiments; include alpha=1.0 (baseline) and alpha=0.0 in evaluation sweep

4. **Evaluation runner compatibility with new params**
   - What we know: Phase 3 evaluation runner (run_trial) doesn't accept alpha/use_time_encoding
   - What's unclear: How tightly evaluation in Phase 3 couples to the env_config
   - Recommendation: Phase 4 plan must include a task to update run_trial() to accept and pass through alpha/use_time_encoding

---

## Sources

### Primary (HIGH confidence)
- `BackEnd/seal/trainer/fed_agent.py` — FedPolicyTrainer code read directly; fedavg(), on_data_recording_step(), policy_type understood
- `BackEnd/seal/trainer/base.py` — BaseTrainer code read directly; __load_policy_type(), init_config(), env_config_fn() understood
- `BackEnd/seal/sumo/env.py` — SumoEnv code read directly; _get_reward(), _observe(), step() understood
- `BackEnd/seal/sumo/config.py` — Feature index constants and N_RANKED_FEATURES=14 verified directly
- `BackEnd/seal/sumo/kernel/trafficlight/hub.py` — tls_graph construction verified directly
- `BackEnd/seal/sumo/kernel/trafficlight/light.py` — get_observation() array construction verified
- `BackEnd/seal/sumo/utils/random_routes.py` — generate_random_routes() signature and vehicles_per_lane_per_hour parameter verified
- `BackEnd/api/routes/train.py` — TrainRequest model read; existing params understood
- `BackEnd/api/training_runner.py` — create_trainer() interface read
- `https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo_torch_policy.py` — PPOTorchPolicy class structure and loss() method signature verified via WebFetch

### Secondary (MEDIUM confidence)
- PyTorch forum: FedProx loss implementation pattern — `(mu/2) * torch.sum((w - w_global)**2)` verified by moderator
- WebSearch: Ray RLLib old API stack supports PPOTorchPolicy subclassing with loss() override; custom loss can be added after super().loss()

### Tertiary (LOW confidence)
- WebSearch: Literature on cooperative reward alpha values for traffic signal control (0.5 as common starting point)
- WebSearch: Time-of-day AM/midday/PM vplph values (600/240/540) — representative but not authoritative for this specific network

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies, all existing
- FedProx subclass pattern: MEDIUM — PPOTorchPolicy loss() override mechanism verified; exact behavior of super().loss() returning a scalar vs list not verified for all Ray versions
- Cooperative reward: HIGH — tls_graph, _get_reward(), and obs dict all read directly from source
- Time-of-day routes: HIGH — generate_random_routes() signature fully read; approach is mechanical extension
- Time encoding obs space: HIGH — space.py and get_observation() fully read; override approach is sound; weight incompatibility pitfall is real and verified
- API wiring: HIGH — train.py and training_runner.py fully read; extension pattern is clear

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable Ray/SUMO APIs; low churn risk)
