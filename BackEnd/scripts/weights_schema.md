# Weight File Schema Reference

Source-inspection of `BackEnd/seal/trainer/*.py` save paths. Use for building the Ray-free inference loop.

## Payload shapes

| Trainer | Top-level payload |
|---------|-------------------|
| SARL | flat `state_dict` |
| FedRL | flat `state_dict` (global model after aggregation) |
| MARL | `{"__multi_policy__": True, "policies": {agent_id: state_dict}}` |
| Gossip | same as MARL |
| HierFed | same as MARL |
| FedDistill | same as MARL (KL buffer keys can be ignored for inference) |
| MeanField | same as MARL |
| CTDE | `{"__multi_policy__": True, "__ctde__": True, "policies": {agent_id: state_dict}}` |

## Tensor naming (RLlib TorchModelV2 convention)

Each `state_dict` (per-agent or global) contains:

```
_hidden_layers.0._model.0.weight    # (hidden[0], obs_dim)
_hidden_layers.0._model.0.bias      # (hidden[0],)
_hidden_layers.1._model.0.weight    # (hidden[1], hidden[0])
_hidden_layers.1._model.0.bias      # (hidden[1],)
_logits._model.0.weight             # (action_dim, hidden[-1])
_logits._model.0.bias               # (action_dim,)

# Not needed for inference — value head:
_value_branch_separate.0._model.0.*
_value_branch_separate.1._model.0.*
_value_branch._model.0.*
```

## Dimensions per trainer (grid-3x3)

| Trainer | obs_dim | action_dim | hidden | Notes |
|---------|:---:|:---:|:---:|-------|
| SARL | 14 | 2 | [256, 256] | ranked obs |
| MARL | 14 | 2 | [256, 256] | |
| FedRL | 14 | 2 | [256, 256] | global model only |
| Gossip | 14 | 2 | [256, 256] | |
| HierFed | 14 | 2 | [256, 256] | |
| FedDistill | 14 | 2 | [256, 256] | KL buffers ignorable |
| MeanField | 15 | 2 | [256, 256] | +1 dim for mean neighbor action |
| CTDE | 140 | 2 | [256, 256] | 14 (local) + 9×14 (neighbors) concat; at eval, pad the 9×14 neighbor portion with zeros |

## Agent keys

For grid-3x3, multi-agent trainers produce 9 sub-policies with keys:
`A0, A1, A2, B0, B1, B2, C0, C1, C2`

All sub-policies have identical shape within a single trainer run.

## Inference pseudocode

```python
def load_policy(pkl_path):
    obj = pickle.load(open(pkl_path, "rb"))
    if isinstance(obj, dict) and obj.get("__multi_policy__"):
        ctde = bool(obj.get("__ctde__"))
        return {"multi": True, "ctde": ctde, "policies": obj["policies"]}
    return {"multi": False, "ctde": False, "policy": obj}

class PPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, action_dim)

    def forward(self, obs):
        return self.head(self.trunk(obs))

def rllib_to_actor_state_dict(src):
    return {
        "trunk.0.weight": src["_hidden_layers.0._model.0.weight"],
        "trunk.0.bias":   src["_hidden_layers.0._model.0.bias"],
        "trunk.2.weight": src["_hidden_layers.1._model.0.weight"],
        "trunk.2.bias":   src["_hidden_layers.1._model.0.bias"],
        "head.weight":    src["_logits._model.0.weight"],
        "head.bias":      src["_logits._model.0.bias"],
    }
```

For CTDE at eval: local observation (14 dims) padded with `np.zeros(126, dtype=np.float32)` to reach 140 dims.

For MeanField at eval: append 1 additional feature to the 14-dim obs = mean of neighbor discrete actions from the previous step.

## Gotchas
- RLlib activation is Tanh by default — matches PPOConfig.
- FedDistill may include extra buffers for KL consensus; ignore all keys not in the list above.
- Agent keys from the pickle must map to TLS ids reported by SUMO. For grid-3x3 the labels already match SUMO's default naming.
