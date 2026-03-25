"""FedProx PPO policy subclass with proximal loss term.

Adds a proximal regularization term to the standard PPO loss that penalizes
local model weights for drifting too far from the global (federated) model.
This improves convergence on heterogeneous clients (different intersection types).

Reference: Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
"""

import torch

from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from typing import Union, List, Type


class FedProxPPOTorchPolicy(PPOTorchPolicy):
    """PPOTorchPolicy with FedProx proximal term in the loss function.

    When fedprox_mu > 0 and global weights are stored, the loss becomes:
        L = L_ppo + (mu / 2) * sum_i || w_i - w_global_i ||^2

    When fedprox_mu == 0 or global weights are not yet stored, the loss
    is identical to standard PPOTorchPolicy (zero overhead).
    """

    def __init__(self, *args, **kwargs):
        # Initialize FedProx state BEFORE super().__init__() because
        # TorchPolicyV2.__init__() calls loss() during setup, and loss()
        # references self._fedprox_mu. Setting after super() would cause
        # AttributeError: 'FedProxPPOTorchPolicy' object has no attribute '_fedprox_mu'
        self._global_weights = None
        self._fedprox_mu = 0.0
        super().__init__(*args, **kwargs)

    def set_fedprox_mu(self, mu: float) -> None:
        """Set the proximal term coefficient (mu)."""
        self._fedprox_mu = mu

    def store_global_weights(self) -> None:
        """Snapshot current model weights as the global reference.

        Must be called AFTER set_weights() with freshly aggregated params
        so the proximal term pulls toward the latest global model.
        """
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
        """PPO loss + optional FedProx proximal term."""
        base_loss = super().loss(model, dist_class, train_batch)

        if self._fedprox_mu > 0.0 and self._global_weights is not None:
            proximal_term = torch.tensor(
                0.0, device=next(model.parameters()).device
            )
            for name, param in model.named_parameters():
                if name in self._global_weights:
                    global_w = self._global_weights[name].to(param.device)
                    proximal_term = proximal_term + torch.sum(
                        (param - global_w) ** 2
                    )
            proximal_term = (self._fedprox_mu / 2.0) * proximal_term
            return base_loss + proximal_term

        return base_loss
