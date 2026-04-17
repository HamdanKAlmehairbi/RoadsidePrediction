"""PPO policy with federated distillation loss.

Adds KL-divergence regularization toward a consensus action distribution
that is computed by the FedDistill trainer after each aggregation step.
Instead of sharing raw model weights (FedAvg), agents share their action
probability distributions (logits) and align toward a consensus — sharing
KNOWLEDGE not WEIGHTS.
"""
import numpy as np
import torch

from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from typing import Union, List, Type


class FedDistillPPOTorchPolicy(PPOTorchPolicy):
    """PPO policy with KL loss toward consensus logits.

    When consensus_logits is set and distill_lambda > 0, the loss becomes:
        L = L_ppo + lambda * KL(current_dist || consensus_dist)

    When consensus_logits is None (before first aggregation), the loss
    is identical to standard PPOTorchPolicy (zero overhead).
    """

    def __init__(self, *args, **kwargs):
        # Initialize distillation state BEFORE super().__init__() because
        # TorchPolicyV2.__init__() calls loss() during setup, and loss()
        # references self._distill_lambda. Setting after super() would cause
        # AttributeError.
        self._distill_lambda = 0.1
        self._consensus_logits = None
        super().__init__(*args, **kwargs)

    def set_distill_lambda(self, lam: float) -> None:
        """Set the KL loss weight (lambda)."""
        self._distill_lambda = lam

    def set_consensus_logits(self, logits: np.ndarray) -> None:
        """Store consensus action distribution for KL loss."""
        self._consensus_logits = logits

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """PPO loss + optional KL divergence toward consensus distribution."""
        base_loss = super().loss(model, dist_class, train_batch)

        if self._consensus_logits is not None and self._distill_lambda > 0:
            # Get current policy's action distribution
            model_out, _ = model(train_batch)
            curr_dist = dist_class(model_out, model)

            # Create consensus distribution
            consensus_tensor = torch.tensor(
                self._consensus_logits, dtype=torch.float32
            ).to(model_out.device)
            # Expand to match batch size
            if consensus_tensor.dim() == 1:
                consensus_tensor = consensus_tensor.unsqueeze(0).expand_as(
                    model_out
                )
            consensus_dist = dist_class(consensus_tensor, model)

            # KL(current || consensus)
            kl = torch.distributions.kl_divergence(
                curr_dist.dist, consensus_dist.dist
            ).mean()

            return base_loss + self._distill_lambda * kl

        return base_loss
