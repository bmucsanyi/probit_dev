"""Cross-entropy loss combined with Bayesian Model Averaging."""

import torch
import torch.nn.functional as F
from torch import nn

from probit.utils.predictive import get_predictive


class RegularizedBMACrossEntropyLoss(nn.Module):
    """Implements a regularized Cross-entropy loss with Bayesian Model Averaging."""

    def __init__(
        self, predictive, use_correction, num_mc_samples, regularization_factor
    ):
        super().__init__()

        if not predictive.startswith("softmax"):
            msg = "Invalid predictive provided"
            raise ValueError(msg)

        self._regularization_factor = regularization_factor

        self.predictive_str = predictive

        if predictive.endswith("mc"):
            self.loss = nn.NLLLoss()
        else:
            self.predictive = get_predictive(
                predictive, use_correction, num_mc_samples, approximate=False
            )
            self.loss = nn.CrossEntropyLoss()

        self.eps = 1e-10
        self.num_mc_samples = num_mc_samples

    def forward(self, logits, targets):
        if len(logits) == 2:
            mean, var = logits

            regularizer = mean.exp().sum(dim=-1).sub(1).square().mean()

            if self.predictive_str.endswith("mc"):
                logits = (
                    var.sqrt()
                    * torch.randn(
                        var.shape[0],
                        self.num_mc_samples,
                        var.shape[1],
                        device=var.device,
                    )
                    + mean
                )
            else:
                logits = self.predictive(mean, var, return_logits=True)
                return (
                    self.loss(logits, targets)
                    + self._regularization_factor * regularizer
                )
        else:
            regularizer = logits[0].mean(dim=1).exp().sum(dim=-1).sub(1).square().mean()

        logits = logits[0]
        log_probs = F.softmax(logits, dim=-1).mean(dim=1).add(self.eps).log()  # [B, C]

        return self.loss(log_probs, targets) + self._regularization_factor * regularizer
