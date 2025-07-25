"""Cross-entropy loss combined with Bayesian Model Averaging."""

import torch
import torch.nn.functional as F
from torch import nn

from probit.utils.predictive import get_predictive


class BMACrossEntropyLoss(nn.Module):
    """Implements Cross-entropy loss combined with Bayesian Model Averaging."""

    def __init__(self, predictive, use_correction, num_mc_samples, approximate=False):
        super().__init__()

        self.predictive_str = predictive
        self.predictive = get_predictive(
            predictive, use_correction, num_mc_samples, approximate
        )

        if predictive.endswith("mc"):
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.eps = 1e-10
        self.num_mc_samples = num_mc_samples

    def forward(self, logits, targets):
        if len(logits) == 2:
            mean, var = logits

            if self.predictive_str.endswith("mc"):
                # Use the predictive function which handles full covariance properly
                probs = self.predictive(mean, var)  # [B, C]
                log_probs = probs.add(self.eps).log()
                return self.loss(log_probs, targets)
            else:
                logits = self.predictive(mean, var, return_logits=True)
                return self.loss(logits, targets)

        # For models that return samples directly (e.g., deep ensembles)
        logits = logits[0]
        log_probs = F.softmax(logits, dim=-1).mean(dim=1).add(self.eps).log()  # [B, C]
        return self.loss(log_probs, targets)
