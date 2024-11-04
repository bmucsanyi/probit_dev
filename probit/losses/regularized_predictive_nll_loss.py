"""Regularized predictive NLL loss for multiclass classification."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.special import ndtr

from probit.utils.ndtr import ndtr_approx
from probit.utils.predictive import get_predictive


class RegularizedPredictiveNLLLoss(nn.Module):
    """Regularized predictive NLL loss."""

    def __init__(
        self,
        predictive,
        use_correction,
        num_mc_samples,
        regularization_factor,
        approximate,
    ):
        super().__init__()

        if not predictive.startswith(("probit", "logit")):
            msg = "Invalid predictive provided"
            raise ValueError(msg)

        self._predictive = get_predictive(
            predictive, use_correction, num_mc_samples, approximate
        )

        if predictive.startswith("probit"):
            self._activation = ndtr_approx if approximate else ndtr
        else:
            self._activation = F.sigmoid

        self._regularization_factor = regularization_factor

    def forward(self, logits, targets):
        preds = self._predictive(logits)
        loss = (
            -preds[torch.arange(0, preds.shape[-1]), targets]
            .log()
            .clamp(torch.finfo(preds.dtype).min)
        ).mean() + self._regularization_factor * self._activation(
            logits.double()
        ).float().sum(dim=-1).sub(1).square().mean()

        return loss
