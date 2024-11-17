"""Unnormalized predictive NLL loss for multiclass classification."""

from functools import partial

from torch import nn

from probit.losses.normcdf_nll_loss import NormCDFNLLLoss
from probit.losses.predictive import PREDICTIVE_DICT
from probit.losses.sigmoid_nll_loss import SigmoidNLLLoss


class UnnormalizedPredictiveNLLLoss(nn.Module):
    """Unnormalized predictive NLL loss."""

    def __init__(self, predictive, approximate):
        super().__init__()

        if not predictive.startswith(("probit", "logit")) or predictive.endswith("mc"):
            msg = "Invalid predictive provided"
            raise ValueError(msg)

        self._predictive = PREDICTIVE_DICT[predictive]

        if predictive.startswith("probit"):
            self._predictive = partial(self._predictive, approximate=approximate)

        self._loss = (
            NormCDFNLLLoss(approximate=approximate)
            if predictive.startswith("probit")
            else SigmoidNLLLoss()
        )

    def forward(self, logits, targets):
        preds = self._predictive(*logits, return_logits=True)
        loss = self._loss(preds, targets)

        return loss
