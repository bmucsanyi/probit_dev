"""Exp + NLL loss for multiclass classification."""

from torch import nn


class RegularizedSoftmaxNLLLoss(nn.Module):
    """Regularized softmax + NLL loss encouraging sum(exp(logits)) approx. 1."""

    def __init__(self, regularization_factor):
        super().__init__()
        self._regularization_factor = regularization_factor
        self._loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        loss = self._loss(logits, targets)
        regularizer = logits.exp().sum(dim=-1).sub(1).square().mean()

        return loss + self._regularization_factor * regularizer
