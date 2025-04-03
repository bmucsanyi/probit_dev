"""Exp + NLL loss for multiclass classification."""

import torch
from torch import nn


class RegularizedSoftmaxNLLLoss2(nn.Module):
    """Regularized softmax + NLL loss encouraging sum(exp(logits)) approx. c."""

    def __init__(self, regularization_factor):
        super().__init__()
        self._regularization_factor = regularization_factor
        self._loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        loss = self._loss(logits, targets)
        regularizer = torch.logsumexp(logits, dim=-1).square()

        return loss + self._regularization_factor * regularizer
