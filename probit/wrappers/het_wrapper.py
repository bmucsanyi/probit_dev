"""HET implementation as a wrapper class.

Heteroscedastic Gaussian sampling based on https://github.com/google/uncertainty-baselines.
"""

import torch
import torch.nn.functional as F
from torch import nn

from probit.wrappers.model_wrapper import DistributionalWrapper


class HETHead(nn.Module):
    """Classification head for the HET method."""

    def __init__(
        self,
        matrix_rank,
        num_mc_samples,
        num_features,
        num_classes,
        temperature,
        classifier,
    ):
        super().__init__()
        self._matrix_rank = matrix_rank
        self._num_mc_samples = num_mc_samples
        self._num_features = num_features
        self._num_classes = num_classes

        if self._matrix_rank > 0:
            self._low_rank_cov_layer = nn.Linear(
                in_features=self._num_features,
                out_features=self._num_classes * self._matrix_rank,
            )
        self._diagonal_var_layer = nn.Linear(
            in_features=self._num_features, out_features=self._num_classes
        )
        self._min_scale_monte_carlo = 1e-3

        self._temperature = temperature
        self._classifier = classifier

    def forward(self, features):
        # Shape variables
        C = self._num_classes
        R = self._matrix_rank

        if R > 0:
            low_rank_cov = self._low_rank_cov_layer(features).reshape(
                -1, C, R
            )  # [B, C, R]

        diagonal_var = (
            F.softplus(self._diagonal_var_layer(features)) + self._min_scale_monte_carlo
        )  # [B, C]

        if R > 0:
            # Compute full covariance matrix: Cov = L @ L^T + D
            # where L is low_rank_cov [B, C, R] and D is diagonal_var [B, C]
            cov = torch.bmm(low_rank_cov, low_rank_cov.transpose(1, 2))  # [B, C, C]
            # Add diagonal variance
            cov = cov + torch.diag_embed(diagonal_var)  # [B, C, C]
            return self._classifier(
                features
            ) / self._temperature, cov / self._temperature**2

        vars = diagonal_var
        return self._classifier(
            features
        ) / self._temperature, vars / self._temperature**2


class HETWrapper(DistributionalWrapper):
    """This module takes a model as input and creates a HET model from it."""

    def __init__(
        self,
        model: nn.Module,
        matrix_rank: int,
        num_mc_samples: int,
        temperature: float,
    ):
        super().__init__(model)

        self._matrix_rank = matrix_rank
        self._num_mc_samples = num_mc_samples
        self._temperature = temperature

        self._classifier = HETHead(
            matrix_rank=self._matrix_rank,
            num_mc_samples=self._num_mc_samples,
            num_features=self.num_features,
            num_classes=self.num_classes,
            temperature=self._temperature,
            classifier=self.model.get_classifier(),
        )

    def get_classifier(self):
        return self._classifier

    def reset_classifier(
        self,
        matrix_rank: int | None = None,
        num_mc_samples: int | None = None,
        temperature: float | None = None,
        *args,
        **kwargs,
    ):
        if matrix_rank is not None:
            self._matrix_rank = matrix_rank

        if num_mc_samples is not None:
            self._num_mc_samples = num_mc_samples

        if temperature is not None:
            self._temperature = temperature

        self.model.reset_classifier(*args, **kwargs)
        self._classifier = HETHead(
            matrix_rank=self._matrix_rank,
            num_mc_samples=self._num_mc_samples,
            num_features=self.num_features,
            num_classes=self.num_classes,
            temperature=self._temperature,
            classifier=self.model.get_classifier(),
        )
