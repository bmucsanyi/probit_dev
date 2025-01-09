"""Cross-entropy loss combined with Bayesian Model Averaging."""

from torch import nn

from probit.utils.predictive import get_predictive

# TODO(bmucsanyi): Implement MC version of log_link if needed


class RegularizedBMACrossEntropyLoss(nn.Module):
    """Implements a regularized Cross-entropy loss with Bayesian Model Averaging."""

    def __init__(self, regularization_factor):
        super().__init__()

        self._regularization_factor = regularization_factor

        self.predictive = get_predictive(
            predictive="softmax_mean_field",
            use_correction=False,
            num_mc_samples=0,
            approximate=False,
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        mean, var = logits

        regularizer = mean.exp().sum(dim=-1).sub(1).square().mean()

        logits = self.predictive(mean, var, return_logits=True)
        return self.loss(logits, targets) + self._regularization_factor * regularizer
