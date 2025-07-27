"""EDL loss."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EDLLoss(nn.Module):
    """Implements the Evidential Deep Learning (EDL) loss.

    This class calculates the EDL loss, which includes error, variance, and
    Kullback-Leibler divergence terms. The KL term is annealed over time.

    Args:
        num_batches: The number of batches per epoch.
        num_classes: The number of classes in the classification task.
        start_epoch: The epoch at which to start including the KL divergence term.
        scaler: A scaling factor for the KL divergence term.
    """

    def __init__(
        self, num_batches: int, num_classes: int, start_epoch: int, scaler: float
    ) -> None:
        super().__init__()

        self.curr_batch = 0
        self.curr_step = 0
        self.max_step = 10 * num_batches
        self.num_batches = num_batches
        self.curr_epoch = 0
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.register_buffer("uniform_alphas", torch.ones((num_classes,)))  # [C]
        self.register_buffer(
            "sum_uniform_alphas", torch.tensor(num_classes, dtype=torch.float32)
        )  # []
        self.register_buffer(
            "log_b_uniform_alphas",
            torch.lgamma(self.uniform_alphas).sum()
            - torch.lgamma(self.sum_uniform_alphas),
        )  # []

    def kullback_leibler_term(self, alpha_tildes: Tensor) -> Tensor:
        """Calculates the Kullback-Leibler divergence term of the EDL loss.

        Args:
            alpha_tildes: The modified alpha values.

        Returns:
            The Kullback-Leibler divergence term.
        """
        sum_alpha_tildes = alpha_tildes.sum(dim=1)  # [B]
        log_b_alpha_tildes = torch.lgamma(alpha_tildes).sum(dim=1) - torch.lgamma(
            sum_alpha_tildes
        )  # [B]

        digamma_sum_alpha_tildes = torch.digamma(sum_alpha_tildes)  # [B]
        digamma_alpha_tildes = torch.digamma(alpha_tildes)  # [B, C]

        kullback_leibler_term = (
            alpha_tildes.sub(self.uniform_alphas)  # [B, C]
            .mul(
                digamma_alpha_tildes.sub(digamma_sum_alpha_tildes.unsqueeze(1))
            )  # [B, C]
            .sum(dim=1)  # [B]
            .sub(log_b_alpha_tildes)  # [B]
            .add(self.log_b_uniform_alphas)  # [B]
        )

        return kullback_leibler_term  # [B]

    def forward(self, alphas: tuple[Tensor], targets: Tensor) -> Tensor:
        """Calculates the EDL loss.

        Args:
            alphas: The predicted alpha values.
            targets: The target class indices.

        Returns:
            The calculated EDL loss.
        """
        alphas = alphas[0]
        sum_alphas = alphas.sum(dim=1)  # [B]
        mean_alphas = alphas.div(sum_alphas.unsqueeze(1))  # [B, C]

        targets_one_hot = F.one_hot(targets, num_classes=alphas.shape[1])  # [B, C]

        error_term = mean_alphas.sub(targets_one_hot).square().sum(dim=1)  # [B]
        variance_term = (
            mean_alphas.mul(1 - mean_alphas).sum(dim=1).div(sum_alphas + 1)
        )  # [B]

        loss = error_term + variance_term

        if self.curr_epoch >= self.start_epoch:
            annealing_coefficient = min(1, self.curr_step / self.max_step)
            alpha_tildes = alphas.sub(1).mul(1 - targets_one_hot).add(1)  # [B, C]
            kullback_leibler_term = self.kullback_leibler_term(alpha_tildes)  # [B]

            loss += annealing_coefficient * self.scaler * kullback_leibler_term

            self.curr_step += 1

        self.curr_batch += 1
        if self.curr_batch == self.num_batches - 1:
            self.curr_epoch += 1
            self.curr_batch = 1

        return loss.mean()  # []
