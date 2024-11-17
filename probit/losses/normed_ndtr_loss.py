"""Backpack extension for NormedNdtrNLLLoss."""

import torch
import torch.nn.functional as F
from backpack.core.derivatives.nll_base import NLLLossDerivatives
from backpack.extensions.secondorder.hbp.losses import HBPLoss
from torch import Tensor
from torch.distributions import Categorical

from probit.losses.predictive import log_normed_ndtr, ndtr


class HBPNormedNdtrNLLLoss(HBPLoss):
    """Hessian backpropagation for the ``NormedNdtrNLLLoss`` layer."""

    def __init__(self):
        """Pass derivatives for ``NormedNdtrNLLLoss``."""
        super().__init__(derivatives=NormedNdtrNLLLossDerivatives())


class NormedNdtrNLLLoss(torch.nn.modules.loss._Loss):
    """Normed sigmoid NLL loss implementation."""

    def __init__(self):
        super().__init__()

        self.log_act_fn = log_normed_ndtr

    def forward(self, logit, target):
        return -self.log_act_fn(logit)[torch.arange(target.shape[0]), target].mean()


class NormedNdtrNLLLossDerivatives(NLLLossDerivatives):
    """Derivatives of the NormedNdtrNLLLoss."""

    def __init__(self):
        """Initialization for NormedNdtrNLLLoss derivative."""
        super().__init__(use_autograd=False)

    def _verify_support(self, module: NormedNdtrNLLLoss):
        """Verification of module support for NormedNdtrNLLLoss.

        Args:
            module: NormedNdtrNLLLoss module
        """
        self._check_input_dims(module)

    @staticmethod
    def _check_input_dims(module: NormedNdtrNLLLoss):
        """Raises an exception if the shapes of the input are not supported.

        Args:
            module: NormedNdtrNLLLoss module

        Raises:
            NotImplementedError: if input is not a batch of scalars.
        """
        if module.input0.dim() != 2:
            msg = "Only 2D inputs are currently supported"
            raise NotImplementedError(msg)

    @staticmethod
    def _make_distribution(subsampled_input: Tensor) -> Categorical:
        """Make the sampling distribution for the NLL loss form of BCEWithLogits.

        Args:
            subsampled_input: input after subsampling

        Returns:
            Categorical distribution with probabilities from the subsampled_input.
        """
        elementwise_probs = ndtr(subsampled_input)
        probs = elementwise_probs / elementwise_probs.sum(dim=-1, keepdim=True)

        return Categorical(probs=probs)

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        return input.shape[0]

    @staticmethod
    def hessian_is_psd() -> bool:
        """Return whether the Hessian is PSD.

        Let fₙ ∈ ℝ be the input and yₙ ∈ [0; 1] be the label, and σ(fₙ) ∈ (0;
        1) be the sigmoid probability. The Hessian ∇²ℓ(fₙ, yₙ) w.r.t. fₙ is
        ∇²ℓ(fₙ, yₙ) = σ'(fₙ) = σ(fₙ) (1 - σ(fₙ)) > 0. Hence, the Hessian is PSD.

        Returns:
            True
        """
        return True

    def _compute_sampled_grads_manual(
        self, subsampled_input: torch.Tensor, mc_samples: int
    ) -> torch.Tensor:
        # probs
        probs = ndtr(subsampled_input)
        expand_dims = [mc_samples] + probs.dim() * [-1]
        probs_unsqeezed = probs.unsqueeze(0).expand(*expand_dims)  # [V N C D1 D2]

        # norm probs
        norm_probs = probs / probs.sum(dim=1, keepdim=True)
        norm_probs_unsqeezed = norm_probs.unsqueeze(0).expand(
            *expand_dims
        )  # [V N C D1 D2]

        # normal pdf
        normal = torch.distributions.Normal(0, 1)
        pdf_vals = normal.log_prob(subsampled_input).exp()
        pdf_vals_unsqeezed = pdf_vals.unsqueeze(0).expand(*expand_dims)  # [V N C D1 D2]

        # labels
        distribution = self._make_distribution(subsampled_input)
        samples = distribution.sample(torch.Size([mc_samples]))  # [V N D1 D2]
        samples_onehot = F.one_hot(samples, num_classes=probs.shape[1])  # [V N D1 D2 C]
        samples_onehot_rearranged = torch.einsum("vn...c->vnc...", samples_onehot).to(
            probs.dtype
        )  # [V N C D1 D2]

        return (
            pdf_vals_unsqeezed
            * (norm_probs_unsqeezed - samples_onehot_rearranged)
            / probs_unsqeezed
        )
