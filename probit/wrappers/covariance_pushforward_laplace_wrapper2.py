"""KFAC covariance pushforward Laplace wrapper."""

import gc
import logging
import re
import time

import numpy as np
import torch
from backpack import backpack, extend
from backpack.extensions import KFAC
from einops import rearrange
from torch import Tensor
from torch.autograd import grad

from probit.losses.normed_ndtr_loss import HBPNormedNdtrNLLLoss, NormedNdtrNLLLoss
from probit.losses.normed_sigmoid_loss import (
    HBPNormedSigmoidNLLLoss,
    NormedSigmoidNLLLoss,
)
from probit.utils.metric import calibration_error
from probit.wrappers.model_wrapper import DistributionalWrapper

logger = logging.getLogger(__name__)


def vjp(
    f: Tensor,
    x: Tensor,
    v: Tensor,
    *,
    retain_graph: bool = False,
    detach: bool = True,
) -> Tensor:
    r"""Multiply the transpose Jacobian of f w.r.t. x onto v.

    See $\text{\Cref{def:vjp}}$.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.
        v: Tensor to be multiplied with the Jacobian.
            Has same shape as f(x).
        retain_graph: If True, keep the computation graph of
            f for future differentiation. Default: False.
        detach: If True, detach the result from the
            computation graph. Default: True.

    Returns:
        Vector-Jacobian product v @ (J_x f).T with shape of x.
    """
    (result,) = grad(f, x, grad_outputs=v, retain_graph=retain_graph)
    return result.detach() if detach else result


def jvp(
    f: Tensor,
    x: Tensor,
    v: Tensor,
    *,
    retain_graph: bool = False,
    detach: bool = True,
) -> Tensor:
    r"""Multiply the Jacobian of f w.r.t. x onto v.

    See $\text{\Cref{def:jvp}}$.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.
        v: Tensor to be multiplied with the Jacobian.
            Has same shape as x.
        retain_graph: If True, keep the computation graph of
            f for future differentiation. Default: False.
        detach: If True, detach the result from the
            computation graph. Default: True.

    Returns:
        Jacobian-Vector product (J_x f) @ v with shape of f.
    """
    u = torch.zeros_like(f, requires_grad=True)
    (ujp,) = grad(f, x, grad_outputs=u, create_graph=True)
    (result,) = grad(ujp, u, grad_outputs=v, retain_graph=retain_graph)
    return result.detach() if detach else result


def jac(f: Tensor, x: Tensor) -> Tensor:
    r"""Compute the general Jacobian tensor of f w.r.t. x.

    See $\text{\Cref{def:general_jacobian}}$.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.

    Returns:
        Jacobian tensor of f w.r.t. x. Has shape
        (*f.shape, *x.shape).
    """
    J = torch.zeros(f.shape + x.shape)

    for d in torch.arange(f.numel()):
        d_unraveled = torch.unravel_index(d, f.shape)
        one_hot_d = torch.zeros_like(f)
        one_hot_d[d_unraveled] = 1.0
        J[d_unraveled] = vjp(f, x, one_hot_d, retain_graph=True)

    return J


def cvec_jac(f: Tensor, x: Tensor) -> Tensor:
    r"""Compute the Jacobian in column-flattening convention.

    See $\text{\Cref{def:cvec_jacobian}}$.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.

    Returns:
        cvec-Jacobian matrix of f w.r.t. x.
        Has shape (f.numel(), x.numel()).
    """
    J = jac(f, x)
    # flatten row indices
    J = cvec(J, end_dim=f.ndim - 1)
    # flatten column indices
    return cvec(J, start_dim=1)


def rvec_jac(f: Tensor, x: Tensor) -> Tensor:
    r"""Compute the Jacobian in row-flattening convention.

    See $\text{\Cref{def:rvec_jacobian}}$.

    Args:
        f: The result tensor of f(x).
        x: Tensor w.r.t. which f is differentiated.

    Returns:
        rvec-Jacobian matrix of f w.r.t. x.
        Has shape (f.numel(), x.numel()).
    """
    J = jac(f, x)
    # flatten row indices
    J = rvec(J, end_dim=f.ndim - 1)
    # flatten column indices
    return rvec(J, start_dim=1)


def rvec(t: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    r"""Flatten a tensor in last-varies-fastest fashion.

    See $\text{\Cref{def:rvec}}$.
    For a matrix, this corresponds to row-stacking.
    This is the common flattening scheme in code.

    Args:
        t: A tensor.
        start_dim: At which dimension to start flattening.
            Default is 0.
        end_dim: The last dimension to flatten. Default is -1.

    Returns:
        The flattened tensor.
    """
    return t.flatten(start_dim=start_dim, end_dim=end_dim)


def cvec(t: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    r"""Flatten a tensor in first-varies-fastest fashion.

    See $\text{\Cref{def:cvec}}$.
    For a matrix, this corresponds to column-stacking.
    This is the common flattening scheme in literature.

    Args:
        t: A tensor.
        start_dim: At which dimension to start flattening.
            Default is 0.
        end_dim: The last dimension to flatten. Default is -1.

    Returns:
        The flattened tensor.
    """
    end_dim = end_dim if end_dim >= 0 else end_dim + t.ndim
    # flip index order, then flatten last-varies-fastest
    before = [f"s{i}" for i in range(start_dim)]
    active = [f"a{i}" for i in range(start_dim, end_dim + 1)]
    after = [f"s{i}" for i in range(end_dim + 1, t.ndim)]
    flipped = active[::-1]

    # build equation, e.g. "s0 a0 a1 s2 -> s0 a1 a0 s2"
    in_equation = " ".join(before + active + after)
    out_equation = " ".join(before + flipped + after)
    equation = f"{in_equation} -> {out_equation}"

    return rvec(
        rearrange(t, equation),
        start_dim=start_dim,
        end_dim=end_dim,
    )


class CovariancePushforwardLaplaceWrapper2(DistributionalWrapper):
    """KFAC covariance pushforward Laplace wrapper."""

    def __init__(
        self,
        model,
        loss_fn,
        predictive_fn,
        mask_regex,
        weight_path,
    ):
        super().__init__(model)
        self._load_model(weight_path)

        if mask_regex is not None:
            for name, module in self.model.named_modules():
                if re.match(rf"^{mask_regex}$", name):
                    setattr(self.model, name, extend(module))
        else:
            self.model = extend(self.model, use_converter=True)

        self.loss_fn = extend(loss_fn)
        self.extension = KFAC()

        if isinstance(loss_fn, NormedSigmoidNLLLoss):
            self.extension.set_module_extension(
                NormedSigmoidNLLLoss, HBPNormedSigmoidNLLLoss()
            )
        elif isinstance(loss_fn, NormedNdtrNLLLoss):
            self.extension.set_module_extension(
                NormedNdtrNLLLoss, HBPNormedNdtrNLLLoss()
            )

        self.extension.set_module_extension()
        self.predictive_fn = predictive_fn
        self.mask_regex = mask_regex
        self.is_laplace_approximated = False

        if mask_regex is not None:
            self.apply_parameter_mask(mask_regex=mask_regex)

    def forward(self, batch):
        if not self.is_laplace_approximated:
            msg = "Model has to be Laplace approximated first"
            raise ValueError(msg)

        device = next(self.model.parameters()).device
        mean = torch.empty((batch.shape[0], self.model.num_classes), device=device)
        var = torch.empty((batch.shape[0], self.model.num_classes), device=device)

        for i, x in enumerate(batch):
            with torch.enable_grad():
                logit = self.model(x.unsqueeze(0))[0]
                mean[i] = logit
            result = torch.zeros((logit.shape[0],), device=logit.device)
            for param, param_kfac in zip(self.theta_0_list, self.kfac, strict=True):
                # Compute dense Jacobian
                with torch.enable_grad():
                    jacobian_T = (
                        rvec_jac(logit, param).detach().T.contiguous()
                    )  # [P, C]

                if param.ndim == 2:
                    A, B = param_kfac
                    matmat = self.kron_matmat(A, B, jacobian_T)  # [P, C]
                else:
                    A = param_kfac[0]
                    matmat = A @ jacobian_T

                result.add_((jacobian_T * matmat).sum(dim=0))  # [C]

                # Free VRAM
                del param, param_kfac, jacobian_T, A, B, matmat
                torch.cuda.empty_cache()
                gc.collect()
            var[i] = result

        return mean, var

    @staticmethod
    def mnxs_to_mxnxs(T, m, n, s):
        """Convert a tensor of shape `mn x s` to a tensor of shape `m x n x s`.

        This is the inverse operation of `mxnxs_to_mnxs`.
        """
        assert T.shape == (m * n, s)
        return T.reshape(m, n, s)

    @staticmethod
    def mxnxs_to_mnxs(T, m, n, s):
        """Convert a tensor of shape `m x n x s` to a tensor of shape `mn x s`.

        This is the inverse operation of `mnxs_to_mxnxs`.
        """
        assert T.shape == (m, n, s)
        return T.reshape(m * n, s)

    @staticmethod
    def kron_matmat(A, B, V):
        """Perform the matrix-matrix product `(A ⊗ B) @ V`.

        This is done without forming the Kronecker product explicitly.
        We use the following notation for the shapes:
        - `A` has shape `m x n`,
        - `B` has shape `p x q`,
        - `A ⊗ B` has shape `mp x nq`,
        - `V` has shape `nq x c`.

        There are three steps:
        1. Reshape `V` from shape `nq x c` to shape `c x q x n`.
        2. Perform the matrix-matrix product. The result `R` has shape `c x p x m`.
        3. Reshape `R` from shape `c x p x m` to shape `mp x c`.
        """
        # Get and check dimensions
        m, n = A.shape
        p, q = B.shape
        assert V.shape[0] == n * q
        c = V.shape[1]

        # Perform steps described above
        V = CovariancePushforwardLaplaceWrapper2.mnxs_to_mxnxs(V, n, q, c).permute(
            2, 1, 0
        )  # step 1
        R = B @ V @ A.T  # step 2

        del A, B, V
        torch.cuda.empty_cache()
        gc.collect()

        return CovariancePushforwardLaplaceWrapper2.mxnxs_to_mnxs(
            R.permute(2, 1, 0), m, p, c
        )  # step 3

    def apply_parameter_mask(self, mask_regex):
        if mask_regex is not None:
            for param_name, param in self.model.named_parameters():
                if not re.match(rf"^{mask_regex}.*$", param_name):
                    param.requires_grad = False

    def perform_laplace_approximation(self, train_loader, val_loader, channels_last):
        self.theta_0_list = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        self.theta_0_vec = torch.nn.utils.parameters_to_vector(
            self.theta_0_list
        ).detach()
        self.prior_precision = 1.0
        self.kfac = self.get_covariance_kfac_loader(train_loader, channels_last)

        self.prior_precision = self.optimize_prior_precision_cv(
            val_loader, channels_last
        )

        self.is_laplace_approximated = True

    @staticmethod
    def get_ece(out_dist, targets):
        confidences, predictions = out_dist.max(dim=-1)  # [B]
        correctnesses = predictions.eq(targets).int()

        return calibration_error(
            confidences=confidences, correctnesses=correctnesses, num_bins=15, norm="l1"
        )

    def optimize_prior_precision_cv(
        self,
        val_loader,
        channels_last,
        log_prior_prec_min=-1,
        log_prior_prec_max=2,
        grid_size=50,
    ):
        interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        self.prior_precision = self.gridsearch(
            interval=interval,
            val_loader=val_loader,
            channels_last=channels_last,
        )

        logger.info(f"Optimized prior precision is {self.prior_precision}.")

    def gridsearch(
        self,
        interval,
        val_loader,
        channels_last,
    ):
        results = []
        prior_precs = []
        for prior_prec in interval:
            logger.info(f"Trying {prior_prec}...")
            start_time = time.perf_counter()
            self.prior_precision = prior_prec

            try:
                out_dist, targets = self.validate(
                    val_loader=val_loader,
                    channels_last=channels_last,
                )
                result = self.get_ece(out_dist, targets).item()
                accuracy = out_dist.argmax(dim=-1).eq(targets).float().mean()
            except RuntimeError as error:
                logger.info(f"Caught an exception in validate: {error}")
                result = float("inf")
                accuracy = float("NaN")
            logger.info(
                f"Took {time.perf_counter() - start_time} seconds, result: {result}, "
                f"accuracy {accuracy}"
            )
            results.append(result)
            prior_precs.append(prior_prec)

        return prior_precs[np.argmin(results)]

    @torch.no_grad()
    def validate(self, val_loader, channels_last):
        self.model.eval()
        device = next(self.model.parameters()).device
        output_means = []
        targets = []

        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)

            if channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            mean, var = self(input)
            out = self.predictive_fn(mean, var)

            output_means.append(out)
            targets.append(target)

        return torch.cat(output_means, dim=0), torch.cat(targets, dim=0)

    def get_covariance_kfac_loader(self, train_loader, channels_last):
        """Compute the KFAC approximation based on a list of mini-batches `datalist`."""
        # Accumulate KFAC approximations over all mini-batches
        num_data = 0
        for i, (X, y) in enumerate(train_loader):
            batch_size = X.shape[0]
            new_num_data = num_data + batch_size
            if i == 0:
                # Initialize `kfac`.
                # If there is only one mini-batch, return the KFAC approximation.
                kfac = self.get_kfac_minibatch(X, y, channels_last)

                if len(train_loader) == 1:
                    return kfac
            else:
                # Compute mini-batch KFAC approximation
                mb_kfac = self.get_kfac_minibatch(X, y, channels_last)

                # Add mini-batch KFAC approximation
                kfac = self.add_kfacs(
                    kfac,
                    mb_kfac,
                    alpha1=num_data / new_num_data,
                    alpha2=batch_size / new_num_data,
                )

            # Free VRAM
            del mb_kfac
            torch.cuda.empty_cache()
            gc.collect()

            # Update number of data points
            num_data = new_num_data

        kfac = self.get_covariance_kfac(kfac, self.prior_precision)

        return kfac

    @staticmethod
    def get_covariance_kfac(kfac_list, prior_precision):
        covariance_kfac_list = []
        for kfac in kfac_list:
            if len(kfac) == 2:
                A, B = kfac
                s_A, U_A = CovariancePushforwardLaplaceWrapper2.get_eigendecomposition(
                    A
                )
                s_B, U_B = CovariancePushforwardLaplaceWrapper2.get_eigendecomposition(
                    B
                )

                A_cov = (
                    U_A @ torch.diag(torch.reciprocal(s_A + prior_precision)) @ U_A.T
                )
                B_cov = (
                    U_B @ torch.diag(torch.reciprocal(s_B + prior_precision)) @ U_B.T
                )
                covariance_kfac_list.append((A_cov, B_cov))
            else:
                A = kfac[0]
                s_A, U_A = CovariancePushforwardLaplaceWrapper2.get_eigendecomposition(
                    A
                )
                A_cov = (
                    U_A @ torch.diag(torch.reciprocal(s_A + prior_precision)) @ U_A.T
                )
                covariance_kfac_list.append((A_cov,))

        return covariance_kfac_list

    @staticmethod
    def get_eigendecomposition(M):
        """Get the eigendecomposition of a symmetric, PSD matrix `M`."""
        # Compute eigendecomposition
        eigvals, U = torch.linalg.eigh(M)

        return eigvals, U

    def get_kfac_minibatch(self, X, y, channels_last):
        """Get the KFAC approximation based on one mini-batch `(X, y)`.

        This returns a list-representation of KFAC. Its entries are lists that contain
        either a single matrix `[Fi]` or a pair of matrices `[Ai, Bi]` such that
        `Fi = Ai ⊗ Bi`. An example could look like this:
        ```
        kfac = [
            [F1],
            [A2, B2],  # F2 = A2 ⊗ B2
            [F3],
            [A4, B4],  # F4 = A4 ⊗ B4
            [A5, B5],  # F5 = A5 ⊗ B5
            ...
        ]
        ```
        """
        # Extend model and loss function. The use_converter parameter is used for ResNet
        # compatibility
        device = next(self.model.parameters()).device

        # Forward and backward pass
        X, y = X.to(device), y.to(device)

        if channels_last:
            X = X.contiguous(memory_format=torch.channels_last)

        loss = self.loss_fn(self.model(X), y)

        with backpack(self.extension):
            loss.backward()

        # Extract KFAC matrix from model
        kfac = [
            [elem.detach() for elem in param.kfac]
            for param in self.model.parameters()
            if param.requires_grad
        ]

        # Free GPU memory
        del loss
        torch.cuda.empty_cache()
        gc.collect()

        return kfac

    @staticmethod
    def add_kfacs(kfac1, kfac2, alpha1=1.0, alpha2=1.0):
        """Add two KFAC approximations by adding all corresponding factors."""
        return [
            tuple(alpha1 * F1 + alpha2 * F2 for F1, F2 in zip(B1, B2, strict=True))
            for B1, B2 in zip(kfac1, kfac2, strict=True)
        ]

    @staticmethod
    def scale_kfac(kfac, alpha):
        """Scale all factors of all blocks by a scalar `alpha`."""
        return [tuple(alpha * F for F in B) for B in kfac]
