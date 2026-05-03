from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch


torch.set_default_dtype(torch.float64)

ActivationName = str


@dataclass
class InterpolatedFirstLayerTerms:
    """
    First-layer kernel blocks needed to propagate the infinite-width kernel recursion.

    Shapes:
        test_diag:   (M,)
        test_train:  (M, N)
        train_train: (N, N)
    """

    test_diag: torch.Tensor
    test_train: torch.Tensor
    train_train: torch.Tensor


@dataclass
class LayerwiseNTKConfig:
    depth: int
    act: ActivationName
    Cb: float
    Cw: float
    diag_reg: float = 1e-12
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64


@dataclass
class LayerwiseNTKResult:
    first_layer: InterpolatedFirstLayerTerms
    train_kernel_layers: list[torch.Tensor]
    train_ntk_layers: list[torch.Tensor]
    test_diag_layers: list[torch.Tensor]
    test_kernel_layers: list[torch.Tensor]
    test_ntk_layers: list[torch.Tensor]
    train_train_ntk: torch.Tensor
    test_train_ntk: torch.Tensor
    train_solution: torch.Tensor
    train_logits: torch.Tensor
    test_logits: torch.Tensor


class ActivationMoments:
    """
    Gaussian expectations needed by the layerwise NTK recursion.

    The erf and ReLU paths use closed forms. Any other smooth activation can be
    supplied through act_fn and act_p_fn and is evaluated with batched
    Gauss-Hermite quadrature.
    """

    def __init__(
        self,
        act: ActivationName,
        *,
        act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        act_p_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        quadrature_n: int = 40,
        batch_size: int = 4096,
        device=None,
        dtype=torch.float64,
    ):
        normalized = act.strip().lower()
        self.act = normalized
        self.act_fn = act_fn
        self.act_p_fn = act_p_fn
        self.batch_size = int(batch_size)
        self.device = device
        self.dtype = dtype

        self.use_closed_form = self.act in {"erf", "relu"} and act_fn is None and act_p_fn is None
        if not self.use_closed_form:
            if act_fn is None or act_p_fn is None:
                raise ValueError(
                    "Generic custom_ntk activations require both act_fn and act_p_fn."
                )
            x_np, w_np = np.polynomial.hermite.hermgauss(int(quadrature_n))
            x = torch.as_tensor(x_np, device=device, dtype=dtype)
            w = torch.as_tensor(w_np, device=device, dtype=dtype)
            self.z1 = math.sqrt(2.0) * x[:, None]
            self.z2 = math.sqrt(2.0) * x[None, :]
            self.weight_2d = w[:, None] * w[None, :] / math.pi

    def nngp(self, q1: torch.Tensor, q2: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.use_closed_form and self.act == "erf":
            return self._erf_nngp(q1, q2, c)
        if self.use_closed_form and self.act == "relu":
            return self._relu_nngp(q1, q2, c)
        return self._generic_bivariate_expectation(self.act_fn, self.act_fn, q1, q2, c)

    def sigma_dot(self, q1: torch.Tensor, q2: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.use_closed_form and self.act == "erf":
            return self._erf_sigma_dot(q1, q2, c)
        if self.use_closed_form and self.act == "relu":
            return self._relu_sigma_dot(q1, q2, c)
        return self._generic_bivariate_expectation(self.act_p_fn, self.act_p_fn, q1, q2, c)

    def _generic_bivariate_expectation(
        self,
        fun1: Callable[[torch.Tensor], torch.Tensor],
        fun2: Callable[[torch.Tensor], torch.Tensor],
        q1: torch.Tensor,
        q2: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2, c = torch.broadcast_tensors(
            q1.to(device=self.device, dtype=self.dtype),
            q2.to(device=self.device, dtype=self.dtype),
            c.to(device=self.device, dtype=self.dtype),
        )
        out_shape = q1.shape
        q1_flat = q1.reshape(-1)
        q2_flat = q2.reshape(-1)
        c_flat = c.reshape(-1)
        result = torch.empty_like(q1_flat)

        z1 = self.z1[None, :, :]
        z2 = self.z2[None, :, :]
        weight = self.weight_2d[None, :, :]

        for start in range(0, q1_flat.numel(), self.batch_size):
            end = min(start + self.batch_size, q1_flat.numel())
            q1_b = torch.clamp(q1_flat[start:end], min=1e-30)[:, None, None]
            q2_b = torch.clamp(q2_flat[start:end], min=1e-30)[:, None, None]
            c_b = c_flat[start:end][:, None, None]

            rho = c_b / torch.sqrt(q1_b * q2_b)
            rho = torch.clamp(rho, -1.0, 1.0)
            u = torch.sqrt(q1_b) * z1
            v = torch.sqrt(q2_b) * (rho * z1 + torch.sqrt(torch.clamp(1.0 - rho * rho, min=0.0)) * z2)

            vals = fun1(u) * fun2(v)
            result[start:end] = torch.sum(weight * vals, dim=(-2, -1))

        return result.reshape(out_shape)

    @staticmethod
    def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(x, min=0.0))

    def _relu_rho(self, q1: torch.Tensor, q2: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        denom = self._safe_sqrt(q1 * q2)
        rho = torch.where(denom > 0.0, c / denom, torch.zeros_like(c))
        rho = torch.clamp(rho, -1.0, 1.0)
        return denom, rho

    def _relu_nngp(self, q1: torch.Tensor, q2: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sqrt_q1q2, rho = self._relu_rho(q1, q2, c)
        theta = torch.arccos(rho)
        return sqrt_q1q2 * (torch.sin(theta) + (math.pi - theta) * torch.cos(theta)) / (2.0 * math.pi)

    def _relu_sigma_dot(self, q1: torch.Tensor, q2: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        _, rho = self._relu_rho(q1, q2, c)
        theta = torch.arccos(rho)
        return (math.pi - theta) / (2.0 * math.pi)

    @staticmethod
    def _erf_nngp(q1: torch.Tensor, q2: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        denom = torch.sqrt(torch.clamp((1.0 + 2.0 * q1) * (1.0 + 2.0 * q2), min=1e-30))
        arg = torch.clamp(2.0 * c / denom, -1.0, 1.0)
        return (2.0 / math.pi) * torch.arcsin(arg)

    @staticmethod
    def _erf_sigma_dot(q1: torch.Tensor, q2: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        denom = torch.sqrt(torch.clamp((1.0 + 2.0 * q1) * (1.0 + 2.0 * q2) - 4.0 * c * c, min=1e-30))
        return (4.0 / math.pi) / denom


def _to_tensor(x, *, device=None, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


def _dimension_normalized_gram(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("x1 and x2 must both have shape (num_points, d).")
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("x1 and x2 must share the same feature dimension.")
    d = x1.shape[1]
    return torch.matmul(x1, x2.T) / float(d)


def build_clean_first_layer_terms(
    X_train,
    X_test,
    *,
    Cb: float,
    Cw: float,
    device=None,
    dtype=torch.float64,
) -> InterpolatedFirstLayerTerms:
    """
    Standard noiseless first-layer kernel:
        K^(1) = C_b + C_w <x, x'>
    """
    X_train = _to_tensor(X_train, device=device, dtype=dtype)
    X_test = _to_tensor(X_test, device=device, dtype=dtype)
    gram_train = _dimension_normalized_gram(X_train, X_train)
    gram_test_train = _dimension_normalized_gram(X_test, X_train)
    test_diag = torch.sum(X_test * X_test, dim=1) / float(X_test.shape[1])

    return InterpolatedFirstLayerTerms(
        test_diag=Cb + Cw * test_diag,
        test_train=Cb + Cw * gram_test_train,
        train_train=Cb + Cw * gram_train,
    )


def build_additive_gaussian_mean_first_layer_terms(
    X_train,
    X_test,
    *,
    eps: float,
    Cb: float,
    Cw: float,
    device=None,
    dtype=torch.float64,
) -> InterpolatedFirstLayerTerms:
    """
    First-layer mean kernel for the additive-Gaussian noise model in the d -> infinity limit.

    The train set is replaced by its mean noisy kernel, while test points remain clean:
        E[z_alpha z_beta] = delta_{alpha beta}(1-eps)^2 + eps^2 <x_alpha x_beta>
        E[x_* z_alpha] = eps <x_* x_alpha>
    """
    X_train = _to_tensor(X_train, device=device, dtype=dtype)
    X_test = _to_tensor(X_test, device=device, dtype=dtype)
    eps_t = _to_tensor(eps, device=device, dtype=dtype)

    gram_train = _dimension_normalized_gram(X_train, X_train)
    gram_test_train = _dimension_normalized_gram(X_test, X_train)
    test_diag = torch.sum(X_test * X_test, dim=1) / float(X_test.shape[1])

    train_train = (eps_t * eps_t) * gram_train
    diag_vals = ((1.0 - eps_t) ** 2) + (eps_t * eps_t) * torch.diag(gram_train)
    train_train = train_train - torch.diag(torch.diag(train_train)) + torch.diag(diag_vals)

    return InterpolatedFirstLayerTerms(
        test_diag=Cb + Cw * test_diag,
        test_train=Cb + Cw * (eps_t * gram_test_train),
        train_train=Cb + Cw * train_train,
    )


def build_linear_interpolated_first_layer_terms(
    X_train,
    X_test,
    *,
    eps: float,
    Cb: float,
    Cw: float,
    pure_train_diag: float = 1.0,
    pure_train_offdiag: float = 0.0,
    pure_test_train: float = 0.0,
    device=None,
    dtype=torch.float64,
) -> InterpolatedFirstLayerTerms:
    """
    Literal linear interpolation between a pure-noise kernel and the clean kernel.

    This is not the same as the additive-Gaussian mean kernel unless the user intends it.
    It is included because sometimes an explicit epsilon-interpolation is the most convenient
    way to probe the recursion away from the pure-noise point.
    """
    X_train = _to_tensor(X_train, device=device, dtype=dtype)
    X_test = _to_tensor(X_test, device=device, dtype=dtype)
    eps_t = _to_tensor(eps, device=device, dtype=dtype)

    clean = build_clean_first_layer_terms(
        X_train=X_train,
        X_test=X_test,
        Cb=Cb,
        Cw=Cw,
        device=device,
        dtype=dtype,
    )

    N = X_train.shape[0]
    pure_train = torch.full((N, N), fill_value=float(pure_train_offdiag), device=device, dtype=dtype)
    pure_train = pure_train + torch.diag(
        torch.full((N,), fill_value=float(pure_train_diag - pure_train_offdiag), device=device, dtype=dtype)
    )
    pure_test_train_block = torch.full(
        clean.test_train.shape,
        fill_value=float(pure_test_train),
        device=device,
        dtype=dtype,
    )

    pure = InterpolatedFirstLayerTerms(
        test_diag=clean.test_diag.clone(),
        test_train=Cb + Cw * pure_test_train_block,
        train_train=Cb + Cw * pure_train,
    )

    return InterpolatedFirstLayerTerms(
        test_diag=clean.test_diag,
        test_train=(1.0 - eps_t) * pure.test_train + eps_t * clean.test_train,
        train_train=(1.0 - eps_t) * pure.train_train + eps_t * clean.train_train,
    )


def propagate_layerwise_ntk(
    first_layer: InterpolatedFirstLayerTerms,
    *,
    depth: int,
    act: ActivationName,
    act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    act_p_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    moment_quadrature_n: int = 40,
    moment_batch_size: int = 4096,
    Cb: float,
    Cw: float,
    y_train,
    diag_reg: float = 1e-12,
    device=None,
    dtype=torch.float64,
) -> LayerwiseNTKResult:
    """
    Propagate K and Theta with the standard infinite-width recursions:
        K^{l+1} = C_b + C_w <sigma sigma>_{K^l}
        Theta^{l+1} = C_b + C_w ( <sigma sigma>_{K^l} + <sigma' sigma'>_{K^l} Theta^l )

    Only the train-train block, the test-train block, and the test diagonals are propagated,
    which is all that is needed to make predictions.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1.")

    moments = ActivationMoments(
        act=act,
        act_fn=act_fn,
        act_p_fn=act_p_fn,
        quadrature_n=moment_quadrature_n,
        batch_size=moment_batch_size,
        device=device,
        dtype=dtype,
    )

    test_diag = _to_tensor(first_layer.test_diag, device=device, dtype=dtype)
    test_train = _to_tensor(first_layer.test_train, device=device, dtype=dtype)
    train_train = _to_tensor(first_layer.train_train, device=device, dtype=dtype)
    y_train = _to_tensor(y_train, device=device, dtype=dtype)

    if train_train.ndim != 2 or train_train.shape[0] != train_train.shape[1]:
        raise ValueError("train_train must have shape (N, N).")
    if test_train.ndim != 2 or test_train.shape[1] != train_train.shape[0]:
        raise ValueError("test_train must have shape (M, N).")
    if test_diag.ndim != 1 or test_diag.shape[0] != test_train.shape[0]:
        raise ValueError("test_diag must have shape (M,).")
    if y_train.ndim != 2 or y_train.shape[0] != train_train.shape[0]:
        raise ValueError("y_train must have shape (N, C).")

    train_kernel_layers = [train_train.clone()]
    train_ntk_layers = [train_train.clone()]
    test_diag_layers = [test_diag.clone()]
    test_kernel_layers = [test_train.clone()]
    test_ntk_layers = [test_train.clone()]

    for _layer in range(2, depth + 1):
        prev_train_k = train_kernel_layers[-1]
        prev_train_theta = train_ntk_layers[-1]
        train_var = torch.diag(prev_train_k)

        train_q1 = train_var[:, None]
        train_q2 = train_var[None, :]
        train_eaa = moments.nngp(train_q1, train_q2, prev_train_k)
        train_epp = moments.sigma_dot(train_q1, train_q2, prev_train_k)

        next_train_k = Cb + Cw * train_eaa
        next_train_theta = Cb + Cw * (train_eaa + train_epp * prev_train_theta)

        prev_test_diag = test_diag_layers[-1]
        prev_test_k = test_kernel_layers[-1]
        prev_test_theta = test_ntk_layers[-1]

        test_q1 = prev_test_diag[:, None]
        test_q2 = train_var[None, :]
        test_eaa = moments.nngp(test_q1, test_q2, prev_test_k)
        test_epp = moments.sigma_dot(test_q1, test_q2, prev_test_k)

        next_test_k = Cb + Cw * test_eaa
        next_test_theta = Cb + Cw * (test_eaa + test_epp * prev_test_theta)

        next_test_diag = Cb + Cw * moments.nngp(prev_test_diag, prev_test_diag, prev_test_diag)

        train_kernel_layers.append(next_train_k)
        train_ntk_layers.append(next_train_theta)
        test_diag_layers.append(next_test_diag)
        test_kernel_layers.append(next_test_k)
        test_ntk_layers.append(next_test_theta)

    reg = _to_tensor(diag_reg, device=device, dtype=dtype)
    identity = torch.eye(train_ntk_layers[-1].shape[0], device=device, dtype=dtype)
    train_solution = torch.linalg.solve(train_ntk_layers[-1] + reg * identity, y_train)
    train_logits = torch.matmul(train_ntk_layers[-1], train_solution)
    test_logits = torch.matmul(test_ntk_layers[-1], train_solution)

    return LayerwiseNTKResult(
        first_layer=InterpolatedFirstLayerTerms(
            test_diag=test_diag,
            test_train=test_train,
            train_train=train_train,
        ),
        train_kernel_layers=train_kernel_layers,
        train_ntk_layers=train_ntk_layers,
        test_diag_layers=test_diag_layers,
        test_kernel_layers=test_kernel_layers,
        test_ntk_layers=test_ntk_layers,
        train_train_ntk=train_ntk_layers[-1],
        test_train_ntk=test_ntk_layers[-1],
        train_solution=train_solution,
        train_logits=train_logits,
        test_logits=test_logits,
    )


def evaluate_additive_gaussian_mean_ntk(
    X_train,
    y_train,
    X_test,
    *,
    eps: float,
    depth: int,
    act: ActivationName,
    act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    act_p_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    moment_quadrature_n: int = 40,
    moment_batch_size: int = 4096,
    Cb: float,
    Cw: float,
    diag_reg: float = 1e-12,
    device=None,
    dtype=torch.float64,
) -> LayerwiseNTKResult:
    first_layer = build_additive_gaussian_mean_first_layer_terms(
        X_train=X_train,
        X_test=X_test,
        eps=eps,
        Cb=Cb,
        Cw=Cw,
        device=device,
        dtype=dtype,
    )
    return propagate_layerwise_ntk(
        first_layer=first_layer,
        depth=depth,
        act=act,
        act_fn=act_fn,
        act_p_fn=act_p_fn,
        moment_quadrature_n=moment_quadrature_n,
        moment_batch_size=moment_batch_size,
        Cb=Cb,
        Cw=Cw,
        y_train=y_train,
        diag_reg=diag_reg,
        device=device,
        dtype=dtype,
    )


def evaluate_clean_ntk(
    X_train,
    y_train,
    X_test,
    *,
    depth: int,
    act: ActivationName,
    act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    act_p_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    moment_quadrature_n: int = 40,
    moment_batch_size: int = 4096,
    Cb: float,
    Cw: float,
    diag_reg: float = 1e-12,
    device=None,
    dtype=torch.float64,
) -> LayerwiseNTKResult:
    first_layer = build_clean_first_layer_terms(
        X_train=X_train,
        X_test=X_test,
        Cb=Cb,
        Cw=Cw,
        device=device,
        dtype=dtype,
    )
    return propagate_layerwise_ntk(
        first_layer=first_layer,
        depth=depth,
        act=act,
        act_fn=act_fn,
        act_p_fn=act_p_fn,
        moment_quadrature_n=moment_quadrature_n,
        moment_batch_size=moment_batch_size,
        Cb=Cb,
        Cw=Cw,
        y_train=y_train,
        diag_reg=diag_reg,
        device=device,
        dtype=dtype,
    )


def _demo():
    torch.manual_seed(0)
    X_train = torch.randn(8, 16)
    X_test = torch.randn(3, 16)
    y_train = torch.eye(2, dtype=torch.float64).repeat(4, 1)

    result = evaluate_additive_gaussian_mean_ntk(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        eps=0.35,
        depth=4,
        act="erf",
        Cb=1e-8,
        Cw=2.0,
    )

    print("Custom layerwise NTK demo")
    print("train_train_ntk shape:", tuple(result.train_train_ntk.shape))
    print("test_train_ntk shape :", tuple(result.test_train_ntk.shape))
    print("test_logits shape    :", tuple(result.test_logits.shape))
    print("first test logits    :", result.test_logits[0].detach().cpu().numpy())


if __name__ == "__main__":
    _demo()
