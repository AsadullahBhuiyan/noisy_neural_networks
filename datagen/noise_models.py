import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from gaussian_integrals import GaussianQuadrature1D, GaussianQuadrature2D


torch.set_default_dtype(torch.float64)

KernelDict = Dict[str, torch.Tensor]


@dataclass
class PureNoiseConfig:
    noise_model: str                  # "additive_gaussian" or "replacement"
    x_test: torch.Tensor             # shape (d,)
    X_train: torch.Tensor            # shape (N, d)
    Cb: float
    Cw: float
    act: Callable[[torch.Tensor], torch.Tensor]
    n_layers: int                    # total number of layers L, so K^(1),...,K^(L)

    # Only used for replacement noise
    replacement_mean: float = 0.5
    replacement_second_moment: float = 1.0 / 3.0

    # Quadrature
    gh_1d_n: int = 80
    gh_2d_n: int = 60
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64


class PureNoiseKernel:
    """
    Computes the pure-noise Bayesian kernel K^(l) for the special symmetry-reduced entries:
        tete         = test-test
        tetr         = test-train
        trtr_diag    = train-train diagonal
        trtr_offdiag = train-train off-diagonal

    Stores kernels for layers l = 1,...,L in self.kernels, where self.kernels[l-1] = K^(l).
    """

    def __init__(self, cfg: PureNoiseConfig):
        self.cfg = cfg
        self.x_test = cfg.x_test.to(device=cfg.device, dtype=cfg.dtype)
        self.X_train = cfg.X_train.to(device=cfg.device, dtype=cfg.dtype)

        if self.x_test.ndim != 1:
            raise ValueError("x_test must have shape (d,).")
        if self.X_train.ndim != 2:
            raise ValueError("X_train must have shape (N, d).")
        if self.X_train.shape[1] != self.x_test.shape[0]:
            raise ValueError("x_test and X_train must have matching feature dimension.")
        if cfg.n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        self.N, self.d = self.X_train.shape

        self.GQ1D = GaussianQuadrature1D(
            n=cfg.gh_1d_n, device=cfg.device, dtype=cfg.dtype
        )
        self.GQ2D = GaussianQuadrature2D(
            n=cfg.gh_2d_n, device=cfg.device, dtype=cfg.dtype
        )

        self.kernels: List[KernelDict] = []
        self._build_kernels()

    def _initial_expectations(self):
        """
        Pure-noise first-layer expectations:
            E_test_test
            E_test_train
            E_train_train_diag
            E_train_train_offdiag
        """
        x_test_sq_mean = torch.mean(self.x_test * self.x_test)

        if self.cfg.noise_model == "additive_gaussian":
            E_tete = x_test_sq_mean
            E_tetr = torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)
            E_trtr_diag = torch.tensor(1.0, device=self.cfg.device, dtype=self.cfg.dtype)
            E_trtr_offdiag = torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)
            return E_tete, E_tetr, E_trtr_diag, E_trtr_offdiag

        if self.cfg.noise_model == "replacement":
            mu = torch.as_tensor(
                self.cfg.replacement_mean, device=self.cfg.device, dtype=self.cfg.dtype
            )
            mu2 = torch.as_tensor(
                self.cfg.replacement_second_moment,
                device=self.cfg.device,
                dtype=self.cfg.dtype,
            )
            x_test_mean = torch.mean(self.x_test)

            E_tete = x_test_sq_mean
            E_tetr = mu * x_test_mean
            E_trtr_diag = mu2
            E_trtr_offdiag = mu * mu
            return E_tete, E_tetr, E_trtr_diag, E_trtr_offdiag

        raise ValueError(f"Unknown noise_model={self.cfg.noise_model}")

    def _next_kernel_from_prev(self, prev: KernelDict) -> KernelDict:
        """
        K^(l+1) = Cb + Cw * <sigma sigma>_{K^(l)}
        """
        act = self.cfg.act
        Cb = torch.as_tensor(self.cfg.Cb, device=self.cfg.device, dtype=self.cfg.dtype)
        Cw = torch.as_tensor(self.cfg.Cw, device=self.cfg.device, dtype=self.cfg.dtype)

        E_tete = self.GQ1D.E(act, act, prev["tete"])
        E_tetr = self.GQ2D.E(
            act, act,
            prev["tete"],         # Var(test)
            prev["trtr_diag"],    # Var(train)
            prev["tetr"],         # Cov(test, train)
        )
        E_trtr_diag = self.GQ1D.E(act, act, prev["trtr_diag"])
        E_trtr_offdiag = self.GQ2D.E(
            act, act,
            prev["trtr_diag"],    # Var(train)
            prev["trtr_diag"],    # Var(train')
            prev["trtr_offdiag"], # Cov(train, train')
        )

        return {
            "tete": Cb + Cw * E_tete,
            "tetr": Cb + Cw * E_tetr,
            "trtr_diag": Cb + Cw * E_trtr_diag,
            "trtr_offdiag": Cb + Cw * E_trtr_offdiag,
        }

    def _build_kernels(self):
        E_tete, E_tetr, E_trtr_diag, E_trtr_offdiag = self._initial_expectations()
        Cb = torch.as_tensor(self.cfg.Cb, device=self.cfg.device, dtype=self.cfg.dtype)
        Cw = torch.as_tensor(self.cfg.Cw, device=self.cfg.device, dtype=self.cfg.dtype)

        K1 = {
            "tete": Cb + Cw * E_tete,
            "tetr": Cb + Cw * E_tetr,
            "trtr_diag": Cb + Cw * E_trtr_diag,
            "trtr_offdiag": Cb + Cw * E_trtr_offdiag,
        }
        self.kernels.append(K1)

        for _ in range(2, self.cfg.n_layers + 1):
            self.kernels.append(self._next_kernel_from_prev(self.kernels[-1]))


class Expectations:
    """
    Computes the Gaussian expectations needed from the pure-noise kernels.

    Stores expectations for layers l = 1,...,L-1, i.e. based on K^(l).
    """

    def __init__(
        self,
        kernels: List[KernelDict],
        act: Callable[[torch.Tensor], torch.Tensor],
        act_p: Callable[[torch.Tensor], torch.Tensor],
        act_pp: Callable[[torch.Tensor], torch.Tensor],
        act_ppp: Callable[[torch.Tensor], torch.Tensor],
        act_pppp: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        gh_1d_n: int = 80,
        gh_2d_n: int = 60,
        device=None,
        dtype=torch.float64,
    ):
        self.kernels = kernels
        self.n_layers = len(kernels)
        self.act = act
        self.act_p = act_p
        self.act_pp = act_pp
        self.act_ppp = act_ppp

        self.GQ1D = GaussianQuadrature1D(n=gh_1d_n, device=device, dtype=dtype)
        self.GQ2D = GaussianQuadrature2D(n=gh_2d_n, device=device, dtype=dtype)

        self.E_act_act: List[KernelDict] = []
        self.E_act_p_act_p: List[KernelDict] = []
        self.E_act_pp_act: List[KernelDict] = []
        self.E_act_ppp_act_p: List[KernelDict] = []
        self.E_act_pp_act_pp: List[KernelDict] = []

        # Expectations are needed for layers l = 1,...,L-1
        for l in range(self.n_layers - 1):
            K = self.kernels[l]

            self.E_act_act.append(self._compute_pair_expectations(self.act, self.act, K))
            self.E_act_p_act_p.append(self._compute_pair_expectations(self.act_p, self.act_p, K))
            self.E_act_pp_act.append(self._compute_pair_expectations(self.act_pp, self.act, K))
            self.E_act_ppp_act_p.append(self._compute_pair_expectations(self.act_ppp, self.act_p, K))
            self.E_act_pp_act_pp.append(self._compute_pair_expectations(self.act_pp, self.act_pp, K))

    def _compute_pair_expectations(
        self,
        fun1: Callable[[torch.Tensor], torch.Tensor],
        fun2: Callable[[torch.Tensor], torch.Tensor],
        K: KernelDict,
    ) -> KernelDict:
        """
        Returns expectations for:
            tete, tetr, trte, trtr_diag, trtr_offdiag

        Note:
            tetr = E[ fun1(test) fun2(train) ]
            trte = E[ fun1(train) fun2(test) ]
        """
        E_tete = self.GQ1D.E(fun1, fun2, K["tete"])

        E_tetr = self.GQ2D.E(
            fun1, fun2,
            K["tete"],
            K["trtr_diag"],
            K["tetr"],
        )

        E_trte = self.GQ2D.E(
            fun1, fun2,
            K["trtr_diag"],
            K["tete"],
            K["tetr"],
        )

        E_trtr_diag = self.GQ1D.E(fun1, fun2, K["trtr_diag"])

        E_trtr_offdiag = self.GQ2D.E(
            fun1, fun2,
            K["trtr_diag"],
            K["trtr_diag"],
            K["trtr_offdiag"],
        )

        return {
            "tete": E_tete,
            "tetr": E_tetr,
            "trte": E_trte,
            "trtr_diag": E_trtr_diag,
            "trtr_offdiag": E_trtr_offdiag,
        }


class PTensor:
    """
    Computes products
        P^{a->b} = prod_{i=a}^{b-1} <sigma' sigma'>_{K^(i)}
    for the symmetry-reduced index types.
    """

    def __init__(self, E_act_p_act_p: List[KernelDict], device=None, dtype=torch.float64):
        self.E_act_p_act_p = E_act_p_act_p
        self.device = device
        self.dtype = dtype

    def compute(self, a: int, b: int) -> KernelDict:
        """
        Compute product from layers i = a,...,b-1.

        Layer numbering is 1-based to match the math.
        """
        prod = {
            "tete": torch.tensor(1.0, device=self.device, dtype=self.dtype),
            "tetr": torch.tensor(1.0, device=self.device, dtype=self.dtype),
            "trte": torch.tensor(1.0, device=self.device, dtype=self.dtype),
            "trtr_diag": torch.tensor(1.0, device=self.device, dtype=self.dtype),
            "trtr_offdiag": torch.tensor(1.0, device=self.device, dtype=self.dtype),
        }

        for layer in range(a, b):
            E = self.E_act_p_act_p[layer - 1]
            for key in prod:
                prod[key] = prod[key] * E[key]

        return prod


class UTensor:
    """
    Computes
        U_alpha^{a->b} = prod_{i=a}^{b-1} ( <sigma' sigma'>_{K_aa^(i)} + <sigma'' sigma>_{K_aa^(i)} )

    We only need single-index types:
        'test'  -> alpha = *
        'train' -> alpha = chi
    """

    def __init__(self, E_act_p_act_p, E_act_pp_act, device=None, dtype=torch.float64):
        self.E_act_p_act_p = E_act_p_act_p
        self.E_act_pp_act = E_act_pp_act
        self.device = device
        self.dtype = dtype

    def compute(self, a: int, b: int, alpha_type: str) -> torch.Tensor:
        if alpha_type not in ["test", "train"]:
            raise ValueError("alpha_type must be 'test' or 'train'")

        key = "tete" if alpha_type == "test" else "trtr_diag"

        prod = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        for layer in range(a, b):
            prod = prod * (
                self.E_act_p_act_p[layer - 1][key]
                + self.E_act_pp_act[layer - 1][key]
            )
        return prod


class PureNoiseNTK:
    """
    Computes the pure-noise NTK recursively:
        Theta^(1) = K^(1)
        Theta^(l+1) = Cb + Cw [ <sigma sigma>_{K^(l)} + <sigma' sigma'>_{K^(l)} * Theta^(l) ]
    """

    def __init__(
        self,
        kernels: List[KernelDict],
        E_act_act: List[KernelDict],
        E_act_p_act_p: List[KernelDict],
        Cb: float,
        Cw: float,
        device=None,
        dtype=torch.float64,
    ):
        self.kernels = kernels
        self.E_act_act = E_act_act
        self.E_act_p_act_p = E_act_p_act_p
        self.n_layers = len(kernels)
        self.Cb = torch.as_tensor(Cb, device=device, dtype=dtype)
        self.Cw = torch.as_tensor(Cw, device=device, dtype=dtype)

        self.ntk_layers: List[KernelDict] = []
        self.NTK: KernelDict = {}
        self.theta_d = None
        self.theta_o = None
        self.theta_star = None

        self._build_ntk()

    def _build_ntk(self):
        # Theta^(1) = K^(1)
        Theta = {k: v.clone() for k, v in self.kernels[0].items()}
        self.ntk_layers.append(Theta)

        # Build Theta^(2),...,Theta^(L)
        for l in range(1, self.n_layers):
            Eaa = self.E_act_act[l - 1]
            Epp = self.E_act_p_act_p[l - 1]
            prev = self.ntk_layers[-1]

            Theta_next = {}
            for key in ["tete", "tetr", "trtr_diag", "trtr_offdiag"]:
                Theta_next[key] = self.Cb + self.Cw * (Eaa[key] + Epp[key] * prev[key])

            self.ntk_layers.append(Theta_next)

        self.NTK = self.ntk_layers[-1]

        # Normalized theta quantities used in your formulas
        norm = self.Cw ** (self.n_layers - 1)
        self.theta_d = self.NTK["trtr_diag"] / norm
        self.theta_o = self.NTK["trtr_offdiag"] / norm
        self.theta_star = self.NTK["tetr"] / norm


class TransferCoefficients:
    """
    Computes M, S, T using the symmetry-reduced pure-noise objects.

    Pair keys:
        trte          = (chi, *)
        trtr_offdiag  = (chi, chi')
    """

    PAIR_KEYS = ["tetr", "trte", "trtr_offdiag"]

    def __init__(
        self,
        kernels: List[KernelDict],      # K^(l)
        ntk_layers: List[KernelDict],   # Theta^(l)
        expectations: Expectations,
        p_tensor: PTensor,
        u_tensor: UTensor,
        device=None,
        dtype=torch.float64,
    ):
        self.kernels = kernels
        self.ntk_layers = ntk_layers
        self.expectations = expectations
        self.P = p_tensor
        self.U = u_tensor
        self.n_layers = len(kernels)
        self.device = device
        self.dtype = dtype

    def _alpha_type_from_pair(self, pair_key: str) -> str:
        if pair_key == "tetr":
            return "test"
        if pair_key == "trte":
            return "train"
        if pair_key == "trtr_offdiag":
            return "train"
        raise ValueError(f"Unknown pair_key={pair_key}")

    def M(self, a: int, b: int, pair_key: str) -> torch.Tensor:
        """
        M_{alpha beta}^{a->b}
        """
        if pair_key not in self.PAIR_KEYS:
            raise ValueError(f"pair_key must be one of {self.PAIR_KEYS}")

        alpha_type = self._alpha_type_from_pair(pair_key)
        out = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for i in range(a, b):
            term = (
                0.5
                * self.expectations.E_act_pp_act[i - 1][pair_key]
                * self.U.compute(a, i, alpha_type)
                * self.P.compute(i + 1, b)[pair_key]
            )
            out = out + term

        return out

    def T(self, a: int, b: int, pair_key: str) -> torch.Tensor:
        """
        T_{alpha beta}^{a->b}
        """
        if pair_key not in self.PAIR_KEYS:
            raise ValueError(f"pair_key must be one of {self.PAIR_KEYS}")

        out = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for i in range(a, b):
            if pair_key in {"tetr", "trte"}:
                theta_i = self.ntk_layers[i - 1]["tetr"]
            else:
                theta_i = self.ntk_layers[i - 1][pair_key]
            term = (
                (
                    self.expectations.E_act_p_act_p[i - 1][pair_key]
                    + theta_i * self.expectations.E_act_pp_act_pp[i - 1][pair_key]
                )
                * self.P.compute(a, i)[pair_key]
                * self.P.compute(i + 1, b)[pair_key]
            )
            out = out + term

        out = out + self.P.compute(a, b)[pair_key]
        return out

    def S(self, a: int, b: int, pair_key: str) -> torch.Tensor:
        """
        S_{alpha beta}^{a->b}
        coefficient multiplying the alpha-alpha source in Theta_{alpha beta}
        """
        if pair_key not in self.PAIR_KEYS:
            raise ValueError(f"pair_key must be one of {self.PAIR_KEYS}")

        alpha_type = self._alpha_type_from_pair(pair_key)
        out = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for i in range(a, b):
            if pair_key in {"tetr", "trte"}:
                theta_i = self.ntk_layers[i - 1]["tetr"]
            else:
                theta_i = self.ntk_layers[i - 1][pair_key]

            first_piece = 0.5 * (
                self.expectations.E_act_pp_act[i - 1][pair_key]
                + theta_i * self.expectations.E_act_ppp_act_p[i - 1][pair_key]
            ) * self.U.compute(a, i, alpha_type) * self.P.compute(i + 1, b)[pair_key]

            second_piece = (
                self.expectations.E_act_p_act_p[i - 1][pair_key]
                + theta_i * self.expectations.E_act_pp_act_pp[i - 1][pair_key]
            ) * self.M(a, i, pair_key) * self.P.compute(i + 1, b)[pair_key]

            out = out + first_piece + second_piece

        return out

    def final_noise_coefficients(self) -> Dict[str, torch.Tensor]:
        """
        Returns the coefficients needed for your final noise variance formula.

        IMPORTANT:
            chi chi means OFFDIAGONAL train-train, i.e. pair_key='trtr_offdiag'
        """
        L = self.n_layers

        S_chi_star = self.S(1, L, "trte")
        S_chi_chi = self.S(1, L, "trtr_offdiag")
        T_chi_star = self.T(1, L, "trte")   # or "tetr"; same here by symmetry
        T_chi_chi = self.T(1, L, "trtr_offdiag")

        return {
            "S_chi_star": S_chi_star,
            "S_chi_chi": S_chi_chi,
            "T_chi_star": T_chi_star,
            "T_chi_chi": T_chi_chi,
        }


@dataclass
class EpsilonCorrectionConfig:
    eps: float
    y_train: torch.Tensor          # shape (N, C), one-hot or soft labels
    num_classes: int
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64


@dataclass
class FirstLayerTrainStats:
    x_sq: torch.Tensor        # shape (N,)
    total_sum: torch.Tensor   # shape (d,)
    class_sums: torch.Tensor  # shape (C, d)


def precompute_first_layer_train_stats(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
) -> FirstLayerTrainStats:
    """
    Precompute train-only statistics reused across all test points.

    This avoids recomputing large reductions for every test input and keeps the
    additive-Gaussian second-order path memory-light.
    """
    if X_train.ndim != 2:
        raise ValueError("X_train must have shape (N, d).")
    if y_train.ndim != 2:
        raise ValueError("y_train must have shape (N, C).")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of training examples.")
    if y_train.shape[1] != num_classes:
        raise ValueError("y_train second dimension must equal num_classes.")

    N, d = X_train.shape
    scale = torch.as_tensor(float(d), device=X_train.device, dtype=X_train.dtype)
    x_sq = torch.einsum("nd,nd->n", X_train, X_train) / scale
    total_sum = torch.sum(X_train, dim=0)
    class_sums = torch.matmul(y_train.T, X_train)

    return FirstLayerTrainStats(
        x_sq=x_sq,
        total_sum=total_sum,
        class_sums=class_sums,
    )


class FirstLayerDerivatives:
    """
    Computes first-layer epsilon-derivative quantities needed for the perturbative correction.

    For each class i, stores:
        DeltaE_diag_dK_chichi[i]
        DeltaE_dK_starchi[i]
        E_DeltaE_dK_chichi[i]

    where
        DeltaE[v]_i = E[v | class i] - E[v]
    """

    def __init__(
        self,
        pure_cfg,
        eps_cfg: EpsilonCorrectionConfig,
        train_stats: Optional[FirstLayerTrainStats] = None,
    ):
        self.pure_cfg = pure_cfg
        self.eps_cfg = eps_cfg
        self.train_stats = train_stats

        self.x_test = pure_cfg.x_test.to(device=eps_cfg.device, dtype=eps_cfg.dtype)
        self.X_train = pure_cfg.X_train.to(device=eps_cfg.device, dtype=eps_cfg.dtype)
        self.y_train = eps_cfg.y_train.to(device=eps_cfg.device, dtype=eps_cfg.dtype)

        self.N, self.d = self.X_train.shape
        _, self.C = self.y_train.shape

        if self.C != eps_cfg.num_classes:
            raise ValueError("y_train second dimension must equal num_classes.")

        self.dK_chichi = None              # (N, N)
        self.dK_starchi = None             # (N,)
        self.diag_dK_chichi = None         # (N,)

        self.x_sq = None                   # (N,)
        self.x_mean = None                 # (N,)
        self.xstar_x = None                # (N,)

        self.deltaE_diag_dK_chichi = None  # (C,)
        self.deltaE_dK_starchi = None      # (C,)
        self.E_deltaE_dK_chichi = None     # (C,)
        self.deltaE_x_sq = None            # (C,)
        self.deltaE_x_mean = None          # (C,)
        self.E_deltaE_xx = None            # (C,)

        self._build()

    def _delta_class_mean_vector(self, v: torch.Tensor) -> torch.Tensor:
        """
        v shape: (N,)
        returns shape: (C,)
        """
        overall = torch.mean(v)
        class_means = self.eps_cfg.num_classes / self.N * torch.sum(
            v[:, None] * self.y_train, dim=0
        )
        return class_means - overall

    def _E_delta_class_mean_matrix(self, M: torch.Tensor) -> torch.Tensor:
        """
        M shape: (N, N)
        First compute rowwise class-centered means, then average over rows.
        returns shape: (C,)
        """
        row_overall = torch.mean(M, dim=1)  # (N,)
        row_class_means = self.eps_cfg.num_classes / self.N * torch.matmul(M, self.y_train)  # (N, C)
        row_delta = row_class_means - row_overall[:, None]  # (N, C)
        return torch.mean(row_delta, dim=0)  # (C,)

    def _build(self):
        Cw = torch.as_tensor(self.pure_cfg.Cw, device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)

        if self.pure_cfg.noise_model == "additive_gaussian":
            if self.train_stats is None:
                train_stats = precompute_first_layer_train_stats(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    num_classes=self.eps_cfg.num_classes,
                )
            else:
                train_stats = self.train_stats

            x_sq = train_stats.x_sq.to(device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)
            total_sum = train_stats.total_sum.to(device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)
            class_sums = train_stats.class_sums.to(device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)

            self.x_sq = x_sq
            self.x_mean = torch.mean(self.X_train, dim=1)
            self.xstar_x = torch.matmul(self.X_train, self.x_test) / self.d

            # dK_chichi: diagonal only, constant -2 Cw
            self.dK_chichi = None
            self.diag_dK_chichi = -2.0 * Cw * torch.ones(
                self.N,
                device=self.eps_cfg.device,
                dtype=self.eps_cfg.dtype,
            )
            self.deltaE_diag_dK_chichi = torch.zeros(
                self.C,
                device=self.eps_cfg.device,
                dtype=self.eps_cfg.dtype,
            )
            self.E_deltaE_dK_chichi = torch.zeros(
                self.C,
                device=self.eps_cfg.device,
                dtype=self.eps_cfg.dtype,
            )

            # dK_starchi(alpha) = Cw * <x_* x_alpha>
            self.dK_starchi = Cw * self.xstar_x

            # Efficient row-averaged class-centered statistic:
            # E[DeltaE[<x x^T>]_i] = C/(N^2 d) total_sum . class_sum_i - ||total_sum||^2/(N^2 d)
            pairwise_scale = torch.as_tensor(
                float(self.N * self.N * self.d),
                device=self.eps_cfg.device,
                dtype=self.eps_cfg.dtype,
            )
            class_pairwise = self.eps_cfg.num_classes * torch.matmul(class_sums, total_sum) / pairwise_scale
            overall_pairwise = torch.dot(total_sum, total_sum) / pairwise_scale
            self.E_deltaE_xx = class_pairwise - overall_pairwise

        elif self.pure_cfg.noise_model == "replacement":
            mu = torch.as_tensor(
                self.pure_cfg.replacement_mean,
                device=self.eps_cfg.device,
                dtype=self.eps_cfg.dtype,
            )
            mu2 = torch.as_tensor(
                self.pure_cfg.replacement_second_moment,
                device=self.eps_cfg.device,
                dtype=self.eps_cfg.dtype,
            )

            mean_X_train = torch.mean(self.X_train, dim=1)  # (N,)
            x_sq = torch.einsum("nd,nd->n", self.X_train, self.X_train) / self.d
            xplusx = mean_X_train[:, None] + mean_X_train[None, :]  # (N, N)
            self.x_mean = mean_X_train
            self.x_sq = x_sq
            self.xstar_x = torch.matmul(self.X_train, self.x_test) / self.d

            diag_part = torch.diag(x_sq - mu2)
            offdiag_part = (
                mu * xplusx
                - 2.0 * mu * mu
                - torch.diag(torch.diag(mu * xplusx - 2.0 * mu * mu))
            )

            self.dK_chichi = Cw * (diag_part + offdiag_part)

            self.dK_starchi = Cw * (
                self.xstar_x
                - mu * torch.mean(self.x_test)
            )

        else:
            raise ValueError(f"Unknown noise_model={self.pure_cfg.noise_model}")

        if self.diag_dK_chichi is None:
            self.diag_dK_chichi = torch.diag(self.dK_chichi)

        if self.deltaE_diag_dK_chichi is None:
            self.deltaE_diag_dK_chichi = self._delta_class_mean_vector(self.diag_dK_chichi)   # (C,)
        self.deltaE_dK_starchi = self._delta_class_mean_vector(self.dK_starchi)               # (C,)
        if self.E_deltaE_dK_chichi is None:
            self.E_deltaE_dK_chichi = self._E_delta_class_mean_matrix(self.dK_chichi)         # (C,)

        if self.x_sq is not None and self.deltaE_x_sq is None:
            self.deltaE_x_sq = self._delta_class_mean_vector(self.x_sq)
        if self.x_mean is not None and self.deltaE_x_mean is None:
            self.deltaE_x_mean = self._delta_class_mean_vector(self.x_mean)


class PredictionCorrection:
    """
    Perturbative correction to the mean output for each class.

    Returns a vector of shape (C,).
    """

    def __init__(
        self,
        pipe,
        pure_cfg,
        eps_cfg: EpsilonCorrectionConfig,
        train_stats: Optional[FirstLayerTrainStats] = None,
    ):
        self.pipe = pipe
        self.pure_cfg = pure_cfg
        self.eps_cfg = eps_cfg
        self.train_stats = train_stats

        self.first_layer = FirstLayerDerivatives(
            pure_cfg,
            eps_cfg,
            train_stats=train_stats,
        )
        self.coeffs = compute_noise_combination_coeffs(pipe)

        self.output_mean_first_order_correction = self._build_first_order()
        self.output_mean_correction = self.output_mean_first_order_correction

    def _build_first_order(self):
        if self.pure_cfg.noise_model == "additive_gaussian":
            return self._build_additive_gaussian_first_order()
        if self.pure_cfg.noise_model == "replacement":
            return self._build_uniform_replacement_first_order()
        raise ValueError(f"Unknown noise_model={self.pure_cfg.noise_model}")

    def _class_mean_minus_q_overall(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        DeltaE^i[v] = C/N sum_{alpha in class i} v_alpha - q E_alpha[v_alpha].
        """
        class_means = self.eps_cfg.num_classes / self.first_layer.N * torch.sum(
            v[:, None] * self.first_layer.y_train,
            dim=0,
        )
        return class_means - q * torch.mean(v)

    def _build_additive_gaussian_first_order(self):
        """
        First-order additive-Gaussian correction from Eq. final_gaussian_output.

        This branch keeps the finite-N inverse factor
            psi = N / (theta_d - theta_o + N theta_o)
        and therefore works without post-hoc recentering, including cases where
        theta_o is zero or very small.
        """
        N = self.first_layer.N
        C = self.eps_cfg.num_classes
        eps = torch.as_tensor(self.eps_cfg.eps, device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)
        Cw = torch.as_tensor(self.pure_cfg.Cw, device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)

        theta_d = self.pipe.ntk.theta_d
        theta_o = self.pipe.ntk.theta_o
        theta_star = self.pipe.ntk.theta_star
        delta = theta_d - theta_o
        psi = torch.as_tensor(float(N), device=self.eps_cfg.device, dtype=self.eps_cfg.dtype) / (
            delta + theta_o * N
        )
        q = theta_o * psi

        S_chi_star = self.coeffs["S_chi_star"]
        S_chi_chi = self.coeffs["S_chi_chi"]
        T_chi_star = self.coeffs["T_chi_star"]

        delta_xstar_x = self._class_mean_minus_q_overall(self.first_layer.xstar_x, q)
        diagonal_source = -2.0 * (1.0 - q) * (
            S_chi_star - 2.0 * theta_star * psi * S_chi_chi
        )

        correction = (
            Cw
            * eps
            * (N / C)
            / delta
            * (diagonal_source + T_chi_star * delta_xstar_x)
        )
        return correction  # shape (C,)

    def _build_uniform_replacement_first_order(self):
        """
        First-order correction for uniform replacement noise.
        """
        uniform_mean = torch.as_tensor(0.5, device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)
        uniform_second_moment = torch.as_tensor(1.0 / 3.0, device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)
        if not torch.isclose(
            torch.as_tensor(self.pure_cfg.replacement_mean, device=self.eps_cfg.device, dtype=self.eps_cfg.dtype),
            uniform_mean,
        ) or not torch.isclose(
            torch.as_tensor(
                self.pure_cfg.replacement_second_moment,
                device=self.eps_cfg.device,
                dtype=self.eps_cfg.dtype,
            ),
            uniform_second_moment,
        ):
            raise ValueError(
                "The replacement-noise correction is implemented for uniform replacement noise "
                "u ~ Uniform[0, 1]."
            )

        N = self.first_layer.N
        C = self.eps_cfg.num_classes
        eps = torch.as_tensor(self.eps_cfg.eps, device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)
        Cw = torch.as_tensor(self.pure_cfg.Cw, device=self.eps_cfg.device, dtype=self.eps_cfg.dtype)

        theta_d = self.pipe.ntk.theta_d
        theta_o = self.pipe.ntk.theta_o
        theta_star = self.pipe.ntk.theta_star
        delta = theta_d - theta_o
        psi = torch.as_tensor(float(N), device=self.eps_cfg.device, dtype=self.eps_cfg.dtype) / (
            delta + theta_o * N
        )
        q = theta_o * psi

        S_chi_star = self.coeffs["S_chi_star"]
        S_chi_chi = self.coeffs["S_chi_chi"]
        T_chi_star = self.coeffs["T_chi_star"]
        T_chi_chi = self.coeffs["T_chi_chi"]
        A_psi = self.coeffs["A_psi"]
        theta_star_psi = theta_star * psi

        x_test_mean = torch.mean(self.first_layer.x_test)
        delta_x_sq = self._class_mean_minus_q_overall(self.first_layer.x_sq, q)
        delta_xstar_x = self._class_mean_minus_q_overall(self.first_layer.xstar_x, q)
        delta_x_mean = self._class_mean_minus_q_overall(self.first_layer.x_mean, q)
        mean_x_sq = torch.mean(self.first_layer.x_sq)
        mean_x = torch.mean(self.first_layer.x_mean)

        bracket = (
            A_psi * (delta_x_sq - (1.0 / 3.0) * (1.0 - q))
            + T_chi_star * (delta_xstar_x - 0.5 * x_test_mean * (1.0 - q))
            - theta_star_psi
            * (
                0.5
                * T_chi_chi
                * ((1.0 - q) * (mean_x - 1.0) + delta_x_mean)
                + S_chi_chi * (1.0 - q) * (mean_x_sq - (1.0 / 3.0))
            )
        )

        return Cw * eps * (N / C) * bracket / delta

class Prediction:
    """
    Pure-noise mean output for equally partitioned classes.
    """

    def __init__(self, pnntk: PureNoiseNTK, n_classes: int, n_train: int):
        self.pnntk = pnntk
        self.n_classes = n_classes
        self.n_train = n_train
        N = torch.as_tensor(n_train, device=pnntk.theta_d.device, dtype=pnntk.theta_d.dtype)
        psi = N / (pnntk.theta_d - pnntk.theta_o + pnntk.theta_o * N)
        self.psi = psi
        self.output_mean = pnntk.theta_star * psi * (1.0 / n_classes)


class PureNoisePipeline:
    """
    One-stop wrapper to build:
        kernels
        expectations
        pure-noise NTK
        pure-noise mean prediction
        P, U, M, S, T coefficients
    """

    def __init__(
        self,
        cfg: PureNoiseConfig,
        act_p,
        act_pp,
        act_ppp,
        n_classes: int,
        act_pppp=None,
    ):
        self.cfg = cfg
        self.kernel_engine = PureNoiseKernel(cfg)

        self.expectations = Expectations(
            kernels=self.kernel_engine.kernels,
            act=cfg.act,
            act_p=act_p,
            act_pp=act_pp,
            act_ppp=act_ppp,
            act_pppp=act_pppp,
            gh_1d_n=cfg.gh_1d_n,
            gh_2d_n=cfg.gh_2d_n,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        self.p_tensor = PTensor(
            self.expectations.E_act_p_act_p,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        self.u_tensor = UTensor(
            self.expectations.E_act_p_act_p,
            self.expectations.E_act_pp_act,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        self.ntk = PureNoiseNTK(
            kernels=self.kernel_engine.kernels,
            E_act_act=self.expectations.E_act_act,
            E_act_p_act_p=self.expectations.E_act_p_act_p,
            Cb=cfg.Cb,
            Cw=cfg.Cw,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        self.transfer = TransferCoefficients(
            kernels=self.kernel_engine.kernels,
            ntk_layers=self.ntk.ntk_layers,
            expectations=self.expectations,
            p_tensor=self.p_tensor,
            u_tensor=self.u_tensor,
            device=cfg.device,
            dtype=cfg.dtype,
        )

        self.prediction = Prediction(
            self.ntk,
            n_classes=n_classes,
            n_train=cfg.X_train.shape[0],
        )
        self.prediction_correction = None



def compute_noise_combination_coeffs(pipe: PureNoisePipeline):
    """
    Returns the coefficients entering the leading perturbative formulas.

    The corrected finite-N contraction uses
        psi = N / (theta_d - theta_o + N theta_o)
        A_psi = S_{chi*} - theta_* psi S_{chi chi}

        T_{chi*}
        T_{chi chi}

    where chi chi means OFFDIAGONAL train-train.
    """
    coeffs = pipe.transfer.final_noise_coefficients()

    N = torch.as_tensor(
        pipe.kernel_engine.X_train.shape[0],
        device=pipe.ntk.theta_d.device,
        dtype=pipe.ntk.theta_d.dtype,
    )
    theta_d = pipe.ntk.theta_d
    theta_star = pipe.ntk.theta_star
    theta_o = pipe.ntk.theta_o
    psi = N / (theta_d - theta_o + theta_o * N)
    q = theta_o * psi

    A_psi = coeffs["S_chi_star"] - theta_star * psi * coeffs["S_chi_chi"]
    theta_o_is_nonzero = torch.abs(theta_o) > torch.finfo(theta_o.dtype).eps
    safe_theta_o = torch.where(theta_o_is_nonzero, theta_o, torch.ones_like(theta_o))
    A_ratio_candidate = coeffs["S_chi_star"] - (theta_star / safe_theta_o) * coeffs["S_chi_chi"]
    A_ratio = torch.where(theta_o_is_nonzero, A_ratio_candidate, A_psi)

    return {
        "A": A_psi,
        "A_psi": A_psi,
        "A_ratio": A_ratio,
        "psi": psi,
        "q": q,
        "S_chi_star": coeffs["S_chi_star"],
        "S_chi_chi": coeffs["S_chi_chi"],
        "T_chi_star": coeffs["T_chi_star"],
        "T_chi_chi": coeffs["T_chi_chi"],
    }


def noise_variance(coeffs, pipe, cfg, num_classes: int):
    """
    Computes the class-wise output noise variance for equally partitioned classes.

    Implemented for:
        - additive_gaussian: corrected finite-N psi contraction from final_gaussian_output
        - replacement: corrected uniform-replacement formula

    The replacement branch matches the uniform replacement-noise formulas,
    i.e. u ~ Uniform[0, 1].
    """
    device = cfg.device
    dtype = cfg.dtype

    T_chi_star = coeffs["T_chi_star"]
    T_chi_chi = coeffs["T_chi_chi"]
    A = coeffs["A"]
    S_chi_chi = coeffs["S_chi_chi"]

    theta_star = pipe.ntk.theta_star
    theta_d = pipe.ntk.theta_d
    theta_o = pipe.ntk.theta_o
    psi = coeffs.get("psi")
    if psi is None:
        N_t = torch.as_tensor(cfg.X_train.shape[0], device=device, dtype=dtype)
        psi = N_t / (theta_d - theta_o + theta_o * N_t)
    q = theta_o * psi
    theta_star_psi = theta_star * psi

    x_test = cfg.x_test.to(device=device, dtype=dtype)
    N, d = cfg.X_train.shape
    Cw = torch.as_tensor(cfg.Cw, device=device, dtype=dtype)
    x_test_mean = torch.mean(x_test)
    x_test_sq_mean = torch.mean(x_test**2)

    if cfg.noise_model == "additive_gaussian":
        A_psi = coeffs.get("A_psi", A)

        v_eta_tilde_d = Cw**2 * (
            2.0
            * A_psi**2
            * (1.0 - (q / num_classes) * (2.0 - q))
            - (4.0 / num_classes)
            * A_psi
            * coeffs["S_chi_chi"]
            * q
            * (1.0 - q) ** 2
            + x_test_sq_mean
            * T_chi_star**2
            * (1.0 - (q / num_classes) * (2.0 - q))
            + (2.0 / num_classes)
            * (coeffs["S_chi_chi"] * q * (1.0 - q)) ** 2
        )

        return (
            torch.as_tensor(N / (num_classes * d), device=device, dtype=dtype)
            * v_eta_tilde_d
            / (theta_d - theta_o) ** 2
        )

    elif cfg.noise_model == "replacement":
        uniform_mean = torch.tensor(0.5, device=device, dtype=dtype)
        uniform_second_moment = torch.tensor(1.0 / 3.0, device=device, dtype=dtype)
        if not torch.isclose(
            torch.as_tensor(cfg.replacement_mean, device=device, dtype=dtype),
            uniform_mean,
        ) or not torch.isclose(
            torch.as_tensor(cfg.replacement_second_moment, device=device, dtype=dtype),
            uniform_second_moment,
        ):
            raise ValueError(
                "The replacement-noise analytical variance is implemented for uniform "
                "replacement noise u ~ Uniform[0, 1], which requires "
                "replacement_mean=0.5 and replacement_second_moment=1/3."
            )

        C = torch.as_tensor(num_classes, device=device, dtype=dtype)
        one_minus_q = 1.0 - q
        label_factor = 1.0 - (q / C) * (2.0 - q)
        pair_factor = 1.0 - (q * (2.0 - q) - one_minus_q**2) / C
        pair_factor_3 = 1.0 - (q * (2.0 - q) - 3.0 * one_minus_q**2) / C

        v_eta_tilde_d = Cw**2 * (
            (4.0 / 45.0) * A**2 * label_factor
            + (x_test_mean / 6.0) * A * T_chi_star * label_factor
            - (1.0 / 12.0) * A * theta_star_psi * T_chi_chi * pair_factor
            - (8.0 / (45.0 * C)) * A * theta_star_psi * S_chi_chi * one_minus_q**2
            + (x_test_sq_mean / 12.0) * T_chi_star**2 * label_factor
            - (x_test_mean / 12.0) * T_chi_star * theta_star_psi * T_chi_chi * pair_factor
            - (x_test_mean / (6.0 * C)) * T_chi_star * theta_star_psi * S_chi_chi * one_minus_q**2
            + (1.0 / 48.0) * (theta_star_psi * T_chi_chi) ** 2 * pair_factor_3
            + (1.0 / (6.0 * C)) * (theta_star_psi * one_minus_q) ** 2 * T_chi_chi * S_chi_chi
            + (4.0 / (45.0 * C)) * (theta_star_psi * one_minus_q * S_chi_chi) ** 2
        )

        return (
            torch.as_tensor(N / (num_classes * d), device=device, dtype=dtype)
            * v_eta_tilde_d
            / (theta_d - theta_o) ** 2
        )

    else:
        raise ValueError(f"Unknown noise_model={cfg.noise_model}")

def SNR_obj(coeffs):
    """
    Computes the signal to noise ratio (SNR) objective function to minimize.

    This is the activation-dependent proxy (A / T_{chi*})^2.
    """
    T_chi_star = coeffs["T_chi_star"]
    A = coeffs["A"]

    return (A / T_chi_star)**2



if __name__ == "__main__":
    torch.manual_seed(0)

    d = 32
    N = 100
    C = 5

    x_test = torch.randn(d)
    X_train = torch.randn(N, d)

    def act(z):
        return torch.tanh(z)

    def act_p(z):
        t = torch.tanh(z)
        return 1.0 - t * t

    def act_pp(z):
        t = torch.tanh(z)
        return -2.0 * t * (1.0 - t * t)

    def act_ppp(z):
        t = torch.tanh(z)
        s2 = 1.0 - t * t
        return -2.0 * s2 * s2 + 4.0 * t * t * s2

    cfg = PureNoiseConfig(
        noise_model="additive_gaussian",
        x_test=x_test,
        X_train=X_train,
        Cb=0.1,
        Cw=1.5,
        act=act,
        n_layers=4,
    )

    pipe = PureNoisePipeline(
        cfg=cfg,
        act_p=act_p,
        act_pp=act_pp,
        act_ppp=act_ppp,
        n_classes=C,
    )

    print("Pure-noise kernels by layer:")
    for l, K in enumerate(pipe.kernel_engine.kernels, start=1):
        print(f"Layer {l}: {K}")

    print("\nFinal pure-noise NTK:")
    print(pipe.ntk.NTK)

    print("\nNormalized theta quantities:")
    print("theta_d   =", pipe.ntk.theta_d.item())
    print("theta_o   =", pipe.ntk.theta_o.item())
    print("theta_*   =", pipe.ntk.theta_star.item())

    print("\nPure-noise mean output:")
    print(pipe.prediction.output_mean.item())

    coeffs = compute_noise_combination_coeffs(pipe)
    print("\nTransfer coefficients:")
    print("S_chi*      =", coeffs["S_chi_star"].item())
    print("S_chi_chi   =", coeffs["S_chi_chi"].item())   # offdiag train-train
    print("T_chi*      =", coeffs["T_chi_star"].item())
    print("T_chi_chi   =", coeffs["T_chi_chi"].item())   # offdiag train-train
    print("A           =", coeffs["A"].item())
