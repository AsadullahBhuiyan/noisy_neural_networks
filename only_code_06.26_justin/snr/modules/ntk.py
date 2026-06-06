from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .data import PreparedDataset
from .metrics import acc_from_logits, mse_onehot_from_logits


def build_stax_network(cfg: ExperimentConfig):
    from jax import jit
    import jax
    import jax.numpy as jnp
    from neural_tangents import stax

    act_name = cfg.act.strip().lower()

    if act_name == "relu":
        raise ValueError("ReLU is intentionally disabled here; use a smooth activation.")
    if act_name == "erf":
        activation_layer = stax.Erf()
    elif act_name == "gelu":
        activation_layer = stax.Gelu()
    elif act_name == "sin":
        activation_layer = stax.Sin()
    elif act_name in {"tanh", "softplus"}:
        if not hasattr(stax, "ElementwiseNumerical"):
            raise ValueError(
                "This neural_tangents version does not expose stax.ElementwiseNumerical, "
                f"which is needed for activation {cfg.act!r}."
            )
        if act_name == "tanh":
            activation_fn = jnp.tanh
            activation_df = lambda z: 1.0 - jnp.tanh(z) ** 2
        else:
            activation_fn = jax.nn.softplus
            activation_df = jax.nn.sigmoid
        try:
            activation_layer = stax.ElementwiseNumerical(
                activation_fn,
                deg=40,
                df=activation_df,
            )
        except TypeError:
            activation_layer = stax.ElementwiseNumerical(activation_fn)
    else:
        raise ValueError("cfg.act must be one of: 'erf', 'gelu', 'tanh', 'sin', 'softplus'.")

    layers = []
    for _ in range(max(1, int(cfg.depth) - 1)):
        layers.extend(
            [
                stax.Dense(int(cfg.width), W_std=cfg.w_std, b_std=cfg.b_std),
                activation_layer,
            ]
        )
    layers.append(stax.Dense(int(cfg.num_classes), W_std=cfg.w_std, b_std=cfg.b_std))

    _, _, kernel_fn = stax.serial(*layers)
    return jit(kernel_fn, static_argnames="get")


@dataclass
class NTKRunResult:
    train_logits: np.ndarray
    test_logits: np.ndarray
    train_acc: float
    train_mse: float
    test_acc: float
    test_mse: float


def run_ntk_inference(cfg: ExperimentConfig, data: PreparedDataset) -> NTKRunResult:
    import jax.numpy as jnp
    import neural_tangents as nt

    kernel_fn = build_stax_network(cfg)

    x_train = jnp.asarray(data.x_train_noisy)
    y_train = jnp.asarray(data.y_train_onehot)
    x_test = jnp.asarray(data.x_test_clean)

    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn,
        x_train,
        y_train,
        diag_reg=float(cfg.diag_reg),
    )

    train_mean = predict_fn(x_test=x_train, get="ntk", compute_cov=False)
    test_mean = predict_fn(x_test=x_test, get="ntk", compute_cov=False)

    train_logits = np.asarray(train_mean).reshape((data.x_train_noisy.shape[0], cfg.num_classes))
    test_logits = np.asarray(test_mean).reshape((data.x_test_clean.shape[0], cfg.num_classes))

    return NTKRunResult(
        train_logits=train_logits,
        test_logits=test_logits,
        train_acc=acc_from_logits(train_logits, data.y_train),
        train_mse=mse_onehot_from_logits(train_logits, data.y_train, cfg.num_classes),
        test_acc=acc_from_logits(test_logits, data.y_test),
        test_mse=mse_onehot_from_logits(test_logits, data.y_test, cfg.num_classes),
    )
