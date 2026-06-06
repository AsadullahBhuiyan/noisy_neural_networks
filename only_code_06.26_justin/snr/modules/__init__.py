from .config import DEFAULT_CONFIG, ExperimentConfig


def run_experiment(*args, **kwargs):
    from .pipeline import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)

__all__ = ["DEFAULT_CONFIG", "ExperimentConfig", "run_experiment"]
