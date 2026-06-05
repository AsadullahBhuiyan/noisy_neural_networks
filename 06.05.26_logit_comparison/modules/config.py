from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class ExperimentConfig:
    num_classes: int = 10
    train_fraction: float = 0.1
    test_fraction: float = 1.0
    data_root: str = "data"
    img_hw: Tuple[int, int] = (200, 200)

    noise_probability: float = 0.99
    noise_type: str = "replacement"
    diag_reg: float = 1e-12

    width: int = 2048
    depth: int = 1
    act: str = "relu"
    w_std: float = 0.5
    b_std: float = 0.0

    seed: int = 22334
    num_independent_runs: int = 1
    histogram_test_index: int | list[int] = 0
    histogram_label: int | list[int] = 0
    output_dir: str = "outputs"

    def resolve_data_root(self, project_dir: Path) -> Path:
        path = Path(self.data_root)
        return path if path.is_absolute() else project_dir / path

    def resolve_output_dir(self, project_dir: Path) -> Path:
        path = Path(self.output_dir)
        return path if path.is_absolute() else project_dir / path


DEFAULT_CONFIG = ExperimentConfig()
