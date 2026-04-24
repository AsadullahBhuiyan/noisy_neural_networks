import os
import sys
from pathlib import Path


def configure_runtime() -> Path:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
