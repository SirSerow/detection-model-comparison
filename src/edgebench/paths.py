"""Repository-root path helpers.

Config loaders resolve YAML relative to the repo root so tests and
editable installs work regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
REPO_ROOT = SRC_DIR.parent
CONFIGS_DIR = REPO_ROOT / "configs"
DEVICES_DIR = CONFIGS_DIR / "devices"
MODELS_DIR = CONFIGS_DIR / "models"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"
