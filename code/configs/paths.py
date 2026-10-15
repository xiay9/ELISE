"""Portable data and generated-state paths (override with ELISE_* variables)."""
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ELISE_DATA_ROOT", REPO_ROOT / "datasets")).expanduser()
ARTIFACT_ROOT = Path(os.environ.get("ELISE_ARTIFACT_ROOT", REPO_ROOT / "artifacts")).expanduser()
CACHE_ROOT = Path(os.environ.get("ELISE_CACHE_ROOT", ARTIFACT_ROOT / "cache")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("ELISE_OUTPUT_ROOT", ARTIFACT_ROOT / "runs")).expanduser()
SCRATCH_ROOT = ARTIFACT_ROOT
WANDB_ROOT = OUTPUT_ROOT / "wandb"
