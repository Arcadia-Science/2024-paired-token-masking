"""Constants for the Modal app(s)."""

from __future__ import annotations

import modal
from rich.console import Console

VOLUME: modal.Volume = modal.Volume.from_name("model-cache", create_if_missing=True)
MODEL_STORAGE_PATH: str = "/vol"
MINUTES: int = 60
HOURS: int = MINUTES * 60

IMAGE: modal.Image = (
    modal.Image.debian_slim()
    .apt_install("libpq-dev")  # TODO(ek): remove if possible
    .pip_install("transformers", "torch", "pandas")
)

PREDICTION_APP: str = "esm-masked-prediction"
CACHING_APP: str = "download-and-cache-models"

CONSOLE: Console = Console()
