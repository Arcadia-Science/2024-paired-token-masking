from __future__ import annotations

import modal
from rich.console import Console

console: Console = Console()

volume: modal.Volume = modal.Volume.from_name("esm-model-cache", create_if_missing=True)
model_storage_path: str = "/vol"
image: modal.Image = (
    modal.Image.debian_slim()
    .apt_install("libpq-dev")
    .pip_install("transformers", "torch", "pandas")
)

MINUTES: int = 60
HOURS: int = MINUTES * 60
PREDICTION_APP_NAME: str = "esm-masked-prediction"
CACHING_APP_NAME: str = "download-and-cache-models"
