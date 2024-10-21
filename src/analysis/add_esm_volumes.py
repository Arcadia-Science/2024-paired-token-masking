import modal
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from rich.console import Console
from transformers import AutoModel, AutoTokenizer

VOLUME = modal.Volume.from_name("model-cache", create_if_missing=True)
MODEL_STORAGE_PATH = "/vol"

image = modal.Image.debian_slim().apt_install("libpq-dev").pip_install("transformers", "torch")

app = modal.App(name="download-model", image=image)

console = Console()


def model_exists(model_name: str) -> bool:
    """Returns whether a HuggingFace model is cached in the remote volume.

    The remote volume is hard-coded as `model-cache` and the storage path models is
    `/vol`.

    Args:
        model_name:
            The name of the model, e.g. "facebook/esm2_t6_8M_UR50D". Any HuggingFace
            model identifier can be used.

    Notes:
        - This only checks whether the `config.json` for the model is present, so it's
          not a bulletproof check.
    """
    try:
        hf_hub_download(
            model_name,
            "config.json",
            cache_dir=MODEL_STORAGE_PATH,
            local_files_only=True,
        )
        return True
    except LocalEntryNotFoundError:
        return False


def tokenizer_exists(model_name: str) -> bool:
    """Returns whether a HuggingFace tokenizer is cached in the remote volume.

    The remote volume is hard-coded as `model-cache` and the storage path models is
    `/vol`.

    Args:
        model_name:
            The name of the model, e.g. "facebook/esm2_t6_8M_UR50D". Any HuggingFace
            model identifier can be used.
    """
    try:
        hf_hub_download(
            model_name,
            "tokenizer_config.json",
            cache_dir=MODEL_STORAGE_PATH,
            local_files_only=True,
        )
        return True
    except LocalEntryNotFoundError:
        return False


@app.function(volumes={MODEL_STORAGE_PATH: VOLUME}, timeout=7200)
def download_and_cache_model(model_name: str, force_download: bool):
    """Download and cache a HuggingFace model in the remote volume.

    This function checks whether a model and its tokenizer are already cached in the
    remote volume. If either are missing, they will be downloaded and cached.

    Args:
        model_name:
            The name of the model, e.g. "facebook/esm2_t6_8M_UR50D". Any HuggingFace
            model identifier can be used.
        force_download:
            If True, the model and its tokenizer will be downloaded and overwrite any
            existing caches.
    """

    modified: bool = False

    if model_exists(model_name) and not force_download:
        console.print(f"{model_name} model already stored in volume.", style="green")
    else:
        AutoModel.from_pretrained(
            model_name,
            cache_dir=MODEL_STORAGE_PATH,
            force_download=force_download,
        )
        console.print(f"{model_name} model is now stored in volume.", style="green")
        modified = True

    if tokenizer_exists(model_name) and not force_download:
        console.print(f"{model_name} tokenizer already stored in volume.", style="green")
    else:
        AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODEL_STORAGE_PATH,
            force_download=force_download,
        )
        console.print(f"{model_name} tokenizer is now stored in volume.", style="green")
        modified = True

    if modified:
        VOLUME.commit()


@app.local_entrypoint()
def main(model_name: str, force_download: bool = False):
    download_and_cache_model.remote(model_name, force_download)
