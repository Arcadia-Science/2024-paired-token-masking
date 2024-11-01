"""Module for downloading and caching model weights / tokenziers to a Modal-hosted remote volume.

By storing the model in a remote-volume, we circumvent the problem of downloading model
weights each time an embedding job is requested. This was especially problematic for the
largest 15B parameter model, which couldn't be reliably downloaded within Modal's maximum
cold-start timeout of 20 minutes.
"""

import modal

from analysis.modal_esm.constants import CACHING_APP, console, image, model_storage_path, volume

app = modal.App(name=CACHING_APP, image=image)

with image.imports():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    from transformers import AutoModel, AutoTokenizer


def model_exists(model_name: str) -> bool:
    """Returns whether a HuggingFace model is cached in the remote volume.

    The remote volume is hard-coded as `esm-model-cache` and the storage path models is
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
            cache_dir=model_storage_path,
            local_files_only=True,
        )
        return True
    except LocalEntryNotFoundError:
        return False


def tokenizer_exists(model_name: str) -> bool:
    """Returns whether a HuggingFace tokenizer is cached in the remote volume.

    The remote volume is hard-coded as `esm-model-cache` and the storage path models is
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
            cache_dir=model_storage_path,
            local_files_only=True,
        )
        return True
    except LocalEntryNotFoundError:
        return False


@app.function(volumes={model_storage_path: volume}, timeout=7200)
def download_and_cache_model(model_name: str, force_download: bool) -> None:
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
            cache_dir=model_storage_path,
            force_download=force_download,
        )
        console.print(f"{model_name} model is now stored in volume.", style="green")
        modified = True

    if tokenizer_exists(model_name) and not force_download:
        console.print(f"{model_name} tokenizer already stored in volume.", style="green")
    else:
        AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_storage_path,
            force_download=force_download,
        )
        console.print(f"{model_name} tokenizer is now stored in volume.", style="green")
        modified = True

    if modified:
        volume.commit()


@app.local_entrypoint()
def main(model_name: str, force_download: bool = False) -> None:
    """The local entrypoint for the app.

    This function defines a CLI for :func:`download_and_cache_model`.

    Example:
        $ modal run cache_model.py --model-name XXX
    """
    download_and_cache_model.remote(model_name, force_download)
