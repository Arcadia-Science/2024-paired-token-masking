"""Module for calculating masked position probabilities using ESM models and Modal.

This module defines functions and classes to compute the probabilities at masked
positions in sequences using ESM models. It leverages Modal for distributed GPU
computation.
"""

from __future__ import annotations

import modal
import numpy as np
import torch
from modal.functions import gather

from analysis.modal_esm.cache_model import app as cache_app
from analysis.modal_esm.cache_model import download_and_cache_model
from analysis.modal_esm.constants import (
    CONSOLE,
    HOURS,
    IMAGE,
    MINUTES,
    MODEL_STORAGE_PATH,
    PREDICTION_APP,
    VOLUME,
)

with IMAGE.imports():
    import time

    from transformers import AutoTokenizer, EsmForMaskedLM


app = modal.App(name=PREDICTION_APP, image=IMAGE)
app.include(cache_app)


@app.cls(
    volumes={MODEL_STORAGE_PATH: VOLUME},
    gpu="any",
    concurrency_limit=1,
    timeout=10 * MINUTES,
    container_idle_timeout=20,
)
class ESMMaskedPrediction:
    """A Modal class for calculating masked position probabilities batch by batch using ESM models.

    This class is instantiated once per container and will remain active until no calls to
    `predict` have been made for `container_idle_timeout` seconds.

    Notes:
        - The model weights are loaded from a local cache specified by `MODEL_STORAGE_PATH`.
        - The class uses a GPU.
    """

    def __init__(self, model_name: str):
        """Initializes the _ESMMaskedPrediction class with the specified model.

        Args:
            model_name:
                The name of the ESM model to use for predictions. Should be a
                HuggingFace model identifier.

        Notes:
            - The `dtype` attribute is currently unused due to memory errors.
        """
        self.model_name = model_name

    @modal.enter()
    def load_model(self):
        """Loads the ESM model and tokenizer into memory.

        This method is called upon entering the Modal context. It loads the tokenizer and
        model from the local cache, moves the model to the GPU, and sets it to evaluation mode.

        Notes:
            - The model and tokenizer are loaded from `MODEL_STORAGE_PATH`.
            - Assumes that CUDA is available.
        """
        assert torch.cuda.is_available()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=MODEL_STORAGE_PATH,
            local_files_only=True,
        )
        self.model = EsmForMaskedLM.from_pretrained(
            self.model_name,
            cache_dir=MODEL_STORAGE_PATH,
            local_files_only=True,
        )
        self.model.to("cuda")  # type: ignore
        self.model.eval()

    @modal.method()
    def predict(
        self,
        sequence: str,
        masked_positions: list[tuple[int, ...]],
        max_length: int = 512,
    ) -> np.ndarray:
        """Calculates the logits for masked positions in a sequence.

        Args:
            sequence:
                The input sequence as a string. Should be tokenizable by the model's tokenizer.
            masked_positions:
                A list of tuples, where each tuple contains positions (ints) in the
                sequence to be masked. Each tuple corresponds to a separate observation
                in the batch.
            max_length:
                The maximum sequence length. Sequences longer than this will be truncated.

        Returns:
            np.ndarray:
                An array of logits from the model's output, with shape (batch_size,
                sequence_length, vocab_size).
        """
        batch_size = len(masked_positions)

        tokenized_sequence = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = tokenized_sequence["input_ids"].to("cuda")
        attention_mask = tokenized_sequence["attention_mask"].to("cuda")

        input_ids = input_ids.repeat(batch_size, 1)
        attention_mask = attention_mask.repeat(batch_size, 1)

        # Apply mask tokens in batch for each pair of positions
        for batch_idx, mask_positions in enumerate(masked_positions):
            for pos in mask_positions:
                input_ids[batch_idx, pos] = self.tokenizer.mask_token_id

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits.cpu().numpy()


@app.function(
    gpu=None,
    memory=1024,
    cpu=1.0,
    timeout=24 * HOURS,
    container_idle_timeout=20,
)
def predict(
    sequence: str,
    masked_positions: list[tuple[int, ...]],
    model_name: str,
    gpu: str,
    num_gpus: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Calculates masked position probabilities by delegating to `ESMMaskedPrediction`.

    This function acts as the entry point for the prediction process. It divides the masked
    positions into batches, spawns GPU workers to process each batch, and gathers the results.

    Args:
        sequence:
            The input sequence as a string.
        masked_positions:
            A list of tuples, where each tuple contains positions (ints) in the sequence
            to be masked.
        model_name:
            The name of the ESM model to use for predictions.
        gpu:
            The type of GPU to use. Can be "any", "A100", etc.
        num_gpus:
            The number of GPU workers to spawn.
        batch_size:
            The number of masked position tuples to process in each batch.
        max_length:
            The maximum sequence length for tokenization.

    Returns:
        np.ndarray:
            An array containing the stacked logits from all batches (sequence_length,
            vocab_size).

    Notes:
        - This function runs on CPU and orchestrates the GPU workers.
        - The results from all batches are gathered and stacked into a single numpy array.
        - Care must be taken to ensure the the GPU chosen can withstand the memory
          requirements of the model and batch size.
    """

    download_and_cache_model.remote(model_name, force_download=False)

    start_time = time.time()

    CONSOLE.print(f"Using model {model_name}.")

    predict_cls = ESMMaskedPrediction.with_options(gpu=gpu, concurrency_limit=num_gpus)  # type: ignore
    worker_factory = predict_cls(model_name)

    CONSOLE.print("Submitting jobs...")

    batch_jobs = []
    for idx in range(0, len(masked_positions), batch_size):
        batch = masked_positions[idx : idx + batch_size]
        batch_job = worker_factory.predict.spawn(sequence, batch, max_length)
        batch_jobs.append(batch_job)

    CONSOLE.print("Gathering results...")

    results = gather(*batch_jobs)

    CONSOLE.print("Stacking results...")

    stacked_results = np.vstack(results)

    elapsed = time.time() - start_time
    CONSOLE.print(f"Finished in {elapsed} seconds.")

    return stacked_results
