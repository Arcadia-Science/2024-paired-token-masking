from __future__ import annotations

import modal
import numpy as np
import torch
from modal.functions import gather

APP_NAME = "esm-masked-prediction"
VOLUME = modal.Volume.from_name("model-cache", create_if_missing=True)
MODEL_STORAGE_PATH = "/vol"
MINUTES = 60
HOURS = MINUTES * 60

image = (
    modal.Image.debian_slim()
    .apt_install("libpq-dev")  # TODO(ek): remove if possible
    .pip_install("transformers", "torch", "pandas")
)

with image.imports():
    import time

    from transformers import AutoTokenizer, EsmForMaskedLM


app = modal.App(name=APP_NAME, image=image)


@app.cls(
    volumes={MODEL_STORAGE_PATH: VOLUME},
    gpu="any",
    concurrency_limit=1,
    timeout=10 * MINUTES,
    container_idle_timeout=20,
)
class _ESMMaskedPrediction:
    """Calculates masked position probabilities batch by batch.

    This is instanced once per container and will remain active until no calls to
    `predict` have been made for `container_idle_timeout` seconds.
    """

    def __init__(self, model_name: str, fp16: bool = False):
        self.model_name = model_name

        # TODO(ek): Currently unused because it was leading to memory errors? I'm
        # curious what the model weights are represented as. Perhaps we should be doing
        # 8-bit or somsething
        self.dtype = torch.float16 if fp16 else torch.float32

    @modal.enter()
    def load_model(self):
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
        self.model.to("cuda")
        self.model.eval()

    @modal.method()
    def predict(
        self,
        sequence: str,
        masked_positions: list[tuple[int, ...]],
        max_length: int = 512,
    ) -> np.ndarray:
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
    fp16: bool = False,
) -> np.ndarray:
    """Calculates masked position probabilities by delegating to _ESMMaskedPrediction.

    This is instanced once per user call and serves as the head CPU node that spawns
    batch jobs.
    """

    start_time = time.time()

    print(f"Using model {model_name}.")

    predict_cls = _ESMMaskedPrediction.with_options(gpu=gpu, concurrency_limit=num_gpus)
    workers = predict_cls(model_name, fp16)

    print("Submitting jobs...")

    batch_jobs = []
    for idx in range(0, len(masked_positions), batch_size):
        batch = masked_positions[idx : idx + batch_size]
        batch_job = workers.predict.spawn(sequence, batch, max_length)
        batch_jobs.append(batch_job)

    print("Gathering results...")

    results = gather(*batch_jobs)

    print("Stacking results...")

    stacked_results = np.vstack(results)

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed} seconds.")

    return stacked_results
