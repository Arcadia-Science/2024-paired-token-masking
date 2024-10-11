from __future__ import annotations

import modal
import numpy as np
import torch
from modal.functions import gather

APP_NAME = "esm-masked-prediction"

image = (
    modal.Image.debian_slim()
    .apt_install("libpq-dev")
    .pip_install("transformers", "torch", "pandas")
)

with image.imports():
    import time

    from transformers import AutoTokenizer, EsmForMaskedLM

VOLUME = modal.Volume.from_name("model-cache", create_if_missing=True)
MODEL_STORAGE_PATH = "/vol"

app = modal.App(name=APP_NAME, image=image)


@app.cls(
    volumes={MODEL_STORAGE_PATH: VOLUME},
    gpu="any",
    concurrency_limit=1,
)
class ESMMaskedPrediction:
    def __init__(self, model_name: str):
        self.model_name = model_name

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
    def predict_batch(
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

        input_ids = tokenized_sequence['input_ids'].to('cuda')
        attention_mask = tokenized_sequence['attention_mask'].to('cuda')

        input_ids = input_ids.repeat(batch_size, 1)
        attention_mask = attention_mask.repeat(batch_size, 1)

        # Apply mask tokens in batch for each pair of positions
        for batch_idx, mask_positions in enumerate(masked_positions):
            for pos in mask_positions:
                input_ids[batch_idx, pos] = self.tokenizer.mask_token_id

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits.cpu().numpy()

    @staticmethod
    def predict(
        sequence: str,
        masked_positions: list[tuple[int, ...]],
        model_name: str,
        gpu: str,
        num_gpus: int,
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        start_time = time.time()

        predictor = ESMMaskedPrediction.get_predictor(model_name, gpu, num_gpus)

        batch_jobs = []
        for idx in range(0, len(masked_positions), batch_size):
            batch = masked_positions[idx : idx + batch_size]
            batch_job = predictor.predict_batch.spawn(sequence, batch, max_length)
            batch_jobs.append(batch_job)

        results = gather(*batch_jobs)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The jobs took {execution_time} seconds to run.")

        return np.vstack(results)

    @staticmethod
    def get_predictor(model_name: str, gpu: str, num_gpus: int) -> modal.cls.Obj:
        assert app.name is not None
        modal_cls = modal.Cls.lookup(APP_NAME, ESMMaskedPrediction.__name__)
        modified_cls = modal_cls.with_options(gpu=gpu, concurrency_limit=num_gpus)
        return modified_cls(model_name=model_name)
