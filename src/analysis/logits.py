from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.modal_esm.predict import app, predict
from analysis.utils import ModelName


@dataclass
class LogitsConfig:
    sequence: str
    single_masks: list[tuple[int]]
    double_masks: list[tuple[int, int]]
    single_logits_path: Path
    double_logits_path: Path


def calculate_or_load_logits(config: LogitsConfig):
    single_logits = _get_logits_if_exists(config.single_logits_path)
    double_logits = _get_logits_if_exists(config.double_logits_path)

    for model, (gpu, batch_size) in gpu_workload_specs.items():
        if model not in single_logits:
            with app.run(show_progress=False):
                all_single_logits = predict.local(
                    sequence=config.sequence,
                    masked_positions=config.single_masks,
                    model_name=model.value,
                    gpu=gpu,
                    num_gpus=1,
                    batch_size=batch_size,
                )
                single_logits[model] = _filter_unmasked_single_logits(
                    all_single_logits, config.single_masks
                )
            print(f"Finished single mask inference with {model.value}...")
        else:
            print(f"Single mask library already loaded for {model.value}. Skipping.")

        if model not in double_logits:
            with app.run(show_progress=False):
                all_double_logits = predict.local(
                    sequence=config.sequence,
                    masked_positions=config.double_masks,
                    model_name=model.value,
                    gpu=gpu,
                    num_gpus=10,
                    batch_size=batch_size,
                )
                double_logits[model] = _filter_unmasked_double_logits(
                    all_double_logits, config.double_masks
                )
            print(f"Finished double mask inference with {model.value}...")
        else:
            print(f"Double mask library already loaded for {model.value}. Skipping.")

        np.savez(config.single_logits_path, **{k.value: v for k, v in single_logits.items()})
        np.savez(config.double_logits_path, **{k.value: v for k, v in double_logits.items()})

    return single_logits, double_logits


# Values are (GPU type, batch size)
gpu_workload_specs = {
    ModelName.ESM2_8M: ("T4", 512),
    ModelName.ESM2_35M: ("T4", 512),
    ModelName.ESM2_150M: ("A10G", 512),
    ModelName.ESM2_650M: ("A10G", 256),
    ModelName.ESM2_3B: ("A100", 256),
    ModelName.ESM2_15B: ("H100", 64),
}


def _get_logits_if_exists(path: Path) -> dict[ModelName, np.ndarray]:
    if not path.exists():
        return {}

    logits_cache = np.load(path)
    return {ModelName(key): logits_cache[key] for key in logits_cache}


def _filter_unmasked_double_logits(array: np.ndarray, masks: list[tuple[int, int]]) -> np.ndarray:
    pairs = np.array(masks) + 1  # Add 1 to account for prepended CLS token
    n_indices = np.arange(len(array))[:, None]  # Shape (N, 1)
    n_indices = np.repeat(n_indices, 2, axis=1)  # Shape (N, 2)
    return array[n_indices, pairs, :]  # Shape (N, 2, L)


def _filter_unmasked_single_logits(array: np.ndarray, masks: list[tuple[int]]) -> np.ndarray:
    masks_array = np.array(masks) + 1  # Add 1 to account for prepended CLS token
    return array[np.arange(len(array)), masks_array.flatten(), :]
