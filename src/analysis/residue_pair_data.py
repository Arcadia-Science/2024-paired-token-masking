from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from analysis.utils import amino_acids


def get_jensenshannon_divergence(array1, array2, **kwargs) -> float:
    result = jensenshannon(array1, array2, **kwargs)

    # https://github.com/scipy/scipy/issues/20083
    if np.isnan(result):
        result = 0

    return float(result)


@dataclass
class ResiduePairData:
    """Stats for a residue pair.

    Attributes:
        position_i: The ith residue position.
        position_j: The jth residue position.
        amino_acid_i: The original amino acid at the ith residue.
        most_probable_i_i: The amino acid with the highest probability in p(i|{i}).
        most_probable_i_ij: The amino acid with the highest probability in p(i|{i,j}).
        amino_acid_j: The original amino acid at the jth residue.
        most_probable_j_j: The amino acid with the highest probability in p(j|{j}).
        most_probable_j_ij: The amino acid with the highest probability in p(j|{i,j}).
        perplex_i_i: The perplexity of p(i|{i}).
        perplex_i_ij: The perplexity of p(i|{i,j}).
        perplex_j_i: The perplexity of p(j|{j}).
        perplex_j_ij: The perplexity of p(j|{i,j}).
        js_div_i: The Jenson-Shannon divergence between p(i|{i}) and p(i|{i,j}).
        js_div_j: The Jenson-Shannon divergence between p(j|{j}) and p(j|{i,j}).
        js_div_avg: (js_div_i + js_div_j) / 2.
    """

    position_i: int
    position_j: int
    amino_acid_i: str
    most_probable_i_i: str
    most_probable_i_ij: str
    amino_acid_j: str
    most_probable_j_j: str
    most_probable_j_ij: str
    perplex_i_i: float
    perplex_i_ij: float
    perplex_j_j: float
    perplex_j_ij: float
    js_div_i: float
    js_div_j: float
    js_div_avg: float


@dataclass
class Accumulator:
    results: list[ResiduePairData] = field(default_factory=list)

    def add(self, result: ResiduePairData) -> None:
        self.results.append(result)

    def to_frame(self) -> pd.DataFrame:
        data = [asdict(record) for record in self.results]
        return pd.DataFrame(data)


def calculate_pairwise_data(
    sequence: str,
    single_probs: np.ndarray,
    double_probs: np.ndarray,
    double_masks: list[tuple[int, int]],
):
    vocab_size = len(amino_acids)
    accumulator = Accumulator()

    for idx, (pos_i, pos_j) in enumerate(double_masks):
        p_i_i = single_probs[pos_i]  # p(i|M={i})
        p_j_j = single_probs[pos_j]  # p(j|M={j})
        p_i_ij = double_probs[idx, 0, :]  # p(i|M={i,j})
        p_j_ij = double_probs[idx, 1, :]  # p(j|M={i,j})

        js_div_i = get_jensenshannon_divergence(p_i_ij, p_i_i, base=2)
        js_div_j = get_jensenshannon_divergence(p_j_ij, p_j_j, base=2)

        # Entropy in base 20
        entropy_i_i = -np.sum(p_i_i * np.log(p_i_i) / np.log(vocab_size))
        entropy_j_j = -np.sum(p_j_j * np.log(p_j_j) / np.log(vocab_size))
        entropy_i_ij = -np.sum(p_i_ij * np.log(p_i_ij) / np.log(vocab_size))
        entropy_j_ij = -np.sum(p_j_ij * np.log(p_j_ij) / np.log(vocab_size))

        # Perplexity is "effective number of options"
        perplexity_i_i = np.power(vocab_size, entropy_i_i)
        perplexity_j_j = np.power(vocab_size, entropy_j_j)
        perplexity_i_ij = np.power(vocab_size, entropy_i_ij)
        perplexity_j_ij = np.power(vocab_size, entropy_j_ij)

        most_probable_i_i = amino_acids[p_i_i.argmax()]
        most_probable_j_j = amino_acids[p_j_j.argmax()]
        most_probable_i_ij = amino_acids[p_i_ij.argmax()]
        most_probable_j_ij = amino_acids[p_j_ij.argmax()]

        result = ResiduePairData(
            pos_i,
            pos_j,
            sequence[pos_i],
            most_probable_i_i,
            most_probable_i_ij,
            sequence[pos_j],
            most_probable_j_j,
            most_probable_j_ij,
            perplexity_i_i,
            perplexity_i_ij,
            perplexity_j_j,
            perplexity_j_ij,
            js_div_i,
            js_div_j,
            (js_div_i + js_div_j) / 2,
        )
        accumulator.add(result)

    return _complete_triangular_data(accumulator.to_frame())


def _complete_triangular_data(df: pd.DataFrame) -> pd.DataFrame:
    mirrored_rows = []
    for _, row in df.iterrows():
        if row["position_i"] < row["position_j"]:
            mirrored_rows.append(_mirror(row))

    mirrored_df = pd.DataFrame(mirrored_rows)
    complete_df = pd.concat([df, mirrored_df], ignore_index=True)

    return complete_df.sort_values(by=["position_i", "position_j"])


def _mirror(row: pd.Series) -> pd.Series:
    new = row.copy()
    new["position_i"] = row["position_j"]
    new["position_j"] = row["position_i"]
    new["js_div_i"] = row["js_div_j"]
    new["js_div_j"] = row["js_div_i"]
    new["amino_acid_i"] = row["amino_acid_j"]
    new["amino_acid_j"] = row["amino_acid_i"]
    new["perplex_i_i"] = row["perplex_j_j"]
    new["perplex_i_ij"] = row["perplex_j_ij"]
    new["perplex_j_j"] = row["perplex_i_i"]
    new["perplex_j_ij"] = row["perplex_i_ij"]
    new["most_probable_i_i"] = row["most_probable_j_j"]
    new["most_probable_j_j"] = row["most_probable_i_i"]
    new["most_probable_i_ij"] = row["most_probable_j_ij"]
    new["most_probable_j_ij"] = row["most_probable_i_ij"]
    return new
