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
        js_div_i: The Jenson-Shannon divergence between p(i|{i}) and p(i|{i,j}).
        js_div_j: The Jenson-Shannon divergence between p(j|{j}) and p(j|{i,j}).
        js_div_j: (js_div_i + js_div_j) / 2.
        amino_acid_i: The original amino acid at the ith residue.
        amino_acid_j: The original amino acid at the jth residue.
        perplex_i: The perplexity of p(i|{i}).
        perplex_j: The perplexity of p(j|{j}).
        most_probable_i: The amino acid with the highest probability in p(i|{i}).
        most_probable_j: The amino acid with the highest probability in p(j|{j}).
    """

    position_i: int
    position_j: int
    js_div_i: float
    js_div_j: float
    js_div_avg: float
    amino_acid_i: str
    amino_acid_j: str
    perplex_i: float
    perplex_j: float
    most_probable_i: str
    most_probable_j: str


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
        entropy_i = -np.sum(p_i_i * np.log(p_i_i) / np.log(vocab_size))
        entropy_j = -np.sum(p_j_j * np.log(p_j_j) / np.log(vocab_size))

        # Perplexity is "effective number of options"
        perplexity_i = np.power(vocab_size, entropy_i)
        perplexity_j = np.power(vocab_size, entropy_j)

        most_probable_i = amino_acids[p_i_i.argmax()]
        most_probable_j = amino_acids[p_j_j.argmax()]

        result = ResiduePairData(
            pos_i,
            pos_j,
            js_div_i,
            js_div_j,
            (js_div_i + js_div_j) / 2,
            sequence[pos_i],
            sequence[pos_j],
            perplexity_i,
            perplexity_j,
            most_probable_i,
            most_probable_j,
        )
        accumulator.add(result)

    return accumulator.to_frame()
