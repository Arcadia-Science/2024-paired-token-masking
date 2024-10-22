# 2024-paired-token-masking

## Purpose & Description

Please refer to the publication:  TODO

## Installation

This repository uses conda to manage software environments and installations. You can find operating system-specific instructions for installing miniconda [here](https://docs.conda.io/projects/miniconda/en/latest/). After installing, run the following command to create the environment.

```{bash}
conda env create -n paired-token-masking --file env.yml
conda activate paired-token-masking
pip install -e .
```

<details><summary>Developer Notes (click to expand/collapse)</summary>

1. Install your pre-commit hooks:

    ```{bash}
    pre-commit install
    ```

    This installs the pre-commit hooks defined in your config (`./.pre-commit-config.yaml`).

2. Export your conda environment before sharing:

    As your project develops, the number of dependencies in your environment may increase. Whenever you install new dependencies (using either `pip install` or `mamba install`), you should update the environment file using the following command.

    ```{bash}
    conda env export --from-history --no-builds > envs/dev.yml
    ```

    `--from-history` only exports packages that were explicitly added by you (e.g., the packages you installed with `pip` or `mamba`) and `--no-builds` removes build specification from the exported packages to increase portability between different platforms.
</details>

## Generate the publication

To generate a local copy of the publication, run

```{bash}
make pub
```

The above command assumes that `notebook.ipynb` has already been ran. If the outputs have been cleared, or you would like to re-run the notebook prior to `make pub`, run

```{bash}
make execute
```

## Data

All input and output data are git-tracked in the repository.

**Inputs**:

```{bash}
  input
  │  P00813.fasta
  └  P00813.pdb
```

`P00813.fasta` and `P00813.pdb` are the amino acid sequence and AlphaFold-predicted structure for human adenosine deaminase and were downloaded from the [P00813 UniProt entry](https://www.uniprot.org/uniprotkb/P00813/entry).

**Outputs**:

```{bash}
  output
  │  logits_single.npz
  └  logits_double.npz
```

`logits_single.npz` and `logits_double.npz` store the raw logits from each ESM2 model for the single token mask library and the double token mask library, respectively. NPZ is a Numpy file format for storing many arrays which can be accessed by key values. These arrays are calculated using Modal and are git-tracked to avoid expensive recomputation. For details on usage, see how the publication uses them.

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
