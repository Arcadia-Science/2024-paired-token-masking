# 2024-paired-token-masking

## Purpose & Description

Please refer to the publication: TODO

## Installation

This repository uses conda to manage software environments and installations. You can find operating system-specific instructions for installing miniconda [here](https://docs.conda.io/projects/miniconda/en/latest/). After installing, run the following command to create the environment.

```bash
conda env create -n paired-token-masking --file env.yml
conda activate paired-token-masking
pip install -e .
```

<details><summary>Developer Notes (click to expand/collapse)</summary>

1. Install your pre-commit hooks:

    ```bash
    pre-commit install
    ```

    This installs the pre-commit hooks defined in your config (`./.pre-commit-config.yaml`).

2. Export your conda environment before sharing:

    As your project develops, the number of dependencies in your environment may increase. Whenever you install new dependencies (using either `pip install` or `mamba install`), you should update the environment file using the following command.

    ```bash
    conda env export --from-history --no-builds > envs/dev.yml
    ```

    `--from-history` only exports packages that were explicitly added by you (e.g., the packages you installed with `pip` or `conda`) and `--no-builds` removes build specification from the exported packages to increase portability between different platforms.
</details>

## Modify

To modify or extend any analyses, open up `index.ipynb` with Jupyter or your favorite IDE:

```bash
jupyter-lab index.ipynb
```

## Preview

To create a live preview of your pub, run the following:

```bash
make preview
```

This will open a local copy of the publication in your default browser. The command watches for changes such that whenever `index.ipynb` is saved, the publication is re-rendered.

## Publish

Publishing is handled automatically with a GitHub Action whenever a pull request is merged into the `main` branch.However, it is important to note that the action doesn't run the notebook. Instead, the action relies on the user to have run the notebook locally. Thus, it is the user's responsibility to execute the notebook with:

```
make execute
```

Then, `index.ipynb` and any runtime artefacts generated in `_freeze/` should be committed and pushed to the branch. Using these artefacts, the GitHub Action is able to render the publication when the branch is merged, circumventing the need to execute the notebook server side. For more information about this workflow, see Quarto's documentation on publishing with Github Actions [here](https://quarto.org/docs/publishing/github-pages.html#github-action).

## Data

All input and output data are git-tracked in the repository.

**Inputs**:

* `inputs/P00813.fasta`: The amino acid sequence for human adenosine deaminase. Downloaded from the P00813 UniProt entry(https://www.uniprot.org/uniprotkb/P00813/entry).
* `inputs/P00813.pdb`: The AlphaFold-predicted structure for human adenosine deaminase. Downloaded from the P00813 UniProt entry(https://www.uniprot.org/uniprotkb/P00813/entry).

**Outputs**:

* `logits_single.npz`: Stores the raw logits of each ESM2 model for the single token mask library. NPZ is a Numpy file format for storing many arrays which can be accessed by key values. This file is calculated using Modal and is git-tracked to avoid expensive recomputation, and so the notebook can be run without needing to run anything using Modal. For details on usage, see how the publication uses it.
* `logits_double.npz`: Same as above, but for the double token mask library.

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
