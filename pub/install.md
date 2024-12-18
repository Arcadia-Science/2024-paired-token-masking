# Installation

This analysis uses conda to manage software environments and installations. You can find operating system-specific instructions for installing miniconda [here](https://docs.conda.io/projects/miniconda/en/latest/).

## Cloning the repository

The source code for this publication can be found [here](https://github.com/Arcadia-Science/2024-paired-token-masking). If you want to contribute, or modify the analysis for yourself, you'll need to clone that repository:

```bash
git clone https://github.com/Arcadia-Science/2024-paired-token-masking.git
cd 2024-paired-token-masking
```

## Setting up the environment

Then, simply install the conda environment,

```bash
conda env create -n paired-token-masking --file env.yml
```

activate the environment,

```bash
conda activate paired-token-masking
```

and finally, install the package contents.

```bash
pip install -e .
```
