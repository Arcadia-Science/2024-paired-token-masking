# Paired residue prediction dependencies in ESM2

This code repository contains all materials required for creating and hosting the publication entitled, *"Paired residue prediction dependencies in ESM2"*.

The publication is hosted at [this URL](https://arcadia-science.github.io/2024-paired-token-masking/).

## Data Description

All input and output data are git-tracked in the repository.

**Inputs**:

* `inputs/P00813.fasta`: The amino acid sequence for human adenosine deaminase. Downloaded from the P00813 UniProt entry(https://www.uniprot.org/uniprotkb/P00813/entry).
* `inputs/P00813.pdb`: The AlphaFold-predicted structure for human adenosine deaminase. Downloaded from the P00813 UniProt entry(https://www.uniprot.org/uniprotkb/P00813/entry).

**Outputs**:

* `logits_single.npz`: Stores the raw logits of each ESM2 model for the single token mask library. NPZ is a NumPy file format for storing many arrays which can be accessed by key values. This file is calculated using Modal and is git-tracked to avoid expensive recomputation, and so the notebook can be run without needing to run anything using Modal. For details on usage, see how the publication uses it.
* `logits_double.npz`: Same as above, but for the double token mask library.

## Reproduce

Please see [SETUP.qmd](SETUP.qmd).

## Contribute

Please see [CONTRIBUTING.qmd](CONTRIBUTING.qmd).
