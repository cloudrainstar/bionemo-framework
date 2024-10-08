# bionemo-geneformer

Geneformer is a foundational single-cell RNA (scRNA) language model using a BERT architecture trained on millions of single-cell RNA sequences. It captures gene co-expression patterns to learn cellular representations, enabling predictive tasks across biology and medicine. Geneformer is trained on a masked language model (MLM) objective, where expression rank-ordered "gene tokens" in single-cell RNA sequences are masked, replaced, or left unchanged, and the model learns to predict these masked genes based on context. This module provides Dataset classes, collators for expression rank ordering, and Config objects for constructing Geneformer-style models.

## Setup
To install, execute the following from this directory (or point the install to this directory):

```bash
pip install -e .
```

To run unit tests, execute:
```bash
pytest -v .
```

## Aquiring Data
Datasets are expected to be in the form of AnnData (.h5ad) objects such as those downloaded from [Cell x Gene | CZI](https://chanzuckerberg.github.io/cellxgene-census/). They are then pre-processed with either `bionemo-geneformer/src/bionemo/geneformer/data/singlecell/sc_memmap.py` or with sc-DL.

## See Also
[sc-DL pypi](https://www.piwheels.org/project/bionemo-scdl/)
[sc-DL github](https://github.com/NVIDIA/bionemo-fw-ea/tree/main/sub-packages/bionemo-scdl)

### References:
* Geneformer, reference foundation model for single-cell RNA: [Transfer learning enables predictions in network biology | Nature](https://www.nature.com/articles/s41586-023-06139-9)
* scGPT, alternative foundation model for single-cell RNA: [scGPT: toward building a foundation model for single-cell multi-omics using generative AI | Nature Methods](https://www.nature.com/articles/s41592-024-02201-0)
* scBERT, alternative foundation model for single-cell RNA: [scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data | Nature Machine Intelligence](https://www.nature.com/articles/s42256-022-00534-z)
* scFoundation, alternative foundation model for single-cell RNA: [Large Scale Foundation Model on Single-cell Transcriptomics | bioRxiv](https://www.biorxiv.org/content/10.1101/2023.05.29.542705v4)
Cell x Gene census, public repository for sc-RNA experiments: [CZ CELLxGENE Discover - Cellular Visualization Tool (cziscience.com)](https://cellxgene.cziscience.com/)
