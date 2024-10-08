# Getting Started

## Repository structure

### High level overview
This repository is structured as a meta-package that collects together many python packages. We designed in this way
because this is how we expect our users to use bionemo, as a package that they themselves import and use in their
own projects. By structuring code like this ourselves we ensure that bionemo developers follow similar patterns to our
end users.

Each model is stored in its own `sub-packages`. Some examples of models include:

* `bionemo-esm2`: the ESM2 model.
* `bionemo-geneformer`: the Geneformer model.
* `bionemo-example_model`: A minimal example MNIST model that demonstrates how you can write a lightweight
    megatron model that doesn't actually support any megatron parallelism, but should run fine as long as you only use
    data parallelism to train.

There are also useful utility packages, for example:
* `bionemo-scdl`: Single Cell Dataloader (SCDL) provides a dataset implementation that can be used by downstream
    single-cell models in the bionemo package.
* `bionemo-webdatamodule`: a reusable PyTorch Lightning Datamodule for WebDataset formatted data.
* `bionemo-size-aware-batching`: provides a simple way to create mini-batches in a memory consumption-aware  manner.
* `bionemo-testing`: a suite of utilities that are useful in testing, think `torch.testing` or `np.testing`.

Finally, some of the packages represent common functions and abstract base classes that expose APIs that are useful for
interacting with specific libraries, such as `Megatron-LM` and `NeMo2`, PyTorch, and PyTorch Geometric.

These "BioNeMo component libraries" include:
* `bionemo-core`: common general, reusable code and high level APIs, core dependency on PyTorch
     (_all bionemo sub-packages depend on bionemo-core_).
* `bionemo-llm`: ABCs for code that multiple large language models (eg BERT variants) share, depends on NeMo,
     Megatron-LM, and PyTorch Lightning.
* `bionemo-geometric`: common code for Graphical Neural Network (GNN) models, dependency on PyTorch Geometric.

Documentation source is stored in `docs/`.

Code used for the development process lives under `internal/`. This code is not intended for BioNeMo users.
Unlike the code in `sub-packages/`, nothing under `internal/` is packaged and distributed.

For instructions on building docker images and testing code from source, please refer to the setup section of the
[README](../../../README.md) file. Or, for more in-depth instructions and explanation, check out the
[initialization guide](initialization-guide.md).

## Installation
### Initializing 3rd-party dependencies as git submodules

For development, the NeMo and Megatron-LM dependencies are vendored in the bionemo-2 repository workspace as git
submodules. The pinned commits for these submodules represent the "last-known-good" versions of these packages that are
confirmed to be working with bionemo2 (and those that are tested in CI).

To initialize these submodules when cloning the repo, add the `--recursive` flag to the git clone command:
```bash
git clone --recursive git@github.com:NVIDIA/bionemo-fw-ea.git
```

To download the pinned versions of these submodules within an existing git repository, run:
```bash
git submodule update --init --recursive
```
