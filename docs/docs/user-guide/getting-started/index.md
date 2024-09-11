# Getting Started

## Initializing 3rd-party dependencies as git submodules

For development, the NeMo and Megatron-LM dependencies are vendored in the bionemo-2 repository workspace as git
submodules. The pinned commits for these submodules represent the "last-known-good" versions of these packages that are
confirmed to be working with bionemo2 (and those that are tested in CI).

To initialize these sub-modules when cloning the repo, add the `--recursive` flag to the git clone command:

```bash
git clone --recursive git@github.com:NVIDIA/bionemo-fw-ea.git
```

To download the pinned versions of these submodules within an existing git repository, run

```bash
git submodule update --init --recursive
```

begin old repo here:

# What is BioNeMo?

Generative AI and large language models (LLMs) are achieving incredible breakthroughs in chemistry and biology, such as enabling 3D protein structure prediction, property prediction, and even the generation of novel protein sequences and molecules. This progress has facilitated developments in the pharmaceutical industry, such as antibody design, small-molecule drug design, and newer approaches like RNA aptamer and peptide-based therapeutics. As each of these pieces comes into play, their respective models may need additional fine-tuning or optimization to thoroughly explore or understand the biomolecular space, leading to the need for centralized infrastructure for model development and deployment.

**BioNeMo Framework** is a free to use collection of programming tools and packages offering access to optimized, pre-trained biomolecular models and workflows, along with versatile functionalities for building and customizing models, including training and fine-tuning. Capabilities span various workloads and therapeutic modalities, such as molecular generation and representation learning, protein structure prediction and representation learning, protein-ligand and protein-protein docking, and DNA/RNA/single-cell embedding.

**BioNeMo NIMs** are easy-to-use enterprise-ready inference microservices with built-in API endpoints. NIMs are engineered for scalable, self-hosted or cloud-hosted deployment of optimized, production-grade biomolecular foundation models on any cloud or data center. Check out the growing list of BioNeMo NIMs [here](https://build.nvidia.com/explore/biology).

![](../../assets/old_images/bionemo_overview_2.png)

## BioNeMo Framework: Fundamentals

BioNeMo Framework provides versatile functionalities for developing and training large-scale biology-based models. BioNeMo allows users to build and train biomolecular models by providing access to pre-trained models and common model components for accelerating drug discovery workflows. Built for supercomputing scale, the framework allows developers to easily configure and train distributed multi-node jobs with minimal code.

![](../../assets/old_images/bionemo_overview_1.png)

BioNeMo is built on [NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/intro.html), a scalable and cloud-native generative AI framework for researchers to create, customize, and deploy large language models (LLMs). NeMo provides a robust environment for working with large learning models, including [NVIDIA Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/megatron.html) models. The BioNeMo Framework provides enhancements to PyTorch Lighting, such as hyperparameter configurability with YAML files and checkpoint management. Users can conveniently and quickly train models using these features, test them for desired tasks, and integrate them alongside existing applications.

Some of the key features of BioNeMo Framework are:

- Development and training of large transformer models using NVIDIA's Megatron framework.
- Easy to configure multi-GPU, multi-node training with data parallelism, model parallelism, and mixed precision.
- Model training recipes that can be readily implemented on DGX compute infrastructure.
- Logging with Tensorboard and Weights and Biases to monitor the model training process.

## BioNeMo Framework: Available Models

State-of-the-art models are being continually integrated into the BioNeMo Framework. The available catalog consists of: 1) Models developed by NVIDIA, 2) Models contributed by NVIDIA’s ecosystem partners, and 3) Community models further enhanced by NVIDIA to take advantage of GPU acceleration. The BioNeMo Framework currently offers the following pre-trained models:

| **Model**                                         | **Modality**       | **Uses**                                      |
| ------------------------------------------------- | ------------------ | --------------------------------------------- |
| [MegaMolBART](./models/megamolbart.md)            | Small Molecule     | Representation Learning + Molecule Generation |
| [MolMIM](./models/molmim.md)                      | Small Molecule     | Representation Learning + Molecule Generation |
| [ESM-1nv](./models/esm1-nv.md)                    | Protein            | Representation Learning                       |
| [ESM-2nv 650M](./models/esm2-nv.md)               | Protein            | Representation Learning                       |
| [ESM-2nv 3B](./models/esm2-nv.md)                 | Protein            | Representation Learning                       |
| [EquiDock DIPS Model](./models/equidock.md)       | Protein            | Protein-Protein Complex Formation             |
| [EquiDock DB5 Model](./models/equidock.md)        | Protein            | Protein-Protein Complex Formation             |
| [OpenFold](./models/openfold.md)                  | Protein            | Protein Structure Prediction                  |
| [ProtT5nv](./models/prott5nv.md)                  | Protein            | Representation Learning                       |
| [DiffDock Confidence Model](./models/diffdock.md) | Protein + Molecule | Generation of Ligand Poses                    |
| [DiffDock Score Model](./models/diffdock.md)      | Protein + Molecule | Generation of Ligand Poses                    |
| [DNABERT](./models/dnabert.md)                    | DNA                | Representation Learning                       |
| [Geneformer](./models/geneformer.md)              | Single Cell        | Representation Learning                       |

For more information about the models included in BioNeMo Framework, you may refer to the Model Cards linked in the table above or the original publications referenced in the respective model descriptions.
