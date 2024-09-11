# BioNeMo Framework: Available Models

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
