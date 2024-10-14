# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import tempfile
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from nemo import lightning as nl
from torch import Tensor

from bionemo.esm2.api import ESM2GenericConfig
from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.finetune_regressor import ESM2FineTuneSeqConfig, InMemorySingleValueDataset
from bionemo.llm.model.biobert.lightning import biobert_lightning_module


__all__: Sequence[str] = ("infer_model",)


class BatchPredictionWriter(BasePredictionWriter, pl.Callback):
    """A callback that writes predictions to disk at specified intervals during training.

    Args:
        output_dir (str): The directory where predictions will be written.
        write_interval (str): The interval at which predictions will be written. (batch, epoch)
    """

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = str(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        result_path = os.path.join(self.output_dir, f"predictions__rank_{trainer.global_rank}__batch_{batch_idx}.pt")

        if batch:  # to skip empty batches
            torch.save(
                {
                    "prediction": prediction,
                    "batch_indices": batch[
                        "batch_indices"
                    ],  # save `batch_indices` to get the information about the data index
                },
                result_path,
            )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        raise NotImplementedError("write_on_epoch_end is not supported by BatchPredictionWriter")


def infer_model(
    config: ESM2GenericConfig,
    data_module: pl.LightningDataModule,
    tokenizer: BioNeMoESMTokenizer = get_tokenizer(),
) -> list[Tensor]:
    """Infers a BioNeMo ESM2 model using PyTorch Lightning.

    Parameters:
        config: The configuration for the ESM2 model.
        data_module: The data module for training and validation.
        tokenizer: The tokenizer to use. Defaults to `get_tokenizer()`.

    Returns:
        A list of tensors containing the predictions of predict_dataset in datamodule
    """
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, ddp="megatron", find_unused_parameters=True
    )

    tempdir = tempfile.mkdtemp()
    pred_writer = BatchPredictionWriter(tempdir, write_interval="batch")
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        num_nodes=1,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[pred_writer],
    )
    module = biobert_lightning_module(config=config, tokenizer=tokenizer)
    results = trainer.predict(module, datamodule=data_module)

    return results, tempdir


if __name__ == "__main__":
    # create a List[Tuple] with (sequence, target) values
    artificial_sequence_data = [
        "TLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
        "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "GRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKN",
        "DELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "KLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
        "LFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
        "LGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "ISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
        "SGSKASSDSQDANQCCTSCEDNAPATSYCVECSEPLCETCVEAHQRVKYTKDHTVRSTGPAKT",
    ]
    data = [(seq, len(seq) / 100.0) for seq in artificial_sequence_data]

    dataset = InMemorySingleValueDataset(data)

    # NOTE: Due to the current limitation in inference of NeMo lightning module, partial batches with
    # size < global_batch_size are not being processed with predict_step(). Therefore we set the global to len(data)
    # and choose the micro_batch_size so that global batch size is divisible by micro batch size x data parallel size
    data_module = ESM2FineTuneDataModule(predict_dataset=dataset, global_batch_size=4, micro_batch_size=2)

    # To download a pre-trained ESM2 model that works with this inference script, run the following command...
    # $ download_bionemo_data esm2/650m:2.0 --source ngc
    # ... and pass the output path (e.g. `.../.cache/bionemo/975d29ee980fcb08c97401bbdfdcf8ce-esm2_650M_nemo2.tar.gz.untar`)
    # as an argument into `initial_ckpt_path` below!
    config = ESM2FineTuneSeqConfig(
        # initial_ckpt_path = finetuned_checkpoint,  # supply the finetuned checkpoint path
        # initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=list)   # reset to avoid skipping the head params
    )

    results, tempdir = infer_model(config, data_module)
    print(results)

    # Manually delete the directory when done
    os.rmdir(str(tempdir))
