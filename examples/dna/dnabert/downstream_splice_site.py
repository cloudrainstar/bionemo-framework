# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.data.preprocess.dna.preprocess import (
    GRCh38Ensembl99FastaResourcePreprocessor,
    GRCh38Ensembl99GFF3ResourcePreprocessor,
)
from bionemo.model.dna.dnabert.splice_site_prediction import (
    SpliceSiteBERTPredictionModel,
)
from bionemo.model.utils import (
    InferenceTrainerBuilder,
    setup_trainer,
)


@hydra_runner(config_path="conf", config_name="dnabert_config_splice_site")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    do_training = cfg.task.get("do_training")
    do_prediction = cfg.task.get("do_prediction")
    do_preprocess = cfg.task.get("do_preprocess")

    if do_preprocess:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = GRCh38Ensembl99GFF3ResourcePreprocessor(
            dest_directory=cfg.task.model.data.dataset_path,  # Set to $BIONEMO_HOME/data
            root_directory=cfg.task.model.data.root_directory,
            train_perc=cfg.task.model.data.train_perc,
            val_perc=cfg.task.model.data.val_perc,
            test_perc=cfg.task.model.data.test_perc,
            size=cfg.task.model.size,
        )
        _data_paths = preprocessor.prepare()

        # Needed for our actual data loaders. Used inside SpliceSiteDataModule.get_fasta_files()
        fasta_preprocessor = GRCh38Ensembl99FastaResourcePreprocessor(
            root_directory=cfg.task.model.data.root_directory,  # Set to $BIONEMO_HOME/data
            dest_directory=cfg.task.model.data.dataset_path,
        )
        fasta_preprocessor.prepare()
        # Simple assertion is making sure the files in cfg.model.data.train/test/val are the same returned.
        from pathlib import PosixPath

        assert PosixPath(cfg.task.model.data.train_file) in _data_paths
        # Validation technically not required!
        logging.info("*************** Finish Preprocessing ************")

    if do_prediction:
        assert PosixPath(cfg.task.model.data.predict_file) in _data_paths
        predictions_file = cfg.task.get("predictions_output_file")

    if do_prediction and predictions_file is None:
        raise ValueError("predictions_output_file must be specified if do_prediction=True")

    seed = cfg.task.model.seed
    np.random.seed(seed)
    pl.seed_everything(seed)

    trainer = setup_trainer(cfg.task, builder=InferenceTrainerBuilder() if not do_training else None)
    model = SpliceSiteBERTPredictionModel(cfg.task.model, trainer)

    if do_training:
        trainer.fit(model)
    if do_prediction:
        if not do_training:
            ckpt_path = cfg.task.model.get("resume_from_checkpoint")
            # NOTE when predicting in distributed, instead use a custom writer, like
            # seen here: https://pytorch-lightning.readthedocs.io/en/latest/deploy/production_basic.html
            model.data_setup()
        else:
            ckpt_path = None

        dataloader = model.predict_dataloader()
        predictions = trainer.predict(model, dataloaders=dataloader, ckpt_path=ckpt_path)
        dataset = model.predict_dataset
        predictions = reformat_predictions(predictions, dataset)
        pd.DataFrame(predictions).to_csv(predictions_file)


def reformat_predictions(predictions, dataset):
    predictions = torch.cat(predictions)
    pred_labels = torch.argmax(predictions, 1)

    # WARNING: this changes the behavior or `dataset` and is intended for use
    # only after the inference step has been completed. Set `do_transforms`
    # back to True if normal behavior is needed again.
    dataset.do_transforms = False
    predictions = [dict(**dataset[i], pred_label=pred_labels[i].item()) for i in range(len(dataset))]

    return predictions


if __name__ == "__main__":
    main()
