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


import pytorch_lightning as pl
from nemo.collections.nlp.parts.nlp_overrides import (
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf, open_dict

from bionemo.data import PhysChemPreprocess
from bionemo.model.molecule.megamolbart import FineTuneMegaMolBART
from bionemo.model.utils import (
    setup_trainer,
)

import pandas as pd
import torch
from bionemo.data.datasets.single_value_dataset import SingleValueDataset
from cuml.metrics.regression import r2_score
from sklearn.metrics import r2_score as r2_score_cpu

@hydra_runner(config_path="conf", config_name="finetune_config")
def main(cfg) -> None:
    logging.info("\n\n************* Fintune config ****************")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Do preprocessing if preprocess
    if cfg.do_preprocessing:
        logging.info("************** Starting Data PreProcessing ***********")
        PhysChemPreprocess().prepare_dataset(
            links_file=cfg.model.data.links_file, output_dir=cfg.model.data.preprocessed_data_path
        )

        if cfg.model.data.split_data:
            PhysChemPreprocess()._process_split(
                links_file=cfg.model.data.links_file,
                output_dir=cfg.model.data.preprocessed_data_path,
                test_frac=cfg.model.data.test_frac,
                val_frac=cfg.model.data.val_frac,
            )
        logging.info("************** Finished Data PreProcessing ***********")

    # Load model
    with open_dict(cfg):
        cfg.model.encoder_cfg = cfg
    seed = cfg.seed
    pl.seed_everything(seed)  # Respect seed set in cfg

    trainer = setup_trainer(cfg, builder=None, reset_accumulate_grad_batches=False)
    if cfg.restore_from_path:
        logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
        model = FineTuneMegaMolBART.restore_from(
            cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )
    else:
        model = FineTuneMegaMolBART(cfg.model, trainer)

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer.fit(model)
        logging.info("************** Finished Training ***********")
    if cfg.do_testing:
        if "test" in cfg.model.data.dataset:
            trainer.limit_train_batches = 0
            trainer.limit_val_batches = 0
            trainer.fit(model)
            ### We can use the built-in add_metrics function to add any cuml metrics.
            model.add_metrics({"r2_score": r2_score}, metrics_args={"r2_score": {"convert_dtype": False}})
            ### End add_metrics
            trainer.test(model, ckpt_path=None)
        else:
            raise UserWarning(
                "Skipping testing, test dataset file was not provided. Please specify 'dataset.test' in yaml config"
            )
        logging.info("************** Finished Testing ***********")
    ### Here we add a predict_file name under data.
    if "predict_file" in cfg.model.data:
        logging.info("************** Starting Predicting ***********")
        # Set up the dataset
        dataset = SingleValueDataset(
            datafiles=[cfg.model.data.predict_file],
            max_seq_length=cfg.model.seq_length,
            emb_batch_size=cfg.model.data.emb_batch_size,
            model=model.encoder_model,
            input_column=cfg.model.data.sequence_column,
            target_column=cfg.model.data.target_column,
            task=cfg.model.data.task_type,
            shuffle=False,
        )
        # Set up the dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.model.data.num_workers,
            pin_memory=True,
            shuffle=False,
            batch_size=cfg.model.micro_batch_size,
        )
        # Same as before
        results = []
        for batch in dataloader:
            batch["embeddings"] = batch["embeddings"].to(model.device)
            pred = model(batch).cpu().tolist()
            results += pred
        df = pd.read_csv(f"{cfg.model.data.predict_file}")
        df["pred"] = results
        df.to_csv(f"{cfg.model.data.dataset_path}/output.csv", index=False)
        # We can also do sklearn metrics here:
        r2 = r2_score_cpu(df["expt"], df["pred"])
        logging.info(f"R2 Score for Pred: {r2}")
        logging.info("************** Finished Predicting ***********")

if __name__ == "__main__":
    main()
