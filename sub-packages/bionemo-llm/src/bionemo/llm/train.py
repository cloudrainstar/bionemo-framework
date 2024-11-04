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


import math
import pathlib
from typing import Optional

from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.utils import logging
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from tokenizers import Tokenizer

from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.run.config_models import (
    DataConfig,
    DataModuleT,
    ExperimentConfig,
    ExposedModelConfig,
    OptimizerSchedulerConfig,
    ParallelConfig,
    TrainingConfig,
)
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


def nemo_logger_factory(experiment_config: ExperimentConfig, wandb_config: Optional[WandbConfig]) -> nl.NeMoLogger:
    """Creates and returns a NeMoLogger instance configured based on the provided experiment and wandb configurations.

    Args:
        experiment_config (ExperimentConfig): Configuration object containing experiment settings such as
            result directory, experiment name, checkpoint settings, and logger preferences.
        wandb_config (Optional[WandbConfig]): Optional configuration object for Weights and Biases logging.

    Returns:
        nl.NeMoLogger: An instance of NeMoLogger configured with the specified settings.
    """
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=experiment_config.save_last_checkpoint,
        monitor=experiment_config.metric_to_monitor_for_checkpoints,
        save_top_k=experiment_config.save_top_k,
        every_n_train_steps=experiment_config.save_every_n_steps,
        always_save_context=True,
    )

    nemo_logger = setup_nemo_lightning_logger(
        root_dir=experiment_config.result_dir,
        name=experiment_config.experiment_name,
        initialize_tensorboard_logger=experiment_config.create_tensorboard_logger,
        wandb_config=wandb_config,
        ckpt_callback=checkpoint_callback,
    )
    return nemo_logger


def setup_trainer(parallel_config: ParallelConfig, training_config: TrainingConfig, callbacks=None) -> nl.Trainer:
    """Set up the trainer for model training using the specified parallel and training configurations.

    Args:
        parallel_config (ParallelConfig): Configuration for parallelism, including tensor and pipeline model parallel sizes,
                                          number of devices, and number of nodes.
        training_config (TrainingConfig): Configuration for training, including maximum steps, accelerator type,
                                          validation batch limit, validation check interval, and precision.
        callbacks (list, optional): List of callback functions to be used during training. Defaults to None,
                                    in which case default callbacks (RichModelSummary and LearningRateMonitor) are used.

    Returns:
        nl.Trainer: Configured trainer object ready for model training.
    """
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
    )
    if callbacks is None:
        callbacks = [
            RichModelSummary(max_depth=4),
            LearningRateMonitor(),
        ]

    trainer = nl.Trainer(
        devices=parallel_config.num_devices,
        max_steps=training_config.max_steps,
        accelerator=training_config.accelerator,
        strategy=strategy,
        limit_val_batches=training_config.limit_val_batches,
        val_check_interval=training_config.val_check_interval,
        num_nodes=parallel_config.num_nodes,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(precision=training_config.precision),
    )
    return trainer


def biobert_lightning_module(
    bionemo_model_config: BioBertConfig,
    tokenizer: Tokenizer,
    optim_config: OptimizerSchedulerConfig,
    num_steps: int,
) -> BioBertLightningModule:
    """Creates a BioBertLightningModule with the specified configuration, tokenizer, and optimizer settings.

    Args:
        bionemo_model_config (BioBertConfig): Configuration for the BioBert model.
        tokenizer (Tokenizer): Tokenizer to be used with the model.
        optim_config (OptimizerSchedulerConfig): Configuration for the optimizer and learning rate scheduler.
        num_steps (int): Total number of training steps.

    Returns:
        BioBertLightningModule: An instance of BioBertLightningModule configured with the provided settings.
    """
    model = BioBertLightningModule(
        bionemo_model_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=optim_config.lr,
                optimizer=optim_config.optimizer,
                use_distributed_optimizer=True,
                fp16=bionemo_model_config.fp16,
                bf16=bionemo_model_config.bf16,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                min_lr=optim_config.lr / 100,
                warmup_steps=int(math.ceil(num_steps * optim_config.cosine_rampup_frac)),
                interval=optim_config.interval,
                monitor=optim_config.monitor,
                constant_steps=int(math.ceil(num_steps * optim_config.cosine_hold_frac)),
            ),
        ),
    )
    return model


def train(
    bionemo_exposed_model_config: ExposedModelConfig,
    data_config: DataConfig[DataModuleT],
    parallel_config: ParallelConfig,
    training_config: TrainingConfig,
    optim_config: OptimizerSchedulerConfig,
    experiment_config: ExperimentConfig,
    wandb_config: Optional[WandbConfig],
    resume_if_exists: bool = True,
):
    """Train a BioNemo model using the provided configurations. Uses the ExposedModelConfig and DataConfig as the primary variants for this method.

    Args:
        bionemo_exposed_model_config (ExposedModelConfig): Configuration for the exposed BioNemo model.
        data_config (DataConfig[DataModuleT]): Configuration for the data module.
        parallel_config (ParallelConfig): Configuration for parallel training.
        training_config (TrainingConfig): Configuration for training parameters.
        optim_config (OptimizerSchedulerConfig): Configuration for the optimizer and scheduler.
        experiment_config (ExperimentConfig): Configuration for the experiment.
        wandb_config (Optional[WandbConfig]): Configuration for Weights and Biases logging.
        resume_if_exists (bool, optional): Flag to resume training if a checkpoint exists. Defaults to True.
    """
    bionemo_model_config = bionemo_exposed_model_config.exposed_to_internal_bionemo_model_config()
    pathlib.Path(data_config.result_dir).mkdir(parents=True, exist_ok=True)

    if experiment_config.save_every_n_steps != training_config.val_check_interval:
        logging.warning("Mutating training_config.save_every_n_steps to be equal to val_check_interval.")
        experiment_config.save_every_n_steps = training_config.val_check_interval

    global_batch_size = infer_global_batch_size(
        micro_batch_size=data_config.micro_batch_size,
        num_nodes=parallel_config.num_nodes,
        devices=parallel_config.num_devices,
        accumulate_grad_batches=parallel_config.accumulate_grad_batches,
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
    )

    data: DataModuleT = data_config.construct_data_module(global_batch_size)

    # TODO BioBertDataModule or BioBertTokenizer abstractions. We know all DataModuleT in this case has data.tokenizer,
    # although this constraint is not documented.
    model: BioBertLightningModule = biobert_lightning_module(
        bionemo_model_config, tokenizer=data.tokenizer, optim_config=optim_config, num_steps=training_config.max_steps
    )
    trainer: nl.Trainer = setup_trainer(parallel_config, training_config)
    nemo_logger: nl.NeMoLogger = nemo_logger_factory(experiment_config, wandb_config=wandb_config)

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=resume_if_exists,
            resume_ignore_no_checkpoint=True,
        ),
    )
