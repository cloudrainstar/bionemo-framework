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


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import math
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, Type, TypeVar, Union

import nemo_run as run
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.utils import logging
from pydantic import BaseModel, Field, ValidationError, field_serializer, field_validator, model_validator
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from tokenizers import Tokenizer

from bionemo.core.utils import dtypes
from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.model.finetune_token_regressor import FineTuneSeqLenBioBertConfig
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.model import BioBertGenericConfig, BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


# If you'd like to register a custom activation function, you can add it to this dictionary to pass validation and allow serialization.
CUSTOM_ACTIVATION_FNS: Dict[str, Callable[[torch.Tensor, Any], torch.Tensor]] = {}

# NOTE(SKH): DO NOT use keys that already exist in torch.nn.functional, as the torch.nn.functional functions are selected first.
for key in CUSTOM_ACTIVATION_FNS:
    assert key not in dir(torch.nn.functional), f"Key {key} already exists in torch.nn.functional"

# NOTE(SKH): it does not matter if values are duplicated as the key=>value mapping still does the right thing. Repeat values should be considered aliases.
REVERSE_CUSTOM_ACTIVATION_FNS: Dict[Callable[[torch.Tensor, Any], torch.Tensor], str] = {
    v: k for k, v in CUSTOM_ACTIVATION_FNS.items()
}


ModelConfigT = TypeVar("ModelConfigT", bound=BioBertGenericConfig)
DataModuleT = TypeVar("DataModuleT", bound=pl.LightningDataModule)


class DataConfig(BaseModel, Generic[DataModuleT]):
    """Base class for all data configurations.

    This class is used to define the interface for all data configurations. It is used to define the data module that
    will be used in the training loop.

    !! note Children **MUST** include the field `data_config_type` to discriminate between available
    data modules in the MasterConfig. Additionally, add the concrete type to the Union type annotation in MasterConfig.
    """

    micro_batch_size: int = 8
    results_dir: str = "./results"

    @abstractmethod
    def construct_data_module(self, global_batch_size: int) -> DataModuleT:
        """Construct the data module from the configuration. Cannot be defined generically."""
        ...


# TODO do we need this?
@dataclass
class GeneformerDataArtifacts:
    """Data artifacts produced by the geneformer preprocess."""

    tokenizer: Tokenizer
    median_dict: dict


class GeneformerPretrainingDataConfig(DataConfig[SingleCellDataModule]):
    """Configuration for the geneformer pre-training data module."""

    # Shadow two attributes from the parent for visibility.
    result_dir: str = "./results"
    micro_batch_size: int = 8

    data_config_type: Literal["geneformer_pretraining_data_config"] = "geneformer_pretraining_data_config"
    data_dir: str
    seq_length: int = 2048
    num_dataset_workers: int = 0

    @property
    def train_data_path(self) -> str:
        return self.data_dir + "/train"

    @property
    def val_data_path(self) -> str:
        return self.data_dir + "/val"

    @property
    def test_data_path(self) -> str:
        return self.data_dir + "/test"

    def geneformer_preprocess(self) -> GeneformerDataArtifacts:
        """Geneformer datamodule expects certain artifacts to be present in the data directory.

        This method uses a legacy 'preprocessor' from BioNeMo 1 to acquire the associated artifacts.
        """
        preprocessor = GeneformerPreprocess(
            download_directory=pathlib.Path(self.train_data_path),
            medians_file_path=pathlib.Path(self.train_data_path + "/medians.json"),
            tokenizer_vocab_path=pathlib.Path(self.train_data_path + "/geneformer.vocab"),
        )
        result = preprocessor.preprocess()
        if "tokenizer" in result and "median_dict" in result:
            logging.info("*************** Preprocessing Finished ************")
            return GeneformerDataArtifacts(tokenizer=result["tokenizer"], median_dict=result["median_dict"])
        else:
            logging.error("Preprocessing failed.")
            raise ValueError("Preprocessing failed to create tokenizer and/or median dictionary.")

    def construct_data_module(self, global_batch_size: int) -> SingleCellDataModule:
        geneformer_data_artifacts: GeneformerDataArtifacts = geneformer_preprocess(self)
        data = SingleCellDataModule(
            seq_length=self.seq_length,
            tokenizer=geneformer_data_artifacts.tokenizer,
            train_dataset_path=self.train_data_path,
            val_dataset_path=self.val_data_path,
            test_dataset_path=self.test_data_path,
            random_token_prob=0.02,
            median_dict=geneformer_data_artifacts.median_dict,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=global_batch_size,
            persistent_workers=self.num_dataset_workers > 0,
            pin_memory=False,
            num_workers=self.num_dataset_workers,
        )
        return data


def geneformer_small_data_recipe(
    data_dir="/workspaces/bionemo-fw-ea/data/cellxgene_2023-12-15_small/processed_data",
) -> GeneformerPretrainingDataConfig:
    """Recipe that produces the base geneformer small data configuration."""
    return GeneformerPretrainingDataConfig(data_dir=data_dir)


def full_geneformer_data_recipe(
    data_dir="/workspaces/bionemo-fw-ea/data/cellxgene_2023-12-15/processed_data",
) -> GeneformerPretrainingDataConfig:
    return GeneformerPretrainingDataConfig(data_dir=data_dir)


def geneformer_preprocess(data_config: GeneformerPretrainingDataConfig) -> GeneformerDataArtifacts:
    """Geneformer datamodule expects certain artifacts to be present in the data directory.

    This method uses a legacy 'preprocessor' from BioNeMo 1 to acquire the associated artifacts.
    """
    preprocessor = GeneformerPreprocess(
        download_directory=pathlib.Path(data_config.train_data_path),
        medians_file_path=pathlib.Path(data_config.train_data_path + "/medians.json"),
        tokenizer_vocab_path=pathlib.Path(data_config.train_data_path + "/geneformer.vocab"),
    )
    result = preprocessor.preprocess()
    if "tokenizer" in result and "median_dict" in result:
        logging.info("*************** Preprocessing Finished ************")
        return GeneformerDataArtifacts(tokenizer=result["tokenizer"], median_dict=result["median_dict"])
    else:
        logging.error("Preprocessing failed.")
        raise ValueError("Preprocessing failed to create tokenizer and/or median dictionary.")


class ParallelConfig(BaseModel):
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    accumulate_grad_batches: int = 1
    ddp: Literal["megatron"] = "megatron"
    remove_unused_parameters: bool = True
    num_devices: int = 1
    num_nodes: int = 1

    @model_validator(mode="after")
    def validate_devices(self):
        # I think we can do a 2x2 split on 2 gpus for pipeline/tensor model parallel
        if self.num_devices < self.tensor_model_parallel_size * self.pipeline_model_parallel_size:
            raise ValidationError(
                "devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"
            )
        return self


def simple_parallel_recipe(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1, num_devices: int = 1
) -> ParallelConfig:
    assert (
        num_devices >= tensor_model_parallel_size * pipeline_model_parallel_size
    ), "devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"
    return ParallelConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        num_devices=num_devices,
    )


class TrainingConfig(BaseModel):
    max_steps: int
    limit_val_batches: int
    val_check_interval: int
    # NOTE this matches whats used by nl.MegatronMixedPrecision which has a restricted set of precisions.
    precision: Literal["32", "bf16-mixed", "16-mixed"] = "bf16-mixed"
    accelerator: str = "gpu"


def default_trainer_config_recipe() -> TrainingConfig:
    return TrainingConfig(max_steps=55000, limit_val_batches=2, val_check_interval=100)


def setup_trainer(parallel_config: ParallelConfig, training_config: TrainingConfig) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
    )

    trainer = nl.Trainer(
        devices=parallel_config.num_devices,
        max_steps=training_config.max_steps,
        accelerator=training_config.accelerator,
        strategy=strategy,
        limit_val_batches=training_config.limit_val_batches,
        val_check_interval=training_config.val_check_interval,
        num_nodes=parallel_config.num_nodes,
        callbacks=[
            RichModelSummary(max_depth=4),
            LearningRateMonitor(),
        ],
        plugins=nl.MegatronMixedPrecision(precision=training_config.precision),
    )
    return trainer


class ExposedModelConfig(BaseModel, Generic[ModelConfigT], ABC):
    """BioNeMo model configuration class, wraps TransformerConfig and friends.

    This class is used to define the interface for all model configurations. It is **Exposed** to guard against ill-typed
    or poorly defined fields in the underlying configuration objects. `ModelConfigT` declares the associated type of the
    underlying config (most commonly a BioBertGenericConfig, but could also be a TransformerConfig or something similar).
    Children should try to expose the minimal set of fields necessary for the user to configure the model while keeping
    the more esoteric configuration private to the underlying ModelConfigT.


    !! note Children **MUST** include the field to discriminate between available
    bionemo_model_config_type: Literal["finetuning_seqlen_biobert"] = "finetuning_seqlen_biobert" # Immutable, declares how to discriminate between model types for pydantic
    data modules in the MasterConfig. Additionally, add the concrete type to the Union type annotation in MasterConfig.
    """

    # Pydantic stuff to allow arbitrary types + validators + serializers
    class Config:
        arbitrary_types_allowed = True

    """ Use this class to hide fields that are not serializable by Pydantic that we do not want to expose. """

    @abstractmethod
    def model_class(self) -> Type[ModelConfigT]: ...

    def exposed_to_internal_bionemo_model_config(self) -> ModelConfigT:
        """Converts the exposed dataclass to the underlying Transformer config.

        The underlying ModelConfigT may both be incomplete and unserializable. We use this transformation as a way to
        hide fields that are either not serializable by Pydantic or that we do not want to expose.

        This is a good candidate for refactoring.
        """

        cls: Type[ModelConfigT] = self.model_class()
        model_dict = {}
        for attr in self.model_fields:
            if attr not in model_dict and attr in cls.__dataclass_fields__:
                model_dict[attr] = getattr(self, attr)
        # Now set fp16 and bf16 based on the precision for the underlying TransformerConfig=>ParallelConfig
        #   the only constraint is that both must not be true.
        model_dict["bf16"] = self.pipeline_dtype == dtypes.precision_to_dtype["bf16-mixed"]
        model_dict["fp16"] = self.pipeline_dtype == dtypes.precision_to_dtype["16-mixed"]
        result = cls(**model_dict)

        return result

    # NOTE: See PrecisionTypes for a list of valid literals that may be deserialized.
    params_dtype: torch.dtype
    pipeline_dtype: torch.dtype
    autocast_dtype: torch.dtype

    num_layers: int = 6
    hidden_size: int = 256
    ffn_hidden_size: int = 512
    num_attention_heads: int = 4
    seq_length: int = 512
    fp32_residual_connection: bool = False
    hidden_dropout: float = 0.02
    init_method_std: float = 0.02
    kv_channels: Optional[int] = None
    apply_query_key_layer_scaling: bool = False
    make_vocab_size_divisible_by: int = 128
    masked_softmax_fusion: bool = True
    fp16_lm_cross_entropy: bool = False
    gradient_accumulation_fusion: bool = False
    layernorm_zero_centered_gamma: bool = False
    layernorm_epsilon: float = 1.0e-12
    activation_func: Callable[[torch.Tensor, Any], torch.Tensor] = F.gelu
    qk_layernorm: bool = False
    apply_residual_connection_post_layernorm: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    get_attention_mask_from_fusion: bool = False
    attention_dropout: float = 0.1
    share_embeddings_and_output_weights: bool = True
    enable_autocast: bool = False
    nemo1_ckpt_path: Optional[str] = None
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_local_spec

    @field_serializer("params_dtype", "pipeline_dtype", "autocast_dtype")
    def serialize_dtypes(self, v: torch.dtype) -> PrecisionTypes:
        return dtypes.dtype_to_precision[v]

    @field_validator("activation_func", mode="before")
    @classmethod
    def validate_activation_func(cls, activation_func: str) -> Callable:
        """
        Validates the activation function, assumes this function exists in torch.nn.functional. For custom
        activation functions, use the CUSTOM_ACTIVATION_FUNCTIONS dictionary in the module.

        This method validates the provided activation function string and returns
        a callable function based on the validation context using the provided validator in the base class.
        Args:
            activation_func (str): The activation function to be validated.
            context (ValidationInfo): The context for validation.
        Returns:
            Callable: A callable function after validation.

        See Also:
            CUSTOM_ACTIVATION_FNS
        """
        func = getattr(torch.nn.functional, activation_func.lower(), None)
        if func is None and activation_func in CUSTOM_ACTIVATION_FNS:
            func = CUSTOM_ACTIVATION_FNS[activation_func]
            return func
        elif func is None:
            raise ValidationError(
                f"activation_func must be a valid function in `torch.nn.functional`, got {activation_func=}"
            )
        else:
            return func

    @field_validator("params_dtype", "pipeline_dtype", "autocast_dtype", mode="before")
    @classmethod
    def precision_validator(cls, v: PrecisionTypes) -> torch.dtype:
        return dtypes.get_autocast_dtype(v)

    @field_serializer("activation_func")
    def serialize_activation_func(self, v: Callable[[torch.Tensor, Any], torch.Tensor]) -> str:
        func_name = v.__name__
        func = getattr(torch.nn.functional, func_name, None)
        if func is not None:
            return func_name
        elif func in REVERSE_CUSTOM_ACTIVATION_FNS:
            return REVERSE_CUSTOM_ACTIVATION_FNS[func]  # Get the serialization key
        else:
            raise ValueError(f"Unsupported activation function: {v}")


class ExposedGeneformerConfig(ExposedModelConfig[GeneformerConfig]):
    """There are no additional arguments for Geneformer, so we simply plugin the associated types and move on."""

    bionemo_model_config_type: Literal["geneformer"] = (
        "geneformer"  # Immutable, declares how to discriminate between model types for pydantic
    )

    def model_class(self) -> Type[GeneformerConfig]:
        return GeneformerConfig


class ExposedFineTuneSeqLenBioBertConfig(ExposedModelConfig[FineTuneSeqLenBioBertConfig]):
    """Config for models that fine-tune a BioBERT model from a pre-trained checkpoint.

    Parameters:
        initial_ckpt_path - path to a directory containing checkpoint files for initializing the model. This is only
            required on the first execution of the model, any restored checkpoints should skip this step.
        initial_ckpt_skip_keys_with_these_prefixes - skip any layer that contains this key during restoration. Useful
            for ignoring extra additional layers used for finetuning. Layers with these keys are then randomly initialized.
    """

    # Used by discriminators
    bionemo_model_config_type: Literal["finetuning_seqlen_biobert"] = (
        "finetuning_seqlen_biobert"  # Immutable, declares how to discriminate between model types for pydantic
    )

    # Custom parameters for FineTuning
    initial_ckpt_path: Optional[str] = None
    initial_ckpt_skip_keys_with_these_prefixes: Optional[List[str]] = None

    def __post_init__(self):
        if not self.initial_ckpt_skip_keys_with_these_prefixes:
            self.initial_ckpt_skip_keys_with_these_prefixes = ["regression_head"]

    def model_class(self) -> Type[FineTuneSeqLenBioBertConfig]:
        return FineTuneSeqLenBioBertConfig


def geneformer_finetuning_regression_head_recipe(
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    initial_ckpt_skip_keys_with_these_prefixes: Optional[List[str]] = None,
) -> ExposedFineTuneSeqLenBioBertConfig:
    # NOTE (SKH): this recipe is sad because it isnt smart enough to know our validator is returning a dtype.
    finetuning_config = ExposedFineTuneSeqLenBioBertConfig(
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
        initial_ckpt_skip_keys_with_these_prefixes=initial_ckpt_skip_keys_with_these_prefixes,
    )
    return finetuning_config


def geneformer10M_pretraining_recipe(
    seq_length: int = 128,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_local_spec,
) -> ExposedGeneformerConfig:
    geneformer_config = ExposedGeneformerConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func="gelu",
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=False,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
    )
    return geneformer_config


class OptimizerSchedulerConfig(BaseModel):
    # TODO could use validators on optimizer, interval, and monitor.

    lr: float = 1e-4
    optimizer: str = "adam"
    cosine_rampup_frac: float = 0.01
    cosine_hold_frac: float = 0.05
    interval: str = "step"
    monitor: str = "val_loss"


def default_adam_optimizer_with_cosine_annealing_recipe() -> OptimizerSchedulerConfig:
    return OptimizerSchedulerConfig()


def biobert_lightning_module(
    bionemo_model_config: BioBertGenericConfig,
    tokenizer: Tokenizer,
    optim_config: OptimizerSchedulerConfig,
    num_steps: int,
) -> BioBertLightningModule:
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


class ExperimentConfig(BaseModel):
    save_every_n_steps: int
    result_dir: str
    experiment_name: str
    restore_from_checkpoint_path: Optional[str]
    resume_if_exists: bool
    wandb_config: Optional[WandbConfig] = None
    save_last_checkpoint: bool = True
    metric_to_monitor_for_checkpoints: str = "reduced_train_loss"
    save_top_k: int = 2
    create_tensorboard_logger: bool = False


def experiment_config_recipe() -> ExperimentConfig:
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir="./results",
        experiment_name="default_experiment",
        restore_from_checkpoint_path=None,
        resume_if_exists=True,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="reduced_train_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )


def nemo_logger_factory(experiment_config: ExperimentConfig, wandb_config: Optional[WandbConfig]) -> nl.NeMoLogger:
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=experiment_config.save_last_checkpoint,
        monitor=experiment_config.metric_to_monitor_for_checkpoints,
        save_top_k=experiment_config.save_top_k,
        every_n_train_steps=experiment_config.save_every_n_steps,
        always_save_context=True,
    )

    wandb_config: Optional[WandbConfig] = (
        None
        if wandb_config is None
        else WandbConfig(
            offline=wandb_config.offline,
            project=wandb_config.project,
            entity=wandb_config.entity,
            log_model=False,
        )
    )

    nemo_logger = setup_nemo_lightning_logger(
        root_dir=experiment_config.result_dir,
        name=experiment_config.experiment_name,
        initialize_tensorboard_logger=experiment_config.create_tensorboard_logger,
        wandb_config=wandb_config,
        ckpt_callback=checkpoint_callback,
    )
    return nemo_logger


@run.cli.entrypoint
def pretrain(
    bionemo_exposed_model_config: ExposedModelConfig,
    data_config: DataConfig[DataModuleT],
    parallel_config: ParallelConfig,
    training_config: TrainingConfig,
    optim_config: OptimizerSchedulerConfig,
    experiment_config: ExperimentConfig,
    wandb_config: Optional[WandbConfig],
    resume_if_exists: bool = True,
):
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

    data: SingleCellDataModule = data_config.construct_data_module(global_batch_size)

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


class MasterConfig(BaseModel):
    """Mulling ways to make this generic over data modules:

    1) ABC in our DataModule that supports DataConfig -> DataModule
        pros:
        cons:
    2) Discriminated union on data_config, additionally needs a method that also takes this union and produces the correct data module.
    3) Pick one and highlight the other approach in either the SDD, PR, or both.

    """

    data_config: Union[GeneformerPretrainingDataConfig] = Field(..., discriminator="data_config_type")
    parallel_config: ParallelConfig
    training_config: TrainingConfig
    # TODO expand this for all other relevant models here.
    bionemo_model_config: Union[ExposedGeneformerConfig, ExposedFineTuneSeqLenBioBertConfig] = Field(
        ..., discriminator="bionemo_model_config_type"
    )
    optim_config: OptimizerSchedulerConfig
    experiment_config: ExperimentConfig
    wandb_config: Optional[WandbConfig] = None

    @model_validator(mode="after")
    def validate_master_config(self) -> "MasterConfig":
        self.bionemo_model_config.seq_length = self.data_config.seq_length
        # What other global validators should we set here?
        return self


def recipes_to_config_json(model_cfg_type="geneformer"):
    """Simple example for creating a JSON from recipes."""

    data_config: GeneformerPretrainingDataConfig = geneformer_small_data_recipe()
    parallel_config = simple_parallel_recipe()
    training_config = default_trainer_config_recipe()
    if model_cfg_type == "geneformer":
        bionemo_model_config = geneformer10M_pretraining_recipe()
    else:
        bionemo_model_config = geneformer_finetuning_regression_head_recipe()

    optim_config = default_adam_optimizer_with_cosine_annealing_recipe()
    experiment_config = experiment_config_recipe()
    wandb_config = WandbConfig(project="bionemo2-demo", entity="nvidia", offline=True)

    # Create the master config
    master_config = MasterConfig(
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=bionemo_model_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        wandb_config=wandb_config,
    )

    # Serialize to JSON
    json_str = master_config.model_dump_json(indent=2)

    # Save to file
    with open(
        "/workspaces/bionemo-fw-ea/sub-packages/bionemo-geneformer/src/bionemo/geneformer/conf/default-geneformer-config.json",
        "w",
    ) as f:
        f.write(json_str)

    print("Configuration saved to config.json")


if __name__ == "__main__":
    recipes_to_config_json("geneformer")
    # recipes_to_config_json('finetune')

    def parse_args():
        parser = argparse.ArgumentParser(description="Run Geneformer pretraining")
        parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file")
        return parser.parse_args()

    def load_config(config_path: str) -> MasterConfig:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return MasterConfig(**config_dict)

    args = parse_args()
    config = load_config(args.config)

    pretrain(
        bionemo_exposed_model_config=config.bionemo_model_config,
        data_config=config.data_config,
        parallel_config=config.parallel_config,
        training_config=config.training_config,
        optim_config=config.optim_config,
        experiment_config=config.experiment_config,
        wandb_config=config.wandb_config,
        resume_if_exists=False,
    )
