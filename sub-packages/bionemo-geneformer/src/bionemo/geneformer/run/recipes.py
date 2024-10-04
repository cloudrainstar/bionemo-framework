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


import argparse
from typing import List, Optional

from nemo.utils import logging

from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.geneformer.run.config_models import ExposedFineTuneSeqLenBioBertConfig, ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig
from bionemo.llm.config.config_models import (
    ExperimentConfig,
    MainConfig,
    OptimizerSchedulerConfig,
    ParallelConfig,
    TrainingConfig,
)
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils.logger_utils import WandbConfig


def geneformer_small_data_recipe(data_dir) -> GeneformerPretrainingDataConfig:
    """Recipe that produces the base geneformer small data configuration."""
    return GeneformerPretrainingDataConfig(data_dir=data_dir)


def full_geneformer_data_recipe(data_dir) -> GeneformerPretrainingDataConfig:
    return GeneformerPretrainingDataConfig(data_dir=data_dir)


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


def default_trainer_config_recipe() -> TrainingConfig:
    return TrainingConfig(max_steps=55000, limit_val_batches=2, val_check_interval=100)


def geneformer10M_pretraining_recipe(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_local_spec,
) -> ExposedGeneformerPretrainConfig:
    geneformer_config = ExposedGeneformerPretrainConfig(
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


def default_adam_optimizer_with_cosine_annealing_recipe() -> OptimizerSchedulerConfig:
    return OptimizerSchedulerConfig()


def experiment_config_recipe() -> ExperimentConfig:
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir="./results",
        experiment_name="default_experiment",
        restore_from_checkpoint_path=None,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="reduced_train_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )


def main():
    def parse_args():
        parser = argparse.ArgumentParser(description="Create Geneformer configuration JSON.")
        parser.add_argument(
            "--dest",
            type=str,
            default="./geneformer-recipe.json",
            required=True,
            help="Path to the JSON configuration file.",
        )
        parser.add_argument(
            "--data-dir", type=str, required=True, help="Path to the directory containing pretraining data."
        )
        args = parser.parse_args()
        return args

    """Simple example for creating a JSON from recipes."""

    args = parse_args()
    data_config: GeneformerPretrainingDataConfig = geneformer_small_data_recipe(data_dir=args.data_dir)
    parallel_config = simple_parallel_recipe()
    training_config = default_trainer_config_recipe()
    # bionemo_model_config = geneformer_finetuning_regression_head_recipe()
    bionemo_model_config = geneformer10M_pretraining_recipe()
    optim_config = default_adam_optimizer_with_cosine_annealing_recipe()
    experiment_config = experiment_config_recipe()
    wandb_config = WandbConfig(
        project="bionemo2-demo",
        entity="nvidia",
        offline=True,
        tags=[],
        group="dev",
        id="dev",
        log_model=False,
        anonymous=True,
    )

    # Create the master config
    master_config = MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig](
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
        args.dest,
        "w",
    ) as f:
        f.write(json_str)
    logging.info(f"Saved configuration to {args.dest=}")
