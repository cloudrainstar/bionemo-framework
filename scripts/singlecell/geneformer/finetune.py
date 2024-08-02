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
import math
import os
from pathlib import Path
from typing import Optional, Type, get_args

import torch
from megatron.core import parallel_state
from megatron.core.models.bert.bert_lm_head import BertLMHead
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import ModelParallelConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import get_vocab_size
from nemo.lightning.megatron_parallel import MegatronLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.lightning.resume import AutoResume
from nemo.utils import logging
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from torch import Tensor
from torch.nn import functional as F

from bionemo.contrib.data.singlecell.datamodule import SingleCellDataModule
from bionemo.contrib.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.contrib.lightning import LossLoggingCallback
from bionemo.contrib.model.biobert.lightning import BioBertLightningModule
from bionemo.contrib.model.biobert.model import BioBertConfig, BiobertSpecOption, MegatronBioBertModel
from bionemo.contrib.model.biobert.transformer_specs import get_biobert_spec
from bionemo.contrib.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.contrib.utils.logger_utils import WandbLoggerOptions, setup_nemo_lightning_logger
from megatron.core.transformer.utils import get_linear_layer
import torch.nn.functional as F
from torch import nn

try:
    import apex

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm

class Linear(MegatronModule):

    def __init__(self, input_size, config: "BioBertFinetuneHeadConfig"):
        super().__init__(config)
        self.dense_layers = nn.ModuleList([
                get_linear_layer(input_size, input_size, config.init_method) for n in range(config.n_layers - 1)
            ])
        self.dense_layers.append(get_linear_layer(input_size, config.output_size, config.init_method))
        self.activation = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = LNImpl(
            config=config,
            hidden_size=input_size,
            eps=config.layernorm_epsilon,
        )

    def forward(self, X):
        return self.dense(X)

class MLPHeadModel(MegatronModule):
    def __init__(self, language_model: MegatronBioBertModel, config: "BioBertFinetuneHeadConfig", input_size):
        super().__init__(config)
        self.language_model = language_model
        # TODO(@georgea) how do I make sure this gets cast properly? when in BertLMHead, it was cast fine
        # self.head = Linear(config, input_size)
        # self.head = get_linear_layer(input_size, config.output_size, config.init_method) # : MegatronModule = BertLMHead(1024, config=config)
        self.dense_layers = nn.ModuleList([
                get_linear_layer(input_size, input_size, config.init_method) for n in range(config.n_layers - 1)
            ])
        self.dense_layers.append(get_linear_layer(input_size, config.output_size, config.init_method))
        self.activation = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = LNImpl(
            config=config,
            hidden_size=input_size,
            eps=config.layernorm_epsilon,
        )
        self.language_model.return_embeddings = True
        # @georgea megatron core pipelining currently depends on model type, so this has to be set
        # it is unlear to me whether this can take any value other than the one inherited from langauge_model
        self.model_type = self.language_model.model_type

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tokentype_ids: Tensor | None = None,
        lm_labels: Tensor | None = None,
        inference_params=None,
    ):
        x = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokentype_ids=tokentype_ids,
            lm_labels=lm_labels,
            inference_params=inference_params,
        )
        for i in range(len(self.dense_layers)):
            x = self.dropout(x)
            layer = self.dense_layers[i]
            x = self.layer_norm(x)
            x = layer(x)
            if i < len(self.dense_layers) - 1:
                x = self.activation(x)

        return x

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, "input_tensor should only be length 1 for gpt/bert"
        self.language_model.encoder.set_input_tensor(input_tensor[0])


from megatron.core.transformer.transformer_config import TransformerConfig

from dataclasses import dataclass
from typing import Callable
from megatron.core.utils import init_method_normal, scaled_init_method_normal

from typing import Sequence, Tuple

from bionemo.contrib.lightning import DataT, ReductionT

class ReductionLoss(MegatronLossReduction):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, batch: DataT, forward_out: torch.Tensor) -> Tuple[torch.Tensor, ReductionT]:
        """
        Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.
        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
            backpropagation and the ReductionT will be passed to the reduce method
            (which currently only works for logging.).
        """
        target = torch.zeros(size=(len(batch['text']), ), dtype=torch.long, device=forward_out.device)
        first_entries = batch['text'][:, 0]
        target[:len(first_entries)] = first_entries % forward_out.size(1)

        loss = F.cross_entropy(forward_out, target)

        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> torch.Tensor:
        """
        Works across micro-batches. (data on single gpu).
        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()

@dataclass
class BioBertFinetuneHeadConfig(ModelParallelConfig):
    # don't want to inherit from `TransformerConfig`, becuase it makes you set the transformer specs (like head) and we
    # are not doing anything different there from whatever is loaded by the fabric

    # Expected for the BERTLMHead
    output_size: int = 1
    n_layers: int = 3
    init_method: Callable | None = None
    init_method_std: float = 0.02
    trainer: nl.Trainer = None
    # params for layernorm module
    layernorm_epsilon: float = 1e-5
    layernorm_zero_centered_gamma: bool = False
    normalization: bool = "LayerNorm"
    persist_layer_norm: bool = False
    memory_efficient_layer_norm: bool = False

    # def __init__(self, trainer: nl.Trainer, *args, **kwargs):
        # super().__init__(*args, **kwargs)
    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        super().__post_init__()
        # Expected for the BERTLMHead
        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)
        # Needed for compatibility with megatron for some reason, I think it contributes to ddp configuration
        # see: "3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py", line 146
        self.calculate_per_token_loss = False

    def configure_model(self, tokenizer) -> "MegatronBioBertModel":
        fabric = self.trainer.to_fabric()
        distributed_model = fabric.load_model(
            "/workspaces/bionemo-github/results/test_experiment/2024-07-31_23-01-09/checkpoints/test_experiment--reduced_train_loss=8.2760-epoch=0"
        )
        lm: MegatronBioBertModel = distributed_model.module.module  # distributed_model is megatron_parallel, .module is the lightning module, and .module.module is the megatronbiobertmodel
        for param in lm.parameters():
            param.requires_grad = False
        # model_with_head = self.head.configure_model(lm)
        model_with_head = MLPHeadModel(lm, self, lm.config.hidden_size)
        return model_with_head

    def get_loss_reduction_class(self) -> Type[MegatronLossReduction]:
        # You could optionally return a different loss reduction class here based on the config settings.
        # return BERTMLMLossWithReduction
        # pass

        return ReductionLoss


def main(
    data_dir: Path,
    num_nodes: int,
    devices: int,
    seq_length: int,
    result_dir: Path,
    wandb_project: Optional[str],
    wandb_offline: bool,
    num_steps: int,
    limit_val_batches: int,
    val_check_interval: int,
    num_dataset_workers: int,
    biobert_spec_option: BiobertSpecOption,
    lr: float,
    micro_batch_size: int,
    cosine_rampup_frac: float,
    cosine_hold_frac: float,
    experiment_name: str,
    resume_if_exists: bool,
    precision: PrecisionTypes,
    wandb_entity: str = "clara-discovery",
    create_tensorboard_logger: bool = False,
    nemo1_init_path: Optional[Path] = None,
):
    """Train a Geneformer model on single cell data.

    Args:
        data_dir (Path): Base directory for the data.
        num_nodes (int): Number of nodes to run on
        devices (int): number of devices
        seq_length (int): sequence length
        result_dir (Path): directory to store results, logs and checkpoints
        wandb_project (Optional[str]): weights and biases project name
        wandb_offline (bool): if wandb should happen in offline mode
        num_steps (int): number of steps to train the model for
        limit_val_batches (int): limit the number of validation global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss and save
            an updated checkpoint
        num_dataset_workers (int): num dataset workers
        biobert_spec_option (BiobertSpecOption): the biobert spec option (architecture) to use for this run
        lr (float): learning rate
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        cosine_rampup_frac (float): fraction of steps at the beginning of the run to ramp up the learning rate
        cosine_hold_frac (float): fraction of steps to hold the minimum learning rate at the end of the run
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        resume_if_exists (bool): attempt to resume if the checkpoint exists [FIXME @skothenhill this doesn't work yet]
        wandb_entity (str): the group to use for the wandb run, sometimes called a team, could also be your username
        create_tensorboard_logger (bool): create the tensorboard logger


    """
    # Create the result directory if it does not exist.
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup train/test/val data paths
    train_data_path = data_dir / "train"
    val_data_path = data_dir / "val"
    test_data_path = data_dir / "test"

    # Setup the strategy and trainer
    pipeline_model_parallel_size = 1
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=False,
    )

    wandb_options: Optional[WandbLoggerOptions] = (
        None
        if wandb_project is None
        else WandbLoggerOptions(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            log_model=False,
        )
    )
    from nemo.lightning.io import track_io

    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,  # TODO(@jstjohn) Checkpoint saving is currently broken, fix and change this.
        num_nodes=num_nodes,
        callbacks=[track_io(LossLoggingCallback)(), track_io(RichModelSummary)(max_depth=4), track_io(LearningRateMonitor)()],
        plugins=nl.MegatronMixedPrecision(precision=precision, amp_O2=False),
    )

    # Preprocess the data to get the tokenizer and median dictionary
    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")

    # Configure the data module and model
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=train_data_path,
        val_dataset_path=val_data_path,
        test_dataset_path=test_data_path,
        random_token_prob=0.1,  # this is the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=micro_batch_size,
        global_batch_size=micro_batch_size * int(num_nodes * devices / pipeline_model_parallel_size),
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        num_workers=num_dataset_workers,
    )

    geneformer_config = BioBertFinetuneHeadConfig(
        trainer=trainer,
        output_size=20,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        gradient_accumulation_fusion=False,  # THIS BREAKS STUFF, leave False
        enable_autocast=False,  # This has to be set to True if we use the mixed precision plugin
    )

    # The lightning class owns a copy of the actual model, and a loss function, both of which are configured
    #  and lazily returned by the `geneformer_config` object defined above.
    model = BioBertLightningModule(
        geneformer_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                # TODO(@jstjohn) try decoupled_lr
                optimizer="adam",
                use_distributed_optimizer=True,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                # minimum learning rate is 1/100th of the initial learning rate, so eg lr=1e-3 -> min_lr=1e-5
                min_lr=lr / 100,
                warmup_steps=int(math.ceil(num_steps * cosine_rampup_frac)),
                interval="step",
                monitor="val_loss",
                constant_steps=int(math.ceil(num_steps * cosine_hold_frac)),
            ),
        ),
    )

    # from nemo.lightning.pytorch.callbacks import ModelCheckpoint
    # checkpoint_callback = ModelCheckpoint(
    #     save_best_model=False,
    #     save_last=True,
    #     monitor="reduced_train_loss",
    #     save_top_k=2,
    #     every_n_train_steps=10,
    #     enable_nemo_ckpt_io=True, # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
    #     async_save=True,
    # )

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_kwargs=wandb_options,
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        # FIXME @skothenhill this doesn't work yet, but this is probably close to what we are supposed to do
        resume=AutoResume(resume_if_exists=resume_if_exists, resume_ignore_no_checkpoint=True),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Geneformer with single cell data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the data base directory, for example this might be "
        "/workspace/bionemo2/data/cellxgene_2023-12-15_small",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=get_args(PrecisionTypes),
        required=False,
        default="bf16-mixed",
        help="Precision type to use for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=1e-4,
        help="Learning rate for training. Default is 1e-4. With bigger global batches try 1e-3",
    )
    parser.add_argument(
        "--create-tensorboard-logger", action="store_true", default=False, help="Create a tensorboard logger."
    )
    # FIXME (@skothenhill) figure out how checkpointing and resumption should work with the new nemo trainer
    parser.add_argument(
        "--resume-if-exists", action="store_true", default=False, help="Resume training if a checkpoint exists."
    )
    parser.add_argument(
        "--result-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
    )
    parser.add_argument(
        "--experiment-name", type=str, required=False, default="geneformer", help="Name of the experiment."
    )
    parser.add_argument("--wandb-offline", action="store_true", default=False, help="Use wandb in offline mode.")
    parser.add_argument(
        "--wandb-project",
        type=str,
        required=False,
        default=None,
        help="Wandb project name. Wandb will only happen if this is set..",
    )
    parser.add_argument(
        "--cosine-rampup-frac",
        type=float,
        required=False,
        default=0.01,
        help="Fraction of steps in which to ramp up the learning rate. Default is 0.01.",
    )
    parser.add_argument(
        "--cosine-hold-frac",
        type=float,
        required=False,
        default=0.05,
        help="Fraction of final steps in which to hold the minimum LR. Default is 0.05.",
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=10000,
        help="Number of steps to use for training. Default is 10000.",
    )
    parser.add_argument(
        "--num-dataset-workers",
        type=int,
        required=False,
        default=0,
        help="Number of steps to use for training. Default is 0.",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        required=False,
        default=10000,
        help="Number of steps to use for training. Default is 10000.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        required=False,
        default=2048,
        help="Sequence length of cell. Default is 2048.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        required=False,
        default=2,
        help="Number of steps to use for training. Default is 2.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=64,
        help="Micro-batch size. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--biobert-spec-option",
        type=BiobertSpecOption,
        choices=[e.value for e in BiobertSpecOption],
        required=False,
        default=BiobertSpecOption.bert_layer_local_spec.value,
        help="Biobert spec option to use for the model. Default is 'bert_layer_local_spec'.",
    )
    parser.add_argument(
        "--nemo1-init-path",
        type=Path,
        required=False,
        help="Path to nemo1 file, if desired to load at init time.",
    )

    # Parse the arguments and pull them out into local variables for ease of future refactor to a
    #   config management system.
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        seq_length=args.seq_length,
        result_dir=args.result_dir,
        wandb_project=args.wandb_project,
        wandb_offline=args.wandb_offline,
        num_steps=args.num_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        num_dataset_workers=args.num_dataset_workers,
        biobert_spec_option=args.biobert_spec_option,
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        cosine_rampup_frac=args.cosine_rampup_frac,
        cosine_hold_frac=args.cosine_hold_frac,
        precision=args.precision,
        experiment_name=args.experiment_name,
        resume_if_exists=args.resume_if_exists,
        nemo1_init_path=args.nemo1_init_path,
    )
