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


from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Type

import pytest
import torch
from _pytest.compat import LEGACY_PATH
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger, io, resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from bionemo.core.model.config import BionemoTrainableModelConfig
from bionemo.example_model import lightning_basic as lb
from bionemo.llm.lightning import LossLoggingCallback
from bionemo.testing import megatron_parallel_state_utils


class MetricTracker(Callback):
    def __init__(self, metrics_to_track_val: List[str], metrics_to_track_train: List[str]):
        self.metrics_to_track_val = metrics_to_track_val
        self.metrics_to_track_train = metrics_to_track_train
        self._collection_val = defaultdict(list)
        self._collection_train = defaultdict(list)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for metric in self.metrics_to_track_val:
            self._collection_val[metric].append(outputs[metric])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for metric in self.metrics_to_track_train:
            self._collection_train[metric].append(outputs[metric])

    def on_validation_epoch_end(self, trainer, pl_module):
        elogs = trainer.logged_metrics  # access it here
        self._collection_val["logged_metrics"].extend(elogs)

    def on_train_epoch_end(self, trainer, pl_module):
        elogs = trainer.logged_metrics  # access it here
        self._collection_train["logged_metrics"].extend(elogs)

    @property
    def collection_val(self) -> Dict[str, torch.Tensor | List[str]]:
        res = {k: torch.tensor(v) for k, v in self._collection_val.items() if k != "logged_metrics"}
        res["logged_metrics"] = self._collection_val["logged_metrics"]
        return res

    @property
    def collection_train(self) -> Dict[str, torch.Tensor | str]:
        res = {k: torch.tensor(v) for k, v in self._collection_train.items() if k != "logged_metrics"}
        res["logged_metrics"] = self._collection_train["logged_metrics"]
        return res


def _train_model_get_ckpt(
    name: str,
    root_dir: Path,
    data_dir: Path,
    model_cfg_cls: Type[BionemoTrainableModelConfig],
    ckpt_path: Path | None,
    skip_weight_prefixes: Set[str],
) -> Tuple[Path, MetricTracker]:
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_best_model=False,
        save_last=True,
        save_on_train_epoch_end=True,
        monitor="reduced_train_loss",  # TODO find out how to get val_loss logged and use "val_loss",
        every_n_train_steps=5,
        enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
        # async_save=False,  # Tries to save asynchronously, previously led to race conditions.
    )
    save_dir = root_dir / name
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)
    # Setup the logger and train the model
    nemo_logger = NeMoLogger(
        dir=str(root_dir),
        name=name,
        tensorboard=tb_logger,
        ckpt=checkpoint_callback,
    )
    # Needed so that the trainer can find an output directory for the profiler
    # nemo_logger.save_dir = tmpdir
    # ckpt_path needs to be a string for SerDe
    ckpt_path_optstr: str | None = str(ckpt_path) if ckpt_path is not None else None
    model = lb.LitAutoEncoder(
        config=model_cfg_cls(initial_weights=ckpt_path_optstr, skip_weight_prefixes=skip_weight_prefixes)
    )
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=True,
    )
    metric_tracker = MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"])
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        limit_val_batches=5,
        val_check_interval=5,
        max_steps=100,
        num_nodes=1,
        log_every_n_steps=5,
        callbacks=[LossLoggingCallback(), metric_tracker],
    )
    data_module = lb.MNISTDataModule(data_dir=str(data_dir), batch_size=64)  # Re-use the same data directory
    llm.train(
        model=model,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            path=None,  # Overrides the path found by resume_if_exists when set.
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
    ckpt_dirpath = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    return ckpt_dirpath, metric_tracker


@pytest.mark.needs_gpu
def test_train_mnist_litautoencoder_with_megatron_strategy_single_gpu(tmpdir: LEGACY_PATH):
    data_dir: Path = tmpdir / "data"
    data_dir.mkdir()
    with megatron_parallel_state_utils.clean_parallel_state_context():
        ckpt_path, initial_metrics = _train_model_get_ckpt(
            "test_experiment",
            tmpdir / "pretrain",
            data_dir,
            lb.ExampleConfig,
            None,
            set(),
        )
        assert ckpt_path.exists()
        assert ckpt_path.is_dir()
        assert io.is_distributed_ckpt(ckpt_path)
        assert initial_metrics.collection_train["loss"][0] > initial_metrics.collection_train["loss"][-1]
    with megatron_parallel_state_utils.clean_parallel_state_context():
        simple_ft_checkpoint, simple_ft_metrics = _train_model_get_ckpt(
            "simple_finetune_experiment",
            tmpdir / "simple_finetune",
            data_dir,
            lb.ExampleConfig,
            ckpt_path,
            set(),
        )
        assert simple_ft_checkpoint.exists()
        assert simple_ft_checkpoint.is_dir()
        assert io.is_distributed_ckpt(simple_ft_checkpoint)
        assert initial_metrics.collection_train["loss"][-1] > simple_ft_metrics.collection_train["loss"][0]
    with megatron_parallel_state_utils.clean_parallel_state_context():
        add_head_checkpoint, add_head_ft_metrics = _train_model_get_ckpt(
            "add_head_finetune_experiment",
            tmpdir / "add_head_finetune",
            data_dir,
            lb.ExampleFineTuneBothConfig,
            simple_ft_checkpoint,
            {"digit_classifier"},
        )
        assert add_head_checkpoint.exists()
        assert add_head_checkpoint.is_dir()
        assert io.is_distributed_ckpt(add_head_checkpoint)
        assert add_head_ft_metrics.collection_train["loss"][0] > add_head_ft_metrics.collection_train["loss"][-1]
        # We're adding a new loss, so the loss should be worse initially at least.
        assert add_head_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]

    with megatron_parallel_state_utils.clean_parallel_state_context():
        drop_head_checkpoint, drop_head_ft_metrics = _train_model_get_ckpt(
            "drop_head_finetune_experiment",
            tmpdir / "drop_head_finetune",
            data_dir,
            lb.ExampleFineTuneDropParentConfig,
            add_head_checkpoint,
            set(),
        )
        assert drop_head_checkpoint.exists()
        assert drop_head_checkpoint.is_dir()
        assert io.is_distributed_ckpt(drop_head_checkpoint)
        # We're dropping a loss, so initially we should be better than before
        assert drop_head_ft_metrics.collection_train["loss"][0] > drop_head_ft_metrics.collection_train["loss"][-1]
        assert add_head_ft_metrics.collection_train["loss"][-1] > drop_head_ft_metrics.collection_train["loss"][0]
