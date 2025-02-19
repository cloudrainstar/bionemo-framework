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


# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.optim.lr_scheduler import _LRScheduler


class AlphaFoldLRScheduler(_LRScheduler):
    """Implements the learning rate schedule defined in the AlphaFold 2
    supplement. A linear warmup is followed by a plateau at the maximum
    learning rate and then exponential decay.

    Note that the initial learning rate of the optimizer in question is
    ignored; use this class' base_lr parameter to specify the starting
    point of the warmup.
    """

    def __init__(
        self,
        optimizer,
        last_epoch: int = -1,
        verbose: bool = False,
        base_lr: float = 0.0,
        max_lr: float = 0.001,
        warmup_no_steps: int = 1000,
        start_decay_after_n_steps: int = 50000,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ):
        step_counts = {
            "warmup_no_steps": warmup_no_steps,
            "start_decay_after_n_steps": start_decay_after_n_steps,
        }

        for k, v in step_counts.items():
            if v < 0:
                raise ValueError(f"{k} must be nonnegative")

        if warmup_no_steps > start_decay_after_n_steps:
            raise ValueError("warmup_no_steps must not exceed start_decay_after_n_steps")

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_no_steps = warmup_no_steps
        self.start_decay_after_n_steps = start_decay_after_n_steps
        self.decay_every_n_steps = decay_every_n_steps
        self.decay_factor = decay_factor

        super(AlphaFoldLRScheduler, self).__init__(
            optimizer,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def state_dict(self):
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}

        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            raise RuntimeError("To get the last learning rate computed by the scheduler, use " "get_last_lr()")

        step_no = self.last_epoch

        if step_no < self.warmup_no_steps:
            lr = self.base_lr + (step_no / self.warmup_no_steps) * self.max_lr
        elif step_no > self.start_decay_after_n_steps:
            steps_since_decay = step_no - self.start_decay_after_n_steps
            exp = (steps_since_decay // self.decay_every_n_steps) + 1
            lr = self.max_lr * (self.decay_factor**exp)
        else:  # plateau
            lr = self.max_lr

        return [lr for group in self.optimizer.param_groups]
