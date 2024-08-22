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

from dataclasses import dataclass, field
from typing import Type

from megatron.core.transformer import TransformerConfig
from nemo.lightning import io

from bionemo.core.model.config import BionemoModelConfig, BionemoTrainableModelConfig, Loss, Model


@dataclass(kw_only=True)
class MegatronBioNeMoModelConfig(BionemoModelConfig[Model], TransformerConfig, io.NeedsIOMixin):
    """A ModelConfig class for bionemo that supports usage with Megatron models, for example as NeMo2 requires."""

    model_cls: Type[Model] = field(init=False)


@dataclass(kw_only=True)
class MegatronBioNeMoTrainableModelConfig(MegatronBioNeMoModelConfig[Model], BionemoTrainableModelConfig[Model, Loss]):
    pass
