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

from typing import Sequence

from nemo.lightning import io

from bionemo.llm.model.biobert.model import BioBertGenericConfig, MegatronBioBertModel


__all__: Sequence[str] = (
    "GeneformerModel",
    "GeneformerConfig",
)

GeneformerModel = MegatronBioBertModel


class GeneformerConfig(BioBertGenericConfig[GeneformerModel], io.IOMixin):
    """A geneformer config overrides the parent config, and adds a leaf-level iomixin, please do not inherit from this
    directly, as your parameters will likely be reset to this method's parameters silently.
    """

    model_cls = GeneformerModel
