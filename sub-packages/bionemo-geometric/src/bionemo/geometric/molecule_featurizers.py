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


from rdkit.Chem import Descriptors, Mol

from bionemo.geometric.base_featurizer import (
    BaseFeaturizer,
)


N_RDKIT2D_FEATS = len(Descriptors.descList)


class RDkit2DDescriptorFeaturizer(BaseFeaturizer):
    """Class for featurizing molecule by 200 computed RDkit properties."""

    def __init__(self) -> None:
        """Initializes RDkit2DDescriptorFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 200

    def get_features(self, mol: Mol) -> list[float | int | bool]:
        """Returns features of the molecule."""
        return [f(mol) for desc_name, f in Descriptors.descList]
