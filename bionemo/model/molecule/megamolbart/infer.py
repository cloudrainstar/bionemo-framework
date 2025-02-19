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


import logging
from typing import Sequence

from bionemo.model.molecule.infer import MolInference


log = logging.getLogger(__name__)

__all__: Sequence[str] = ("MegaMolBARTInference",)


class MegaMolBARTInference(MolInference):
    """Inferenec class for MegaMolBART. If any specific methods are needed for this vs molmim
    we will refactor, for now see `MolInference` for documentation.
    """
