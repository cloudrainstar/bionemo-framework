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
import pytest
from megatron.core import parallel_state

from bionemo.llm.utils.megatron_utils import is_only_data_parallel
from bionemo.testing import megatron_parallel_state_utils as mpsu


def test_no_parallelism_raises():
    with pytest.raises(RuntimeError):
        is_only_data_parallel()


def test_base_case():
    with mpsu.distributed_model_parallel_state():
        # our test instance with 1 GPU is trivially this case, also default initializations should be this case.
        assert is_only_data_parallel()


def test_pp2():
    with mpsu.mock_distributed_parallel_state(world_size=8, pipeline_model_parallel_size=2):
        assert not is_only_data_parallel()


def test_cp2():
    with mpsu.mock_distributed_parallel_state(world_size=8, context_parallel_size=2):
        assert not is_only_data_parallel()


def test_tp2():
    with mpsu.mock_distributed_parallel_state(world_size=8, tensor_model_parallel_size=2):
        assert not is_only_data_parallel()


def test_tp2pp2cp2():
    with mpsu.mock_distributed_parallel_state(
        world_size=8, tensor_model_parallel_size=2, pipeline_model_parallel_size=2, context_parallel_size=2
    ):
        assert not is_only_data_parallel()


def test_tp8():
    with mpsu.mock_distributed_parallel_state(world_size=8, tensor_model_parallel_size=8):
        assert not is_only_data_parallel()


def test_dp_only():
    with mpsu.mock_distributed_parallel_state(world_size=8):
        assert is_only_data_parallel()


def test_get_pp_group():
    with mpsu.mock_distributed_parallel_state(world_size=2, pipeline_model_parallel_size=2):
        assert parallel_state.get_pipeline_model_parallel_group() is not None


def test_get_tp_group():
    with mpsu.mock_distributed_parallel_state(world_size=2, tensor_model_parallel_size=2):
        assert parallel_state.get_tensor_model_parallel_group() is not None


def test_get_cp_group():
    with mpsu.mock_distributed_parallel_state(world_size=2, context_parallel_size=2):
        assert parallel_state.get_context_parallel_group() is not None


def test_pp_stage():
    with mpsu.mock_distributed_parallel_state(world_size=2, pipeline_model_parallel_size=2, rank=0):
        assert parallel_state.is_pipeline_first_stage()
        assert not parallel_state.is_pipeline_last_stage()

    with mpsu.mock_distributed_parallel_state(world_size=2, pipeline_model_parallel_size=2, rank=1):
        assert not parallel_state.is_pipeline_first_stage()
        assert parallel_state.is_pipeline_last_stage()
