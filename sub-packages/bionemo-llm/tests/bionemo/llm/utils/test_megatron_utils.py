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
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed

from bionemo.llm.utils.megatron_utils import is_only_data_parallel
from bionemo.testing import megatron_parallel_state_utils


def test_no_parallelism_raises():
    with pytest.raises(RuntimeError):
        is_only_data_parallel()


def test_base_case_false():
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        # our test instance with 1 GPU is trivially this case, also default initializations should be this case.
        assert is_only_data_parallel()


def test_mp_case(mocker, mocked_world_size: int = 8):
    state_util = megatron_parallel_state_utils.Utils  # static
    # Conditionally mock torch.distributed.new_group based on backend argument
    ori_dist_new_group = torch.distributed.new_group

    def mock_new_group(*args, **kwargs):
        if kwargs.get("backend") == "gloo":
            # Return a specific mock if backend is 'gloo'
            return MagicMock(name="gloo_group")
        else:
            # Return another mock or a different behavior for other backends
            return ori_dist_new_group(*args, **kwargs)

    ori_destroy_pg = torch.distributed.destroy_process_group

    def mock_destroy_gloo_group(pg=None):
        if isinstance(pg, MagicMock):
            return None
        ori_destroy_pg(pg)

    # Apply the conditional mock to torch.distributed.new_group
    mocker.patch("torch.distributed.new_group", side_effect=mock_new_group)
    mocker.patch("torch.distributed.destroy_process_group", side_effect=mock_destroy_gloo_group)
    state_util.set_world_size(world_size=mocked_world_size, rank=0)
    state_util.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    assert not is_only_data_parallel()
    state_util.destroy_model_parallel()
    state_util.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    assert is_only_data_parallel()
    state_util.destroy_model_parallel()
