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

from typing import Callable, TypeVar
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.utils.data


IndexT = TypeVar("IndexT")

TensorLike = torch.Tensor | np.ndarray
TensorCollectionOrTensor = TensorLike | dict[str, TensorLike]


def assert_dict_tensors_approx_equal(actual: TensorCollectionOrTensor, expected: TensorCollectionOrTensor) -> None:
    """Assert that two tensors are equal."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        a_keys, b_keys = actual.keys(), expected.keys()
        assert a_keys == b_keys
        for key in a_keys:
            torch.testing.assert_close(actual=actual[key], expected=expected[key])
    else:
        torch.testing.assert_close(actual=actual, expected=expected)


class DatasetLocallyNondeterministic(AssertionError):
    """Datasets are not locally deterministic."""


class DatasetDistributedNondeterministic(AssertionError):
    """Datasets are not locally deterministic."""


def assert_dataset_compatible_with_megatron(
    dataset: torch.utils.data.Dataset[TensorCollectionOrTensor],
    index: IndexT = 0,
    assert_elements_equal: Callable[
        [TensorCollectionOrTensor, TensorCollectionOrTensor], None
    ] = assert_dict_tensors_approx_equal,
):
    """Make sure that a dataset passes some basic sanity checks for megatron determinism constraints.

    Constraints tested:
        * dataset[i] returns the same element regardless of device
        * dataset[i] doesn't make calls to known problematic randomization procedures (currently `torch.manual_seed`).

    As more constraints are discovered, they should be added to this test.
    """
    # 1. Make sure the dataset is deterministic when you ask for the same elements.
    n_elements = len(dataset)  # type: ignore
    assert n_elements > 0, "Need one element or more to test"
    try:
        assert_elements_equal(dataset[index], dataset[index])
    except AssertionError as e_0:
        raise DatasetLocallyNondeterministic(e_0)
    with patch("torch.manual_seed") as mock_manual_seed:
        _ = dataset[index]
    if mock_manual_seed.call_count > 0:
        raise DatasetDistributedNondeterministic(
            "You cannot safely use torch.manual_seed in a cluster with model parallelism. Use torch.Generator directly."
            " See https://github.com/NVIDIA/Megatron-LM/blob/dddecd19/megatron/core/tensor_parallel/random.py#L198-L199"
        )


def assert_dataset_elements_not_equal(
    dataset: torch.utils.data.Dataset[TensorCollectionOrTensor],
    index_a: IndexT = 0,
    index_b: IndexT = 1,
    assert_elements_equal: Callable[
        [TensorCollectionOrTensor, TensorCollectionOrTensor], None
    ] = assert_dict_tensors_approx_equal,
):
    """Test the case where two indices return different elements on datasets that employ randomness, like masking.

    With epoch upsampling approaches, some underlying index, say index=0, will be called multiple times by some wrapping
    dataset object. For example if you have a dataset of length 1, and you wrap it in an up-sampler that maps it to
    length 2 by mapping index 0 to 0 and 1 to 0, then when we apply randomness in that wrapping dataset and we expect
    different masks to be used for each call. Again this test only applies to a dataset that employs randomness. Another
    approach some of our datasets take is to use a special index that captures both the underlying index, and the epoch
    index. This tuple of indices is used internally to seed the mask. If that kind of dataset is used, then index_a
    could be (0,0) and index_b could be (0,1) for example. We expect those to return different random features.

    Args:
        dataset: dataset object with randomness (eg masking) to test.
        index_a: index for some element. Defaults to 0.
        index_b: index for a different element. Defaults to 1.
        assert_elements_equal: Function to compare two returned batch elements. Defaults to
            `assert_dict_tensors_approx_equal` which works for both tensors and dictionaries of tensors.
    """
    # 0, first sanity check for determinism/compatibility on idx0 and idx1
    assert_dataset_compatible_with_megatron(dataset, index=index_a, assert_elements_equal=assert_elements_equal)
    assert_dataset_compatible_with_megatron(dataset, index=index_b, assert_elements_equal=assert_elements_equal)
    # 1, now check that index_a != index_b
    with pytest.raises(AssertionError):
        assert_elements_equal(dataset[index_a], dataset[index_b])
