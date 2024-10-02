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

from typing import Callable, Optional, Sequence, TypeVar
from unittest.mock import patch

import torch
import torch.utils.data


__all__: Sequence[str] = (
    "assert_matrix_mape_below_value",
    "assert_matrix_correlation_above_value",
)

BatchT = TypeVar("BatchT")


def assert_dict_tensors_approx_equal(
    actual: dict[str, torch.Tensor] | torch.Tensor, expected: dict[str, torch.Tensor] | torch.Tensor
) -> None:
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
    dataset: torch.utils.data.Dataset[BatchT],
    assert_elements_equal: Callable[[BatchT, BatchT], None] = assert_dict_tensors_approx_equal,
):
    # 1. Make sure the dataset is deterministic when you ask for the same elements.
    n_elements = len(dataset)  # type: ignore
    assert n_elements > 0, "Need one element or more to test"
    try:
        assert_elements_equal(dataset[0], dataset[0])
    except AssertionError as e_0:
        raise DatasetLocallyNondeterministic(e_0)
    with patch("torch.manual_seed") as mock_manual_seed:
        _ = dataset[0]
    if mock_manual_seed.call_count > 0:
        raise DatasetDistributedNondeterministic(
            "You cannot safely use torch.manual_seed in a cluster with model parallelism. Use torch.Generator directly."
            " See https://github.com/NVIDIA/Megatron-LM/blob/dddecd19/megatron/core/tensor_parallel/random.py#L198-L199"
        )


def assert_matrix_mape_below_value(  # noqa: D417
    actual: torch.Tensor,
    expected: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_mape: float = 0.1,
    eps: float = 1e-3,
    msg: str = "",
) -> None:
    """Assert that two tensors are close with a root mean squared error (RMSE)
        relative to the scaled root mean square values for each matrix. This tells
        you if the RMSE implies that the two matrices are more similar to eachother
        as-is than would be the case if values were randomly permuted.

    Args:
        actual: The actual tensor.
        expected: The expected tensor.
        mask: If there are only some values you want to compare,
            apply this mask and RMSE will be computed on the unmasked items only.
        min_relative_rmse: The relative tolerance parameter.
    """  # noqa: D205
    if mask is None:
        mask = torch.ones_like(actual)
    else:
        if len(mask.shape) < len(actual.shape):
            mask = mask[..., None]
    masked_actual = actual[mask.expand_as(actual).to(bool)]
    masked_expected = expected[mask.expand_as(expected).to(bool)]
    mape = (
        torch.mean(
            torch.abs(masked_actual - masked_expected)
            / torch.maximum(torch.abs(masked_expected), torch.zeros_like(masked_expected) + eps)
        )
        * 100.0
    )
    if mape > max_mape:
        raise AssertionError(f"MAPE below threshold: {mape} > {max_mape}. {msg}")


def assert_matrix_correlation_above_value(  # noqa: D417
    actual: torch.Tensor,
    expected: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    min_correlation: float = 0.95,
    msg: str = "",
) -> None:
    """Assert that two tensors are close with a root mean squared error (RMSE)
        relative to the scaled root mean square values for each matrix. This tells
        you if the RMSE implies that the two matrices are more similar to eachother
        as-is than would be the case if values were randomly permuted.

    Args:
        actual: The actual tensor.
        expected: The expected tensor.
        mask: If there are only some values you want to compare,
            apply this mask and RMSE will be computed on the unmasked items only.
        min_relative_rmse: The relative tolerance parameter.
    """  # noqa: D205
    if mask is None:
        mask = torch.ones_like(actual)
    else:
        if len(mask.shape) < len(actual.shape):
            mask = mask[..., None]
    masked_actual = actual[mask.expand_as(actual).to(bool)]
    masked_expected = expected[mask.expand_as(expected).to(bool)]
    corr = torch.corrcoef(torch.stack([masked_actual, masked_expected]))[0, 1]
    if corr < min_correlation:
        raise AssertionError(f"Correlation below threshold: {corr} < {min_correlation}. {msg}")
