# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

import pathlib
import pickle
from argparse import ArgumentParser
from typing import Tuple

import torch


"""
This script is designed to compare the outputs of the ESM2 model under different Tensor Parallel (TP) configurations.
It achieves this by loading model predictions from two  pickle files generated by `examples/protein/esm2nv/infer.sh` script,
and comparing the outputs using the `compare_predictions` function.

Example to run this conversion script:
    python compare.py \
    --prediction_file_1 "/data/esm2_tp1.pkl" \
    --prediction_file_2 "/data/esm2_tp2.pkl"
"""


def compare_outputs(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """
    Compares two PyTorch tensors, `x` and `y`, by computing the maximum absolute difference
    and the mean relative difference between them.
    This function is used for comparing outputs of models.

    Parameters:
    ----------
        x: The first tensor to compare.
        y: The second tensor to compare.

    Returns:
    -------
        Tuple[float, float]: A tuple containing the maximum absolute difference and
            the mean relative difference between the two tensors.

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> y = torch.tensor([1.001, 2.0, 3.0])
        >>> compare_outputs(x, y)
        ('max_absolute_diff = 0.001', 'mean_relative_difference = 0.00033333')
    """
    max_absolute_diff = torch.max(torch.abs(x - y)).item()
    mean_relative_diff = torch.mean(torch.abs(x - y) / (torch.abs(x) + 1e-8)).item()
    return max_absolute_diff, mean_relative_diff


def compare_predictions(outputs_file_x: pathlib.Path, outputs_file_y: pathlib.Path):
    """
    Loads predictions from two pkl files generated by the ESM1nvInference class (via `infer.sh` script).

    This function is used to evaluate the consistency of model predictions under varying TP partitions.

    The function assumes that each pkl file contains a dictionary with keys including 'hiddens',
    where the value is a list of tensors representing model embeddings.
    The comparison is done on the first element of the 'hiddens' list from each file.

    Parameters:
    ----------
        outputs_file_x: Path to the pkl file storing test predictions of the ESM2 model with a given TP partition.
        outputs_file_y: Path to the pkl file storing test predictions of the ESM2 model with a different TP partition.

    Note:
        This function relies on `compare_outputs` to perform the actual comparison.
    """
    predictions_1 = pickle.load(open(outputs_file_x, "rb"))
    predictions_2 = pickle.load(open(outputs_file_y, "rb"))
    compare_outputs(torch.Tensor(predictions_1[1]["hiddens"]), torch.Tensor(predictions_2[1]["hiddens"]))


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--prediction_file_1",
        type=pathlib.Path,
        default=None,
        required=True,
        help="Path to the prediction file to load for comparison",
    )
    parser.add_argument(
        "--prediction_file_2",
        type=pathlib.Path,
        default=None,
        required=True,
        help="Path to the second prediction file to load for comparison",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    compare_predictions(args.prediction_file_2, args.prediction_file_2)
