# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

import torch
import torch.nn as nn

from bionemo.model.protein.openfold.attention import Attention
from bionemo.model.protein.openfold.layer_norm import LayerNorm


class MSAColumnAttention(nn.Module):
    """MSA Column Attention module.

    Supplementary '1.6.2 MSA column-wise gated self-attention': Algorithm 8.

    Args:
        c_m: MSA representation dimension (channels).
        c_hidden: Per-head hidden dimension (channels).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_m: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super(MSAColumnAttention, self).__init__()
        self.layer_norm_m = LayerNorm(c_m)
        self.mha = Attention(
            c_q=c_m,
            c_k=c_m,
            c_v=c_m,
            c_hidden=c_hidden,
            num_heads=num_heads,
            gating=True,
            inf=inf,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSA Column Attention forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask

        Returns:
            m_update: [batch, N_seq, N_res, c_m] MSA representation update

        """
        m = m.transpose(-2, -3)
        # m: [batch, N_res, N_seq, c_m]

        mask = mask.transpose(-1, -2)
        # mask: [batch, N_res, N_seq]

        mask = mask.unsqueeze(-2).unsqueeze(-3)
        # mask: [batch, N_res, 1, 1, N_seq]

        m = self.layer_norm_m(m)
        m = self.mha(
            input_q=m,
            input_k=m,
            input_v=m,
            mask=mask,
            bias=None,
        )
        # m: [batch, N_res, N_seq, c_m]

        m = m.transpose(-2, -3)
        # m: [batch, N_seq, N_res, c_m]

        return m
