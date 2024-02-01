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

from bionemo.model.protein.openfold.attention import Attention, CrossAttentionNoGate
from bionemo.model.protein.openfold.optim_hub import OptimHub


class TemplatePointwiseAttention(nn.Module):
    """Template Pointwise Attention module.

    Supplementary '1.7.1 Template stack': Algorithm 17.

    Args:
        c_t: Template representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden: Hidden dimension (per-head).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_t: int,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super(TemplatePointwiseAttention, self).__init__()
        if OptimHub.config('mha_fused_gemm'):  # [optim-hub]
            self.mha = CrossAttentionNoGate(
                c_q=c_z,
                c_kv=c_t,
                c_hidden=c_hidden,
                num_heads=num_heads,
                inf=inf,
                chunk_size=chunk_size,
            )
        else:
            self.mha = Attention(
                c_q=c_z,
                c_k=c_t,
                c_v=c_t,
                c_hidden=c_hidden,
                num_heads=num_heads,
                gating=False,
                inf=inf,
                chunk_size=chunk_size,
            )

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        template_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Template Pointwise Attention forward pass.

        Args:
            t: [batch, N_templ, N_res, N_res, c_t] template representation
            z: [batch, N_res, N_res, c_z] pair representation
            template_mask: [batch, N_templ] template mask

        Returns:
            z_update: [batch, N_res, N_res, c_z] pair representation update
                from template representation

        """
        t = t.movedim(-4, -2)
        # t: [batch, N_res, N_res, N_templ, c_t]

        z = z.unsqueeze(-2)
        # z: [batch, N_res, N_res, 1, c_z]

        template_mask = template_mask.unsqueeze(-2).unsqueeze(-3).unsqueeze(-4).unsqueeze(-5)
        # template_mask: [batch, 1, 1, 1, 1, N_templ]

        if OptimHub.config('mha_fused_gemm'):
            z = self.mha(
                input_q=z,
                input_kv=t,
                mask=template_mask,
                bias=None,
            )
        else:
            z = self.mha(
                input_q=z,
                input_k=t,
                input_v=t,
                mask=template_mask,
                bias=None,
            )
        # z: [batch, N_res, N_res, 1, c_z]

        z = z.squeeze(-2)
        # z: [batch, N_res, N_res, c_z]

        return z
