# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch

from bionemo.model.core.infer import BaseEncoderDecoderInference, SeqsOrBatch


log = logging.getLogger(__name__)

__all__: Sequence[str] = (
    "MegaMolBARTInference",
    "SamplingMethods",
)

SamplingMethods = Literal["greedy-perturbate", "topkp-perturbate", "beam-search-perturbate"]


class MegaMolBARTInference(BaseEncoderDecoderInference):
    '''
    All inference functions
    '''

    def __init__(
        self,
        cfg,
        model: Optional[Any] = None,
        freeze: bool = True,
        restore_path: Optional[str] = None,
        training: bool = False,
        adjust_config: bool = True,
        interactive: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            model=model,
            freeze=freeze,
            restore_path=restore_path,
            training=training,
            adjust_config=adjust_config,
            interactive=interactive,
        )

    def _tokenize(self, sequences: List[str]) -> List[int]:
        """
        ProtT5 expects input/output format:

        encoder input ids - [tokens] (without <BOS> and <EOS>)
        decoder input ids - <BOS> + [tokens]
        decoder output ids - [tokens] + <EOS>
        """
        # Tokenize sequences
        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]

        return token_ids

    def seq_to_hiddens(self, sequences) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Transforms Sequences into hidden state.
        Should be implemented in a child class, since it is model specific.
        This method returns hidden states and masks.
        Hiddens states contain paddings but do not contain special tokens
        such as <BOS> and <EOS> tokens.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''
        token_ids, enc_mask = self.tokenize(sequences)
        embedding = self.model.encode(
            tokens_enc=token_ids,
            enc_mask=enc_mask,
            reconfigure_microbatch=not self.interactive,
        )

        return embedding, enc_mask

    def hiddens_to_seq(self, hidden_states, enc_mask, **kwargs) -> SeqsOrBatch:
        '''
        Transforms hidden state into sequences (i.e., sampling in most cases).
        This class should be implemented in a child class, since it is model specific.
        This class should return the sequence with special tokens such as
         <BOS> and <EOS> tokens, if used.

        Args:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        Returns:
            sequences (list[str]) or list[list[str]]): list of sequences
        '''

        predicted_tokens_ids, _ = self.model.decode(
            tokens_enc=None,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.model.cfg.max_position_embeddings,
            enc_output=hidden_states,
            reconfigure_microbatch=not self.interactive,
            **kwargs,
        )
        sequences = self.detokenize(tokens_ids=predicted_tokens_ids)
        return sequences

    @property
    def default_sampling_kwargs(self) -> Dict[SamplingMethods, Dict[str, Any]]:
        """
        Returns a dict of default sampling kwargs per sampling method.
        """
        return {
            # smis - a list of SMILES strings to perturbate num_samples times each
            "greedy-perturbate": {"scaled_radius": 1, "smis": []},
            # top-k limits maximum number of token candidtaes, top-p can further reduce to accumulate top-p probability mass
            "topkp-perturbate": {"scaled_radius": 1, "smis": [], "top_k": 0, "top_p": 0.9, "temperature": 1.0},
            # beam search, "beam_size" is number of the best sequences at each decode iteration to be left per target
            # and "beam_alpha" is the parameter of length penalty applied to predicted sequences
            "beam-search-perturbate": {"scaled_radius": 1, "smis": [], "beam_size": 1, "beam_alpha": 0},
        }

    def sample(
        self,
        num_samples: Optional[int] = 10,
        return_embedding: bool = False,
        sampling_method: SamplingMethods = "greedy-perturbate",
        **sampling_kwarg,
    ) -> Union[SeqsOrBatch, Tuple[SeqsOrBatch, torch.Tensor]]:
        """
        Sample from the model given sampling_method.

        Args:
            num_samples (int): number of samples to generate (depends on sampling method)
            return_embedding (bool): return embeddings corresponding to each of the samples in addition to the samples
            sampling_method (str): sampling method to use. Should be replaced with default sampling method in child class
            sampling_kwarg (dict): kwargs for sampling method. Depends on the sampling method.
        """
        # get sampling kwargs
        default_sampling_kwarg = self.default_sampling_kwargs
        if sampling_method not in default_sampling_kwarg:
            raise ValueError(
                f'Invalid samping method {sampling_method}, supported sampling methods are {default_sampling_kwarg.keys()}'
            )

        cur_sampling_kwarg = default_sampling_kwarg[sampling_method].copy()
        cur_sampling_kwarg.update(sampling_kwarg)
        sampling_kwarg = cur_sampling_kwarg

        # execute selected sampling method
        assert (
            sampling_method in default_sampling_kwarg.keys()
        ), f'Invalid sampling method {sampling_method}, supported sampling methods are {list(default_sampling_kwarg.keys())}'

        smis = sampling_kwarg.pop("smis")
        if len(smis) == 0:
            raise ValueError('No SMILES strings provided for sampling via "smis" argument')

        hidden_states, enc_masks = self.seq_to_hiddens(smis)

        if sampling_method in ['greedy-perturbate', 'topkp-perturbate']:
            sample_masks = enc_masks.repeat_interleave(num_samples, 0)
            perturbed_hiddens = hidden_states.repeat_interleave(num_samples, 0)
        else:
            sample_masks = enc_masks.clone()
            perturbed_hiddens = hidden_states.clone()

        scaled_radius = sampling_kwarg.pop('scaled_radius')
        perturbed_hiddens = perturbed_hiddens + (
            scaled_radius * torch.randn(perturbed_hiddens.shape).to(perturbed_hiddens.device)
        )

        if sampling_method == 'greedy-perturbate':
            samples = self.hiddens_to_seq(
                perturbed_hiddens, sample_masks, sampling_method="greedy-search", sampling_kwargs={}
            )

        elif sampling_method == 'topkp-perturbate':
            samples = self.hiddens_to_seq(
                perturbed_hiddens, sample_masks, sampling_method="topkp-sampling", sampling_kwargs=sampling_kwarg
            )

        elif sampling_method == 'beam-search-perturbate':
            if num_samples is not None:
                sampling_kwarg['beam_size'] = num_samples
            samples = self.hiddens_to_seq(
                perturbed_hiddens, sample_masks, sampling_method="beam-search", sampling_kwargs=sampling_kwarg
            )

        else:
            raise ValueError(f"Unknown {sampling_method=}, supported {SamplingMethods=}")

        if return_embedding:
            embs = self.hiddens_to_embedding(perturbed_hiddens, sample_masks)
            return samples, embs
        else:
            return samples
