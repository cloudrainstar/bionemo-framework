# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from omegaconf import ListConfig
import torch
from typing import List, Union
from pytorch_lightning.core import LightningModule
from pandas import Series

from nemo.utils import logging

from bionemo.model.utils import _reconfigure_inference_batch
from bionemo.model.utils import restore_model
from bionemo.data.utils import pad_token_ids

try:
    from apex.transformer import parallel_state

    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

# FIXME: add mask for all non-special tokens (add hiddens_tokens_only)
# TODO: add model-specific prepare_for_inference and release_from_inference methods
class BaseEncoderDecoderInference(LightningModule):
    '''
    Base class for inference.
    '''

    def __init__(self, cfg, model=None, freeze=True, restore_path=None, training=False):
        super().__init__()

        self.cfg = cfg
        self._freeze_model = freeze
        self.training = training
        self.model = self.load_model(cfg, model=model, restore_path=restore_path)
        self._trainer = self.model.trainer
        self.tokenizer = self.model.tokenizer
    
    def load_model(self, cfg, model=None, restore_path=None):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            Loaded model
        """        
        # load model class from config which is required to load the .nemo file
        if model is None:
            if restore_path is None:
                restore_path = cfg.model.downstream_task.restore_from_path
            model = restore_model(
                restore_path=restore_path,
                cfg=cfg,
            )
        # move self to same device as loaded model
        self.to(model.device)

        # check whether the DDP is initialized
        if parallel_state.is_unitialized():
            logging.info("DDP is not initialized. Initializing...")
            def dummy():
                return

            if model.trainer.strategy.launcher is not None:
                model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
            model.trainer.strategy.setup_environment()

        # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
        _reconfigure_microbatch_calculator(
            rank=0,  # This doesn't matter since it is only used for logging
            rampup_batch_size=None,
            global_batch_size=1,
            micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
            data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
        )
        if self._freeze_model:
            model.freeze()

        self.model = model

        return model

    def forward(self, batch):
        """Forward pass of the model. Can return embeddings or hiddens, as required"""
        sequences = batch[self.cfg.model.data.data_fields_map.sequence]
        sequence_ids = batch[self.cfg.model.data.data_fields_map.id]
        prediction_data = {"sequence_ids": sequence_ids}
        outputs = self.cfg.model.downstream_task.outputs
        # make sure we have a list
        if not isinstance(outputs, ListConfig):
            outputs = [outputs]

        # adjust microbatch size
        _reconfigure_inference_batch(global_batch_per_gpu=len(sequences))
        
        with torch.set_grad_enabled(self._freeze_model):
            for output_type in outputs:
                if output_type == 'hiddens':
                    hiddens, mask = self.seq_to_hiddens(sequences)
                    prediction_data["hiddens"] = hiddens
                    prediction_data["mask"] = mask
                elif output_type == 'embeddings':
                    prediction_data["embeddings"] = self.seq_to_embeddings(sequences)
                else:
                    raise ValueError(f"Invalid prediction type: {self.cfg.model.downstream_task.prediction}")
        
        return prediction_data
    
    def _tokenize(self, sequences: List[str]):
        """
        Model specific tokenization.
        Here <BOS> and <EOS> tokens are added for instance.
        
        Returns:
            token_ids (torch.Tensor, long): token ids
        """
        raise NotImplementedError("Please implement in child class")

    def tokenize(self, sequences: List[str]):
        """
        Tokenize sequences.
        Returns:
            token_ids (torch.Tensor, long): token ids
            mask (torch.Tensor, long, float): boolean mask for padded sections
        """
        token_ids = self._tokenize(sequences=sequences)

        # Validate input sequences length
        if any([len(t) > self.model.cfg.seq_length for t in token_ids]):
            raise Exception(f'One or more sequence exceeds max length({self.model.cfg.seq_length}).')

        # Pad token ids (1/True = Active, 0/False = Inactive)
        token_ids, mask = pad_token_ids(
            token_ids, 
            padding_value=self.tokenizer.pad_id, 
            device=self.device,
            )
        
        return token_ids, mask

    def detokenize(self, tokens_ids: List[str]):
        """
        Detokenize a matrix of tokens into a list of sequences (i.e., strings).

        Args:
            tokens_ids (torch.Tensor, long): a matrix of token ids

        Returns:
            sequences (list[str]): list of sequences
        """
        tokens_ids = tokens_ids.cpu().detach().numpy().tolist()
        sequences = []
        for i, cur_tokens_id in enumerate(tokens_ids):
            if self.tokenizer.eos_id in cur_tokens_id:
                idx = cur_tokens_id.index(self.tokenizer.eos_id)
                tokens_ids[i] = cur_tokens_id[:idx]
            else:
                tokens_ids[i] = [id for id in cur_tokens_id if id != self.tokenizer.pad_id]

        sequences = self.tokenizer.ids_to_text(tokens_ids)

        return sequences

    def seq_to_hiddens(self, sequences: List[str]):
        '''
        Transforms Sequences into hidden state.
        This class should be implemented in a child class, since it is model specific.
        This class should return only the hidden states, without the special tokens such as
         <BOS> and <EOS> tokens, for example.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''
        raise NotImplementedError("Please implement in child class")

    def hiddens_to_embedding(self, hidden_states, enc_mask):
        '''
        Transforms hidden_states into embedding.

        Args:
            hidden_states (torch.Tensor, float): hidden states
            enc_mask (torch.Tensor, long): boolean mask for padded sections

        Returns:
            embeddings (torch.Tensor, float):
        '''
        # compute average on active hiddens
        lengths = enc_mask.sum(dim=1, keepdim=True)
        if (lengths == 0).any():
            raise ValueError("Empty input is not supported (no token was proveded in one or more of the inputs)")

        embeddings = torch.sum(hidden_states*enc_mask.unsqueeze(-1), dim=1) / lengths
        
        return embeddings

    def seq_to_embeddings(self, sequences: List[str]):
        """Compute hidden-state and padding mask for sequences.

        Params
            sequences: strings, input sequences

        Returns
            embedding array
        """
        # get hiddens and mask
        hiddens, enc_mask = self.seq_to_hiddens(sequences)
        # compute embeddings from hiddens
        embeddings = self.hiddens_to_embedding(hiddens, enc_mask)

        return embeddings

    def hiddens_to_seq(self, hidden_states, enc_mask):
        '''
        Transforms hidden state into sequences (i.e., sampling in most cases).
        This class should be implemented in a child class, since it is model specific.
        This class should return the sequence with special tokens such as
         <BOS> and <EOS> tokens, if used.

        Args:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections

        Returns:
            sequences (list[str]): list of sequences
        '''
        raise NotImplementedError("Please implement in child class")

    @property
    def supported_sampling_methods(self):
        """
        Returns a list of supported sampling methods.
        Example:
            ["greedy-perturbate", "beam-search"]
        """
        return list(self.default_sampling_kwargs.keys())

    @property
    def default_sampling_kwargs(self):
        """
        Returns a dict of default sampling kwargs per sampling method.
        Example:
            {
                "greedy-perturbate": {"scaled_radius": 1, "topk": 10},
                "beam-search": {"beam_size": 5, "beam_alpha": 0.6, "beam_min_length": 1, "beam_max_length": 100},
            }
            
        Should be overridden in child class if sampling is supported.
        """
        return {}
    
    def sample(self,
               num_samples=1,
               return_embedding=False,
               sampling_method=None,
               **sampling_kwarg):
        """
        Sample from the model given sampling_method.
        
        Args:
            num_samples (int): number of samples to generate (depends on sampling method)
            return_embedding (bool): return embeddings corresponding to each of the samples in addition to the samples
            sampling_method (str): sampling method to use. Should be replaced with default sampling method in child class
            sampling_kwarg (dict): kwargs for sampling method. Depends on the sampling method.
        """
        raise NotImplementedError(f"Sampling is not supported in this class ({self.__class__.__name__})")

    def __call__(self, sequences: Union[Series, List[str]]) -> torch.Tensor:
        """
        Computes embeddings for a list of sequences.
        Embeddings are detached from model.
        
        Params
            sequences: Pandas Series containing a list of strings or or a list of strings (e.g., SMILES)
            
        Returns
            embeddings
        """
        if isinstance(sequences, Series):
            sequences = sequences.tolist()

        return self.seq_to_embeddings(sequences).float().detach().clone()
