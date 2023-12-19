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

"""
Runs inference over all models. Supports extracting embeddings, and hiddens.

NOTE: If out of memory (OOM) error occurs, try spliting the data to multiple smaller files.
"""
import os
import pickle
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig

from bionemo.model.loading import setup_inference
from bionemo.model.run_inference import extract_output_filename, predict_ddp, validate_output_filename


__all__: Sequence[str] = ()  # pragma: no cover


@hydra_runner()
def entrypoint(cfg: DictConfig) -> None:
    """Provide --config-dir and --config-name when using from the CLI."""
    model, trainer, dataloader = setup_inference(cfg)

    try:
        output_fname: str = extract_output_filename(cfg)
    except ValueError:
        filename = "infer_output"
        # Convention: put into cfg's exp_manager.exp_dir if possible!
        try:
            output_fname = str(Path(cfg.exp_manager.exp_dir).absolute() / filename)
        except AttributeError:
            logging.warning("No exp_manager.exp_dir in configuration! Must write output to current directory.")
            output_fname = f"./{filename}"

    if os.path.exists(output_fname):
        output_fname += f"--{uuid4()}.pkl"
        logging.warning(
            "Configuration's output filename already exists! " f"Making unique with filename suffix: {output_fname=}"
        )

    if not output_fname.endswith(".pkl"):
        output_fname += ".pkl"
        logging.warning(f"Ensuring that output filename has .pkl extension: {output_fname}")

    validate_output_filename(output_fname, overwrite=False)

    # predict outputs for all sequences in batch mode
    predictions = predict_ddp(model, dataloader, trainer, cfg.model.downstream_task.outputs)
    if predictions is None:
        logging.info("From non-rank 0 process: exiting now. Rank 0 will gather results and write.")
        return
    # from here only rank 0 should continue

    logging.info(f"Saving {len(predictions)} samples to {output_fname}")
    with open(output_fname, "wb") as wb:
        pickle.dump(predictions, wb)


if __name__ == '__main__':
    entrypoint()
