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


import logging
import time
from functools import wraps

import pandas as pd

from bionemo.core.data.multi_epoch_dataset import EpochIndex
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.testing.data.load import load


def timeit(method):
    """Wrapper to time functions."""

    @wraps(method)
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Method {method.__name__} took {run_time:.4f} seconds")
        return result, run_time

    return timed


def time_all_methods(cls):
    """Time all methods in class."""
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and attr_name != "__init__":  # Check if the attribute is a method
            setattr(cls, attr_name, timeit(attr_value))
    return cls


@time_all_methods
class GeneformerDatasetMetrics:
    """SCDL Metrics."""

    def __init__(self, memmap_dir, tokenizer, median_dict):
        """Instantiate class."""
        self.memmap_dir = memmap_dir
        self.tokenizer = tokenizer
        self.median_dict = median_dict

    def create_from_memmap(self):
        """Create from memmap dir."""
        self.ds = SingleCellDataset(
            self.memmap_dir, tokenizer=self.tokenizer, median_dict=self.median_dict, bypass_tokenizer_vocab=True
        )

    def get_length(self):
        """Length."""
        self.length = len(self.ds)
        return self.length

    def get_first_item(self):
        """Get first item."""
        index = EpochIndex(epoch=0, idx=0)
        return self.ds.__getitem__(index)

    def get_last_item(self):
        """Get last item."""
        index = EpochIndex(epoch=0, idx=self.length - 1)
        return self.ds.__getitem__(index)

    def get_middle_item(self):
        """Get middle item."""
        index = EpochIndex(epoch=0, idx=(self.length - 1) // 2)
        return self.ds.__getitem__(index)


if __name__ == "__main__":
    results_dict = {}
    memap_data_path = load("single_cell/testdata-memmap-format") / "cellxgene_2023-12-15_small_mmap" / "train"
    preprocessor = GeneformerPreprocess(
        download_directory=memap_data_path,
        medians_file_path=memap_data_path / "medians.json",
        tokenizer_vocab_path=memap_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
    geneformer_metrics = GeneformerDatasetMetrics(
        memmap_dir=memap_data_path, tokenizer=tokenizer, median_dict=median_dict
    )  # type: ignore
    results_dict[" Create Geneformer Dataset"] = geneformer_metrics.create_from_memmap()[1]
    results_dict["Geneformer Dataset Get Length (s)"] = geneformer_metrics.get_length()[1]
    results_dict["Geneformer Dataset Get First Item (s)"] = geneformer_metrics.get_first_item()[1]
    results_dict["Geneformer Dataset Get Middle Item (s)"] = geneformer_metrics.get_middle_item()[1]
    results_dict["Geneformer Dataset Get Last Item (s)"] = geneformer_metrics.get_last_item()[1]
    df = pd.DataFrame([results_dict])
    df.to_csv("full_runtime.csv")
