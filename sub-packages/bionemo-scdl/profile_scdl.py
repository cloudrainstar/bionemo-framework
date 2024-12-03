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

import random
import subprocess
import sys
import time
from enum import Enum
from functools import wraps
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch

class FileNames(str, Enum):
    """Names of files that are generated in SingleCellCollection."""

    DATA = "data.npy"
    COLPTR = "col_ptr.npy"
    ROWPTR = "row_ptr.npy"
    METADATA = "metadata.json"
    DTYPE = "dtypes.json"
    FEATURES = "features"
    VERSION = "version.json"


def get_disk_size(directory):
    """Size of directory on disk."""
    result = subprocess.run(["du", "-sb", directory], stdout=subprocess.PIPE, text=True)
    size_in_bytes = int(result.stdout.split()[0])
    return size_in_bytes

class AnnDataset(Dataset):
    """Ann Data Dataset."""

    def __init__(self, anndata_obj: ad.AnnData):
        """Custom Dataset for AnnData objects compatible with PyTorch's DataLoader.

        Args:
            anndata_obj (ad.AnnData): The AnnData object to load data from.
        """
        self.anndata_obj = anndata_obj

    def __len__(self):
        """Returns the total number of samples."""
        return self.anndata_obj.shape[0]

    def _get_row(self, idx, ret_features=False, select_feat=None):
        """Returns a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features (X) and, if available, the label (y).
        """
        # Extract data for the given index
        row = self.anndata_obj.X[idx]
        if not ret_features:
            return (row.indices, row.data), None
        else: 
            return (row.indices, row.data), self.anndata_obj.var[select_feat]
    
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
class AnnDataMetrics:
    """AnnData Metrics."""

    def __init__(self, adatapath):
        """Instantiate class."""
        self.adatapath = adatapath

    def load(self):
        """Create from anndataset."""
        self.ad = ad.read_h5ad(self.adatapath)
    
    def _instantiate_dataset(self): 
        self.ann_data = AnnDataset(self.ad)
    def load_backed(self):
        """Create from anndataset."""
        self.ad_backed = ad.read_h5ad(self.adatapath, backed="r")

    def size_disk_bytes(self):
        """Size of anndata on disk."""
        return get_disk_size(self.adatapath)
    
    def get_all_rows(self, num_rows): 
        """Get all rows in a dataset"""
        for i in range(num_rows):
           _, _ = self.ann_data._get_row(i)
        return 0
    
    def get_all_rows_with_features(self, num_rows): 
        """Get all rows in dataset with features"""
        for i in range(num_rows):
           _, _ = self.ann_data._get_row(i, ret_features=True, select_feat="feature_id")
        return 0
    
    def get_all_rows_and_modify_val(self, num_rows): 
        """Get all rows in dataset and modify the return value"""
        for i in range(num_rows):
           vals, _ = self.ann_data._get_row(i, ret_features=True, select_feat="feature_id")
           new_val = vals[0]+ 1 # type: ignore
           if i == 0:
               print("Type", type(vals[0]))
        return 0





@time_all_methods
class SCDLMetrics:
    """SCDL Metrics."""

    def __init__(self, adatapath, memmap_dir):
        """Instantiate class."""
        self.adatapath = adatapath
        self.memmap_dir = memmap_dir

    def create_from_adata(self):
        """Create from anndataset."""
        self.ds = SingleCellMemMapDataset(
            self.memmap_dir,
            self.adatapath,
        )

    def save(self):
        """Save."""
        self.first_ds.save()
        del self.first_ds

    def load_backed(self):
        """Load Scdl from disk."""
        self.ds = SingleCellMemMapDataset(self.memmap_dir)

    def num_values(self):
        """Number of values."""
        return self.ds.number_of_values()

    def sparsity_stats(self):
        """Sparsity of dataset."""
        return self.ds.sparsity()

    def size_disk_bytes(self):
        """Size of scdl on disk."""
        return get_disk_size(self.memmap_dir)

    def anndata_size_disk_bytes(self):
        """Size of anndata on disk."""
        return get_disk_size(self.adatapath)

    def size_mem_dataset_bytes(self):
        """Size of dataset in memory."""
        return sys.getsizeof(self.ds)
    
    def get_all_rows(self, num_rows): 
        """Get all rows in a dataset"""
        for i in range(num_rows):
           _, _ = self.ds.get_row(i)
        return 0
    
    def get_all_rows_with_features(self, num_rows): 
        """Get all rows in dataset with features"""
        for i in range(num_rows):
            self.ds.get_row(i, return_features=True, feature_vars=["feature_id"])
        return 0
    
    def get_all_rows_and_modify_val(self, num_rows): 
        """Get all rows in dataset and modify the return value"""
        for i in range(num_rows):
           vals, _ = self.ds.get_row(i)
           new_val = vals[0]+ 1 # type: ignore
           if i == 0:
               print("Type", type(vals[0]))
           
        return 0
    

if __name__ == "__main__": 
    print ("in script")
    anndatapath = "/workspace/bionemo2/sub-packages/data/merged_adata.h5ad" 
    anndata_m = AnnDataMetrics(anndatapath)
    dicts = []
    # path = Path("/workspace/bionemo2/cellxgene_2023-12-15_small_mmap/train")
    path = Path("/workspace/bionemo2/sub-packages/data/newest_cellxsmall_train")
    
    # large_path = Path("/workspace/bionemo2/sub-packages/data/scdl_memmap_profiling")

    scdl_m = SCDLMetrics(anndatapath, path)
    results_dict = {}
    # results_dict["AnnData Dataset Size on Disk (MB)"] = scdl_m.anndata_size_disk_bytes()[0] / (1_024**2)

    # results_dict["SCDL Dataset Load Time (s)"] = scdl_m.create_from_adata()[1]
    results_dict["Ann Dataset Load Time (s)"] = anndata_m.load()
    results_dict["Ann Dataset Instantiate Dataset Time (s)"] = anndata_m._instantiate_dataset()
    results_dict["Ann Data Time to get all rows"] = anndata_m.get_all_rows(len(anndata_m.ann_data))[1]
    results_dict["Ann Data Time to get all rows and modify value"] = anndata_m.get_all_rows_and_modify_val(len(anndata_m.ann_data))[1]
    results_dict["Ann Data Time to get all rows with features"] = anndata_m.get_all_rows_with_features(len(anndata_m.ann_data))[1]


    results_dict["SCDL Dataset Load Time (s)"] = scdl_m.load_backed()[1]
    results_dict["SCDL Time to get all rows"] = scdl_m.get_all_rows(len(scdl_m.ds))[1]
    results_dict["SCDL Time to get all rows and modify values "] = scdl_m.get_all_rows_and_modify_val(len(scdl_m.ds))[1]
    results_dict["SCDL Time to get all rows with features "] = scdl_m.get_all_rows_with_features(len(scdl_m.ds))[1]


    results_dict["SCDL Dataset Size on Disk (MB)"] = scdl_m.size_disk_bytes()[0] / (1_024**2)
    dicts.append(results_dict)
    combined = {key: [d[key] for d in dicts] for key in dicts[0]}
    df = pd.DataFrame(combined)
    print(df)
    df.to_csv("all_runtime.csv", index=False)