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

import os
import shlex
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.testing.data.load import load


data_path: Path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data"


def test_bionemo2_rootdir():
    data_error_str = (
        "Please download test data with:\n"
        "`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    )
    assert data_path.exists(), f"Could not find test data directory.\n{data_error_str}"
    assert data_path.is_dir(), f"Test data directory is supposed to be a directory.\n{data_error_str}"


@pytest.fixture
def dummy_protein_dataset(tmp_path):
    """Create a mock protein dataset."""
    db_file = tmp_path / "protein_dataset.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE protein (
            id TEXT PRIMARY KEY,
            sequence TEXT
        )
    """
    )

    proteins = [
        ("UniRef90_A", "ACDEFGHIKLMNPQRSTVWY"),
        ("UniRef90_B", "DEFGHIKLMNPQRSTVWYAC"),
        ("UniRef90_C", "MGHIKLMNPQRSTVWYACDE"),
        ("UniRef50_A", "MKTVRQERLKSIVRI"),
        ("UniRef50_B", "MRILERSKEPVSGAQLA"),
    ]
    cursor.executemany("INSERT INTO protein VALUES (?, ?)", proteins)

    conn.commit()
    conn.close()

    return db_file


@pytest.fixture
def dummy_parquet_train_val_inputs(tmp_path):
    """Create a mock protein train and val cluster parquet."""
    train_cluster_path = tmp_path / "train_clusters.parquet"
    train_clusters = pd.DataFrame(
        {
            "ur90_id": [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]],
        }
    )
    train_clusters.to_parquet(train_cluster_path)

    valid_cluster_path = tmp_path / "valid_clusters.parquet"
    valid_clusters = pd.DataFrame(
        {
            "ur50_id": ["UniRef50_A", "UniRef50_B", "UniRef50_A", "UniRef50_B"],  # 2 IDs more than confest
        }
    )
    valid_clusters.to_parquet(valid_cluster_path)
    return train_cluster_path, valid_cluster_path


def test_pretrain_pydantic_cli(dummy_protein_dataset, dummy_parquet_train_val_inputs, tmpdir):
    # result_dir = Path(tmpdir.mkdir("results"))
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs
    # result_dir = Path("/tmp/results").mkdir(exist_ok=True)
    result_dir = tmpdir.mkdir("results")

    open_port = find_free_network_port()
    config = "test_config.json"

    # Invoke with blocking
    cmd_str = f"""bionemo-esm2-recipe --dest {config} --recipe test-tiny
      --train-database-path {dummy_protein_dataset}
      --train-cluster-path {train_cluster_path}
      --valid-database-path {dummy_protein_dataset}
      --valid-cluster-path {valid_cluster_path}
      --result-dir {result_dir}""".strip()
    print(cmd_str)
    # continue when finished
    env = dict(**os.environ)  # a local copy of the environment
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    # Now do pretrain
    if result.returncode != 0:
        raise Exception(f"Pretrain script failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")

    cmd_str = f"""bionemo-esm2-train --conf {config}""".strip()
    env = dict(**os.environ)  # a local copy of the environment
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Exception(f"Pretrain script failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")
    # NOTE this looks a lot like a magic value. But we also could do json.loads(config)['experiment_config']['experiment_name']
    assert (result_dir / "default_experiment").exists(), "Could not find test experiment directory."
