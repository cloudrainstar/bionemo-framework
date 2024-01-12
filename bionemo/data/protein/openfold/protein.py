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

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser

import bionemo.data.protein.openfold.residue_constants as rc


# Complete sequence of chain IDs supported by the PDB format.
_SUPPORTED_PDB_CHAIN_ID_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_MAX_NUM_SUPPORTED_PDB_CHAINS = len(_SUPPORTED_PDB_CHAIN_ID_CHARACTERS)  # := 62.


@dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # ATOM_TYPES from residue_constants, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and 20,
    # where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom.
    # 1.0 if an atom is present and 0.0 if not.
    # This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > _MAX_NUM_SUPPORTED_PDB_CHAINS:
            raise ValueError(
                "Cannot build an instance with more than "
                f"{_MAX_NUM_SUPPORTED_PDB_CHAINS} chains "
                "because these cannot be written to PDB format."
            )

    @classmethod
    def from_prediction(
        cls,
        aatype: np.ndarray,
        final_atom_positions: np.ndarray,
        final_atom_mask: np.ndarray,
        residue_index: np.ndarray,
        b_factors: Optional[np.ndarray] = None,
        chain_index: Optional[np.ndarray] = None,
    ) -> Protein:
        if b_factors is None:
            b_factors = np.zeros_like(final_atom_mask)

        if chain_index is None:
            chain_index = np.zeros_like(aatype)

        return cls(
            aatype=aatype,
            atom_positions=final_atom_positions,
            atom_mask=final_atom_mask,
            residue_index=residue_index + 1,
            b_factors=b_factors,
            chain_index=chain_index,
        )

    @classmethod
    def from_pdb_string(cls, pdb_str: str, chain_id: Optional[str] = None) -> Protein:
        return _protein_from_pdb_string(pdb_str=pdb_str, chain_id=chain_id)

    def to_pdb_string(self) -> str:
        return _protein_to_pdb(self)

    @property
    def sequence(self) -> str:
        return "".join([rc.RESTYPES_WITH_X[aatype_int] for aatype_int in self.aatype])


def _protein_from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK.
        All non-standard atoms will be ignored.

    Args:
        pdb_str: The contents of the pdb file.
        chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
            Otherwise all chains are parsed.

    Returns:
        A new `Protein` parsed from the pdb contents.

    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Only single model PDBs are supported. Found {len(models)} models.")
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = rc.RESTYPE_3TO1.get(res.resname, "X")
            restype_idx = rc.RESTYPE_ORDER.get(res_shortname, rc.RESTYPE_NUM)
            pos = np.zeros((rc.ATOM_TYPE_NUM, 3))
            mask = np.zeros((rc.ATOM_TYPE_NUM,))
            res_b_factors = np.zeros((rc.ATOM_TYPE_NUM,))
            for atom in res:
                if atom.name not in rc.ATOM_TYPES:
                    continue
                pos[rc.ATOM_ORDER[atom.name]] = atom.coord
                mask[rc.ATOM_ORDER[atom.name]] = 1.0
                res_b_factors[rc.ATOM_ORDER[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    return f"{'TER':<6}{atom_index:>5}      {end_resname:>3} " f"{chain_name:>1}{residue_index:>4}"


def _protein_to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
        prot: The protein to convert to PDB.

    Returns:
        PDB string.

    """
    restypes = rc.RESTYPES + ["X"]

    def res_1to3(r):
        return rc.RESTYPE_1TO3.get(restypes[r], "UNK")

    atom_types = rc.ATOM_TYPES

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > rc.RESTYPE_NUM):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= _MAX_NUM_SUPPORTED_PDB_CHAINS:
            raise ValueError("The PDB format supports at most " f"{_MAX_NUM_SUPPORTED_PDB_CHAINS} chains.")
    chain_ids[i] = _SUPPORTED_PDB_CHAIN_ID_CHARACTERS[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                ),
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        ),
    )
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def _ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
        prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
        An ideal atom mask.

    """
    return rc.STANDARD_ATOM_MASK[prot.aatype]
