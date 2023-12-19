from typing import Sequence, Tuple

import pytorch_lightning as pl
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bionemo.data.mapped_dataset import FilteredMappedDataset
from bionemo.data.memmap_csv_fields_dataset import CSVFieldsMemmapDataset
from bionemo.data.memmap_fasta_fields_dataset import FASTAFieldsMemmapDataset
from bionemo.data.utils import expand_dataset_paths
from bionemo.triton.types_constants import M


__all__: Sequence[str] = ("setup_inference",)


# TODO [mgreaves] uncommment with !553
# def setup_inference(cfg: DictConfig, *, interactive: bool = False) -> Tuple[M, pl.Trainer, DataLoader]:
def setup_inference(cfg: DictConfig) -> Tuple[M, pl.Trainer, DataLoader]:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    infer_class = import_class_by_path(cfg.infer_target)

    # TODO [mgreaves] uncomment with !553
    # infer_model = infer_class(cfg, interactive=interactive)
    infer_model = infer_class(cfg)
    trainer = infer_model.trainer

    logging.info("\n\n************** Restored model configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(infer_model.model.cfg)}')

    if not cfg.model.data.data_impl:
        # try to infer data_impl from the dataset_path file extension
        if cfg.model.data.dataset_path.endswith('.fasta'):
            cfg.model.data.data_impl = 'fasta_fields_mmap'
        else:
            # Data are assumed to be CSV format if no extension provided
            logging.info('File extension not supplied for data, inferring csv.')
            cfg.model.data.data_impl = 'csv_fields_mmap'
        logging.info(f'Inferred data_impl: {cfg.model.data.data_impl}')

    if cfg.model.data.data_impl == "csv_fields_mmap":
        dataset_paths = expand_dataset_paths(cfg.model.data.dataset_path, ext=".csv")
        ds = CSVFieldsMemmapDataset(
            dataset_paths,
            index_mapping_dir=cfg.model.data.index_mapping_dir,
            **cfg.model.data.data_impl_kwargs.get("csv_fields_mmap", {}),
        )
    elif cfg.model.data.data_impl == "fasta_fields_mmap":
        dataset_paths = expand_dataset_paths(cfg.model.data.dataset_path, ext=".fasta")
        ds = FASTAFieldsMemmapDataset(
            dataset_paths,
            index_mapping_dir=cfg.model.data.index_mapping_dir,
            **cfg.model.data.data_impl_kwargs.get("fasta_fields_mmap", {}),
        )
    else:
        raise ValueError(f'Unknown data_impl: {cfg.model.data.data_impl}')

    # remove too long sequences
    filtered_ds = FilteredMappedDataset(
        dataset=ds,
        criterion_fn=lambda x: len(infer_model._tokenize([x["sequence"]])[0]) <= infer_model.model.cfg.seq_length,
    )

    dataloader = DataLoader(
        filtered_ds,
        batch_size=cfg.model.data.batch_size,
        num_workers=cfg.model.data.num_workers,
        drop_last=False,
    )
    return infer_model, trainer, dataloader
