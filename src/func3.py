import torch

from utils.utils import configure_logger
from data.dataset import get_dataset, OHGraphDataset
from algorithms.gmixup.gmixup import GMixupGDA

import config

from typing import Union, List, Dict, Any
import torch_geometric.data as pyg_data


def func() -> None:
    torch.set_printoptions(edgeitems=config.EDGELIMIT_PRINT, precision=3)
    logger = configure_logger(file_logging=config.FILE_LOGGING, logging_path=config.LOGGING_PATH)

    dataset_name = "TRIANGLES"
    train_ds, _, _, _ = get_dataset(
        download=config.DOWNLOAD_DATASET,
        data_dir=config.DATA_PATH,
        logger=logger,
        dataset_name=dataset_name
    )

    gm = GMixupGDA(train_ds)
    _ = gm()

func()