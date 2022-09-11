import torch

from utils.utils import configure_logger
from data.dataset import get_dataset, OHGraphDataset
from algorithms.gmixup.gmixup import GMixupGDA

import config

from typing import Union, List, Dict, Any
import torch_geometric.data as pyg_data


def func() -> None:
    torch.set_printoptions(edgeitems=config.EDGELIMIT_PRINT)
    logger = configure_logger(file_logging=config.FILE_LOGGING, logging_path=config.LOGGING_PATH)

    dataset_name = "TRIANGLES"
    train_ds, _, _, _ = get_dataset(
        download=config.DOWNLOAD_DATASET, 
        data_dir=config.DATA_PATH, 
        logger=logger,
        dataset_name=dataset_name
    )

    gm = GMixupGDA(train_ds)
    new_ds = gm()

    # oh_train_ds = OHGraphDataset(train_ds)
    # print(oh_train_ds[0])
    print(new_ds)
    first_item_key = min(new_ds.graph_ds.keys())
    print(new_ds[first_item_key][0].shape)

func()