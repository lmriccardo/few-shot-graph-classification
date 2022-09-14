from collections import defaultdict
import torch

from utils.utils import configure_logger
from data.dataset import get_dataset, OHGraphDataset
from data.dataloader import get_dataloader
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
    ds = gm()

    dl = get_dataloader(train_ds, 3, 3, 4, 10, True, 1, None, False)
    support, support_list, query, query_list = next(iter(dl))
    print(support)
    print(query)

    # d = defaultdict(int)
    # for l, v in ds.get_graphs_per_label().items():
    #     for gid in v:
    #         d[gid] += 1

    # print(sum([(1 if v > 1 else 0) for _, v in d.items()]))
    

func()