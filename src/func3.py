import torch

from utils.utils import to_datalist, configure_logger
from data.dataset import get_dataset
from data.dataloader import GraphDataLoader

import config


def func() -> None:
    torch.set_printoptions(edgeitems=config.EDGELIMIT_PRINT)
    logger = configure_logger(file_logging=config.FILE_LOGGING, logging_path=config.LOGGING_PATH)

    dataset_name = config.DEFAULT_DATASET
    train_ds, _, _, _ = get_dataset(
        download=config.DOWNLOAD_DATASET, 
        data_dir=config.DATA_PATH, 
        logger=logger,
        dataset_name=dataset_name
    )

    dl = GraphDataLoader(dataset=train_ds, batch_size=10)
    sample, sample_list = next(iter(dl))
    print(sample)
    print(sample_list)

    print(to_datalist(sample))


func()