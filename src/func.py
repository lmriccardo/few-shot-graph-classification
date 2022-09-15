import torch

from data.dataset import get_dataset
from data.dataloader import get_dataloader, FewShotDataLoader
from paper import get_dataloader as get_dataloader_paper
from paper import get_dataset as get_dataset_paper
from utils.utils import configure_logger, setup_seed

import config
import sys
import random


setup_seed(0)


def main() -> None:
    torch.set_printoptions(edgeitems=config.EDGELIMIT_PRINT)
    logger = configure_logger(file_logging=config.FILE_LOGGING, logging_path=config.LOGGING_PATH)

    dataset_name = config.DEFAULT_DATASET
    train_ds, _, _, _ = get_dataset(
        download=config.DOWNLOAD_DATASET, 
        data_dir=config.DATA_PATH, 
        logger=logger,
        dataset_name=dataset_name
    )

    train_dl = get_dataloader(train_ds, 3, 10, 15, 200, True, 1, None, False, FewShotDataLoader)
    
    # 

    print("First sample from FewShotDataLoader")
    sample, _, _, _ = next(iter(train_dl))
    mapping = {v : k for k, v in sample.old_classes_mapping.items()}
    
    print(f"Sampled classes: {[mapping[x.item()] for x in sample.y]}")
    print(f"Dataset Classes: {list(train_ds.count_per_class.keys())}")

    print(random.sample(list(train_ds.count_per_class.keys()), k=3))

def p() -> None:

    paper_train_ds = get_dataset_paper(val=False)
    paper_train_dl = get_dataloader_paper(
        paper_train_ds, 3, 10, 15, 200, 1
    )

    # print(random.sample(list(paper_train_ds.label2graphs.keys()), k=3))

    print("First sample from FewShotDataLoaderPaper")
    _, _, classes = next(iter(paper_train_dl(0)))

    print(f"Sampled classes: {classes}")
    # print(f"Dataset Classes: {list(paper_train_ds.label2graphs.keys())}")


if __name__ == "__main__":
    if sys.argv[-1] == "1":
        main()
    else:
        p()