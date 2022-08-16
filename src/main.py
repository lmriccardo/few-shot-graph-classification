import sys
import os
sys.path.append(os.getcwd())

from data.dataset import generate_train_val_test
from data.dataloader import FewShotDataLoader
from data.sampler import TaskBatchSampler
from utils.utils import delete_data_folder, setup_seed, get_batch_number

import config
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    setup_seed()

    train_ds, test_ds, val_ds = generate_train_val_test(
        download_data=False, 
        perc_train=50, 
        perc_test=30
    )

    logging.debug("--- Datasets ---")
    print("\n- Train: ", train_ds)
    print("- Test : ", test_ds)
    print("- Validation: ", val_ds)
    print()

    logging.debug("--- Creating the DataLoader for Training ---")
    graph_train_loader = FewShotDataLoader(
        dataset=train_ds,
        batch_sampler=TaskBatchSampler(
            dataset_targets=train_ds.targets(),
            n_way=config.TRAIN_WAY,
            k_shot=config.TRAIN_SHOT,
            n_query=config.TRAIN_QUERY,
            epoch_size=config.TRAIN_EPISODE,
            shuffle=True,
            batch_size= 1# config.BATCH_PER_EPISODES
        )
    )

    logging.debug("--- Getting the First Sample ---")
    support, query = next(iter(graph_train_loader))
    print("\n- Support Sample Batch: ", support)
    print("- Query Sample Batch: ", query)
    print()

    # delete_data_folder()


if __name__ == "__main__":
    main()