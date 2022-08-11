import sys
import os
sys.path.append(os.getcwd())

from data.dataset import generate_train_val_test
from data.dataloader import FewShotDataLoader
from data.sampler import TaskBatchSampler
from utils.utils import delete_data_folder

import config


def main():
    train_ds, test_ds, val_ds = generate_train_val_test(download_data=True, perc_train=50, perc_test=30)
    print("--- Datasets ---")
    print("- Train: ", train_ds)
    print("- Test : ", test_ds)
    print("- Validation: ", val_ds)

    print("--- Creating the DataLoader for Training ---")
    graph_train_loader = FewShotDataLoader(
        dataset=train_ds,
        batch_sampler=TaskBatchSampler(
            dataset_targets=train_ds.targets(),
            n_way=config.TRAIN_WAY,
            k_shot=config.TRAIN_SHOT,
            n_query=config.TRAIN_QUERY,
            epoch_size=config.TRAIN_EPISODE,
            shuffle=True,
            batch_size=config.BATCH_PER_EPISODES
        )
    )

    print("--- Getting the First Sample ---")
    support, query = next(iter(graph_train_loader))
    print("- Support Sample Batch: ", support)
    print("- Query Sample Batch: ", query)

    delete_data_folder()


if __name__ == "__main__":
    main()