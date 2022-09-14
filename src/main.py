import sys
import os

sys.path.append(os.getcwd())

from data.dataset import get_dataset
from utils.utils import configure_logger
from utils.trainer1 import Trainer
from algorithms.asmaml.asmaml import AdaptiveStepMAML

import paper
import config
import torch


def asmaml():
    torch.set_printoptions(edgeitems=config.EDGELIMIT_PRINT)
    logger = configure_logger(file_logging=config.FILE_LOGGING, logging_path=config.LOGGING_PATH)

    dataset_name = config.DEFAULT_DATASET
    train_ds, test_ds, val_ds, _ = get_dataset(
        download=config.DOWNLOAD_DATASET, 
        data_dir=config.DATA_PATH, 
        logger=logger,
        dataset_name=dataset_name
    )

    logger.debug("--- Datasets ---")
    print("\n- Train: ", train_ds, file=sys.stdout if not config.FILE_LOGGING else open(logger.handlers[1].baseFilename, mode="a"))
    print("- Test : ", test_ds, file=sys.stdout if not config.FILE_LOGGING else open(logger.handlers[1].baseFilename, mode="a"))
    print("- Validation: ", val_ds, file=sys.stdout if not config.FILE_LOGGING else open(logger.handlers[1].baseFilename, mode="a"))
    print("\n", file=sys.stdout if not config.FILE_LOGGING else open(logger.handlers[1].baseFilename, mode="a"))

    print("\n- Train: ", train_ds)
    print("- Test : ", test_ds)
    print("- Validation: ", val_ds)
    print()

    logger.debug("--- Configurations ---")

    configurations = ("\nDEVICE: {device}\n"                            +
                      "DATASET NAME: {dataset_name}\n"                + 
                      "TRAIN SUPPORT SIZE: {train_support_size}\n"    +
                      "TRAIN QUERY SIZE: {train_query_size}\n"        +
                      "VALIDATION SUPPORT SIZE: {val_support_size}\n" +
                      "VALIDATION QUERY SIZE: {val_query_size}\n"     +
                      "TEST SUPPORT SIZE: {test_support_size}\n"      +
                      "TEST QUERY SIZE: {test_query_size}\n"          +
                      "TRAIN EPISODE: {train_episode}\n"              +
                      "VALIDATION EPISODE: {val_episode}\n"           +
                      "NUMBER OF EPOCHS: {number_of_epochs}\n"        +
                      "BATCH PER EPISODES: {batch_per_episodes}\n"
        ).format(
            device=config.DEVICE, dataset_name=dataset_name,
            train_support_size=f"{config.TRAIN_WAY} x {config.TRAIN_SHOT}",
            train_query_size=f"{config.TRAIN_WAY} x {config.TRAIN_QUERY}",
            val_support_size=f"{config.TEST_WAY} x {config.VAL_SHOT}",
            val_query_size=f"{config.TEST_WAY} x {config.VAL_QUERY}",
            test_support_size=f"{config.TEST_WAY} x {config.VAL_SHOT}",
            test_query_size=f"{config.TEST_WAY} x {config.VAL_QUERY}",
            train_episode=config.TRAIN_EPISODE, val_episode=config.VAL_EPISODE,
            number_of_epochs=config.EPOCHS, batch_per_episodes=config.BATCH_PER_EPISODES
        )

    print(configurations, file=sys.stdout if not config.FILE_LOGGING else open(logger.handlers[1].baseFilename, mode="a"))

    # optimizer = ASMAMLTrainer(train_ds, val_ds, logger, 
    #                           epochs=config.EPOCHS, 
    #                           dataset_name=dataset_name,
    #                           model_name="sage"
    #             )
            
    # optimizer.train()

    optimizer = Trainer(
        train_ds=train_ds, val_ds=val_ds, logger=logger,
        model_name="sage", paper=False, epochs=config.EPOCHS, batch_size=1,
        dataset_name=config.DEFAULT_DATASET, meta_model=AdaptiveStepMAML,
        save_suffix="ASMAML_", use_mevolve=False, use_flag=False, use_gmixup=False
    )

    optimizer.run()

    # best_model_path = os.path.join(config.MODELS_SAVE_PATH, f"{dataset_name}_BestModel.pth")
    # tester = Tester(test_ds, logger, best_model_path)
    # tester.test()

    # delete_data_folder(data_dir)


def run_paper() -> None:
    logger = configure_logger(file_logging=config.FILE_LOGGING, logging_path=config.LOGGING_PATH)
    dataset = paper.GraphDataset(val=False)
    val_dataset = paper.GraphDataset(val=True)
    train_loader = paper.FewShotDataLoaderPaper(
        dataset=dataset,
        n_way=3,
        k_shot=10,
        n_query=15,
        batch_size=1,
        num_workers=4,
        epoch_size=200
    )

    val_loader = paper.FewShotDataLoaderPaper(
        dataset=val_dataset,
        n_way=3,
        k_shot=10,
        n_query=15,
        batch_size=1,
        num_workers=4,
        epoch_size=200
    )

    optimizer = ASMAMLTrainer(dataset, val_dataset, logger, epochs=config.EPOCHS, dataset_name=config.DEFAULT_DATASET, paper=True)
    optimizer.train_dl = train_loader(0)
    optimizer.val_dl   = val_loader(0)

    optimizer.train()


if __name__ == "__main__":
    asmaml()
    # run_paper()