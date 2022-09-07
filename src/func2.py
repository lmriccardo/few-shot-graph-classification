import torch

from utils.utils import configure_logger
from data.dataset import get_dataset, split_dataset
from utils.trainers import KFoldTrainer
from algorithms.asmaml.asmaml1 import AdaptiveStepMAML
from algorithms.mevolve.mevolve import MEvolveGDA

import config
import sys


def m_evolve() -> None:
    config.DEFAULT_DATASET = "TRIANGLES"

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
            number_of_epochs=config.N_CROSSVALIDATION, batch_per_episodes=config.BATCH_PER_EPISODES
        )

    print(configurations, file=sys.stdout if not config.FILE_LOGGING else open(logger.handlers[1].baseFilename, mode="a"))

    # For M-Evolve we need to create a total train + val dataset
    # and then resplit it into train and dataset 8:2
    total_dataset = train_ds + val_ds
    _, m_evolve_val_ds = split_dataset(total_dataset)

    # Initialize the trainer
    trainer = KFoldTrainer(
        train_ds=train_ds, val_ds=val_ds, logger=logger, 
        model_name=config.MODEL_NAME, epochs=config.N_CROSSVALIDATION,
        dataset_name=config.DEFAULT_DATASET
    )
    
    # FIXME: now we have only AS-MAML, but in the future more models will be used
    meta = AdaptiveStepMAML(trainer.model,
                            inner_lr=config.INNER_LR,
                            outer_lr=config.OUTER_LR,
                            stop_lr=config.STOP_LR,
                            weight_decay=config.WEIGHT_DECAY,
                            paper=False
        ).to(config.DEVICE)
    
    trainer._configure_meta(meta)

    me = MEvolveGDA(
        trainer=trainer, n_iters=config.ITERATIONS, 
        logger=logger, train_ds=train_ds, 
        validation_ds=m_evolve_val_ds
    )

    final_classifier = me.evolve()


m_evolve()