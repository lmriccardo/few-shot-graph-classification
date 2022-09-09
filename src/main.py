import sys
import os

sys.path.append(os.getcwd())

from data.dataset import get_dataset, split_dataset
from utils.utils import configure_logger
from utils.trainers import ASMAMLTrainer, KFoldTrainer
from utils.testers import ASMAMLTester
# from algorithms.mevolve.mevolve import MEvolveGDA
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

    optimizer = ASMAMLTrainer(train_ds, val_ds, logger, 
                              epochs=config.EPOCHS, 
                              dataset_name=dataset_name,
                              model_name="sage"
                )
            
    optimizer.train()

    # best_model_path = os.path.join(config.MODELS_SAVE_PATH, f"{dataset_name}_BestModel.pth")
    # tester = Tester(test_ds, logger, best_model_path)
    # tester.test()

    # delete_data_folder(data_dir)


# def m_evolve() -> None:
#     config.DEFAULT_DATASET = "TRIANGLES"

#     torch.set_printoptions(edgeitems=config.EDGELIMIT_PRINT)
#     logger = configure_logger(file_logging=config.FILE_LOGGING, logging_path=config.LOGGING_PATH)

#     dataset_name = config.DEFAULT_DATASET
#     train_ds, test_ds, val_ds, _ = get_dataset(
#         download=config.DOWNLOAD_DATASET, 
#         data_dir=config.DATA_PATH, 
#         logger=logger,
#         dataset_name=dataset_name
#     )

#     logger.debug("--- Datasets ---")
#     print("\n- Train: ", train_ds)
#     print("- Test : ", test_ds)
#     print("- Validation: ", val_ds)
#     print()

#     logger.debug("--- Configurations ---")

#     configurations = ("\nDEVICE: {device}\n"                            +
#                       "DATASET NAME: {dataset_name}\n"                + 
#                       "TRAIN SUPPORT SIZE: {train_support_size}\n"    +
#                       "TRAIN QUERY SIZE: {train_query_size}\n"        +
#                       "VALIDATION SUPPORT SIZE: {val_support_size}\n" +
#                       "VALIDATION QUERY SIZE: {val_query_size}\n"     +
#                       "TEST SUPPORT SIZE: {test_support_size}\n"      +
#                       "TEST QUERY SIZE: {test_query_size}\n"          +
#                       "TRAIN EPISODE: {train_episode}\n"              +
#                       "VALIDATION EPISODE: {val_episode}\n"           +
#                       "NUMBER OF EPOCHS: {number_of_epochs}\n"        +
#                       "BATCH PER EPISODES: {batch_per_episodes}\n"
#         ).format(
#             device=config.DEVICE, dataset_name=dataset_name,
#             train_support_size=f"{config.TRAIN_WAY} x {config.TRAIN_SHOT}",
#             train_query_size=f"{config.TRAIN_WAY} x {config.TRAIN_QUERY}",
#             val_support_size=f"{config.TEST_WAY} x {config.VAL_SHOT}",
#             val_query_size=f"{config.TEST_WAY} x {config.VAL_QUERY}",
#             test_support_size=f"{config.TEST_WAY} x {config.VAL_SHOT}",
#             test_query_size=f"{config.TEST_WAY} x {config.VAL_QUERY}",
#             train_episode=config.TRAIN_EPISODE, val_episode=config.VAL_EPISODE,
#             number_of_epochs=config.N_CROSSVALIDATION, batch_per_episodes=config.BATCH_PER_EPISODES
#         )

#     print(configurations, file=sys.stdout if not config.FILE_LOGGING else open(logger.handlers[1].baseFilename, mode="a"))

#     # For M-Evolve we need to create a total train + val dataset
#     # and then resplit it into train and dataset 8:2
#     total_dataset = train_ds + val_ds
#     _, m_evolve_val_ds = split_dataset(total_dataset)

#     # Initialize the trainer
#     trainer = KFoldTrainer(
#         train_ds=train_ds, val_ds=val_ds, logger=logger, 
#         model_name=config.MODEL_NAME, epochs=config.N_CROSSVALIDATION,
#         dataset_name=config.DEFAULT_DATASET
#     )
    
#     # FIXME: now we have only AS-MAML, but in the future more models will be used
#     meta = AdaptiveStepMAML(trainer.model,
#                             inner_lr=config.INNER_LR,
#                             outer_lr=config.OUTER_LR,
#                             stop_lr=config.STOP_LR,
#                             weight_decay=config.WEIGHT_DECAY,
#                             paper=False
#         ).to(config.DEVICE)
    
#     trainer._configure_meta(meta)

#     me = MEvolveGDA(
#         trainer=trainer, n_iters=config.ITERATIONS, 
#         logger=logger, train_ds=train_ds, 
#         validation_ds=m_evolve_val_ds
#     )

#     final_classifier = me.evolve()

#     # TODO: Implement a Tester


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


def func() -> None:
    from data.dataloader import GraphDataLoader
    from utils.utils import rename_edge_indexes, data_batch_collate, setup_seed, plot_graph, data2graph
    from torch_geometric.utils import add_remaining_self_loops
    from torch_geometric.data import Batch
    from utils.utils import load_with_pickle

    setup_seed(42)
    torch.set_printoptions(edgeitems=config.EDGELIMIT_PRINT)

    logger = configure_logger(file_logging=config.FILE_LOGGING, logging_path=config.LOGGING_PATH)

    dataset_name = config.DEFAULT_DATASET
    train_ds, _, val_ds, _ = get_dataset(
        download=config.DOWNLOAD_DATASET, 
        data_dir=config.DATA_PATH, 
        logger=logger,
        dataset_name=dataset_name
    )

    total_dataset = train_ds + val_ds
    train_ds, val_ds = split_dataset(total_dataset)

    # gdl = GraphDataLoader(val_ds, batch_size=3, shuffle=True)
    # sample = next(iter(gdl))
    # new_data = rename_edge_indexes(sample.to_data_list())
    # new_sample, _ = data_batch_collate(new_data)
    # batch_sample = Batch.from_data_list(new_data)

    # for d in new_data:
    #     print(d.edge_index)

    # print(new_sample)
    # print(sample)
    # print(sample.y)

    # print(batch_sample)
    # print(batch_sample.y)
    # print(batch_sample.edge_index)
    # print(new_sample.edge_index)

    # print(torch.hstack((batch_sample.edge_index[0], batch_sample.edge_index[1])).unique())
    # print(torch.hstack((new_sample.edge_index[0], new_sample.edge_index[1])).unique())

    # print(add_remaining_self_loops(batch_sample.edge_index)[0].shape)
    # print(add_remaining_self_loops(new_sample.edge_index)[0].shape)

    # print("equals for X: ", torch.all(new_sample.x == sample.x))
    # print(new_sample.y, sample.y)

    # print(torch.all(new_sample.batch == sample.batch))

    # plot_graph(data2graph(new_sample), "New Sample")

    dataset = paper.GraphDataset(val=False)
    train_loader = paper.FewShotDataLoaderPaper(
        dataset=dataset,
        n_way=3,
        k_shot=1,
        n_query=1,
        batch_size=1,
        num_workers=1,
        epoch_size=200
    )

    print("============ PAPER ==============")

    paper_sample, _ = next(iter(train_loader(42)))
    sample_nodes, sample_edge_index, sample_graph_indicator, sample_label = paper_sample
    sample_edge_index = sample_edge_index[0]
    print(sample_edge_index.shape)
    print(sample_edge_index)
    print(sample_edge_index.transpose(0,1))

    print("========== NOT PAPER ============")
    index = [1093, 1434, 2108]
    data_list = []
    for i in index:
        data_list.append(total_dataset.get(i))

    d,_ = data_batch_collate(rename_edge_indexes(data_list))

    print(d.edge_index)
    print(d.edge_index.transpose(0,1))

    print(load_with_pickle("../data/COIL-DEL/COIL-DEL_train_set.pickle")["graph2edges"][1093])
    print(load_with_pickle("../data/COIL-DEL/COIL-DEL_train_set.pickle")["graph2edges"][1434])
    print(load_with_pickle("../data/COIL-DEL/COIL-DEL_train_set.pickle")["graph2edges"][2108])


if __name__ == "__main__":
    # asmaml()
    # m_evolve()
    run_paper()
    # func()
    # func2()