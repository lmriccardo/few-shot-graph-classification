import torch
import sys
import os
sys.path.append(os.getcwd())

from models.gcn4maml import GCN4MAML
from algorithms.mevolve.mevolve import MEvolve
from data.dataset import get_dataset, \
    random_mapping_heuristic, \
    motif_similarity_mapping_heuristic, \
    split_dataset
from data.dataloader import get_dataloader
from utils.utils import configure_logger, setup_seed

import config


def test():
    setup_seed(432)
    loss = torch.nn.CrossEntropyLoss()
    logger = configure_logger(dataset_name=config.DEFAULT_DATASET)
    train_ds, val_ds, _, _ = get_dataset(logger, dataset_name=config.DEFAULT_DATASET, data_dir=config.DATA_PATH)
    val_loader = get_dataloader(val_ds, n_way=config.TEST_WAY,
                                  k_shot=config.VAL_SHOT, n_query=config.VAL_QUERY,
                                  epoch_size=config.VAL_EPISODE, batch_size=1, shuffle=True)

    print("Validation Set: ", val_ds)
    print("Training Set: ", train_ds)

    print("Train + Validation: ", train_ds + val_ds)
    
    support, support_list, query, query_list = next(iter(val_loader))
    print("Support Data: ", support)
    print("Query Data: ", query)

    net = GCN4MAML(num_features=config.NUM_FEATURES[config.DEFAULT_DATASET], num_classes=config.TRAIN_WAY)
    pred, _, _ = net(support.x, support.edge_index, support.batch)
    pred_idx = torch.nn.functional.softmax(pred, dim=1).argmax(dim=1)
    loss_values = loss(pred, support.y)

    print("Predictions: ", pred)
    print("Prediction Size: ", pred.shape)
    print("Prediction index: ", pred_idx)
    print("Losses: ", loss_values)

    heuristics = {
        "random_mapping" : random_mapping_heuristic,
        "motif_similarity_mapping" : motif_similarity_mapping_heuristic
    }

    chosen_heuristic = heuristics[config.HEURISTIC]
    augmented_data = chosen_heuristic(train_ds)

    print("Used Heuristic: ", chosen_heuristic.__name__)
    print("Lenght Augmented Data: ", len(augmented_data))
    
    # TODO: input classes should be the classes of the training set.
    prob_vect = dict(zip(range(pred.shape[0]), pred))
    filtered_data = MEvolve.data_filtering(
        val_ds, prob_vect, support_list, 
        train_ds.targets(), 
        net, logger, augmented_data
    )

    print(filtered_data)


def test2():
    setup_seed(432)
    loss = torch.nn.CrossEntropyLoss()
    logger = configure_logger(dataset_name=config.DEFAULT_DATASET)
    train_ds, val_ds, _, _ = get_dataset(logger, dataset_name=config.DEFAULT_DATASET, data_dir=config.DATA_PATH)
                                  
    print("Validation Set: ", val_ds)
    print("Training Set: ", train_ds)

    total = train_ds + val_ds
    new_train_ds, new_val_ds = split_dataset(total)
    print("New Training Set: ", new_train_ds)
    print("New Validation Set: ", new_val_ds)

    print(total.number_of_classes())
    net = GCN4MAML(num_features=config.NUM_FEATURES[config.DEFAULT_DATASET], num_classes=total.number_of_classes())
    total_data, total_data_list = new_val_ds.to_data()
    # print(total_data)

    print(total_data_list[0].edge_index)
    # rename_edge_indexes(total_data_list)
    # print(total_data.y)
    # print(total_data.old_classes_mapping)
    # print([x.y.item() for x in ])

    # pred, _, _ = net(total_data.x, total_data.edge_index, total_data.batch)
    # pred_idx = torch.nn.functional.softmax(pred, dim=1).argmax(dim=1)
    # loss_values = loss(pred, total_data.y)

    # print("Predictions: ", pred)
    # print("Prediction Size: ", pred.shape)
    # print("Prediction index: ", pred_idx)
    # print("Losses: ", loss_values)

    # heuristics = {
    #     "random_mapping" : random_mapping_heuristic,
    #     "motif_similarity_mapping" : motif_similarity_mapping_heuristic
    # }

    # chosen_heuristic = heuristics[config.HEURISTIC]
    # augmented_data = chosen_heuristic(new_train_ds)

    # print("Used Heuristic: ", chosen_heuristic.__name__)
    # print("Lenght Augmented Data: ", len(augmented_data))

    # prob_vect = dict(zip(range(pred.shape[0]), pred))
    # filtered_data = data_filtering(
    #     new_val_ds, prob_vect, total_data_list, 
    #     new_train_ds.targets().tolist(),
    #     net, logger, augmented_data
    # )

    # print("Final Lenght Filtered Data: ", len(filtered_data))


def test3():
    setup_seed(432)
    logger = configure_logger(dataset_name=config.DEFAULT_DATASET)
    train_ds, val_ds, _, _ = get_dataset(logger, dataset_name=config.DEFAULT_DATASET, data_dir=config.DATA_PATH)

    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=5, shuffle=True)
    for _, (a, b) in enumerate(kfold.split(train_ds)):
        a_dl = get_dataloader(train_ds, n_way=config.TRAIN_WAY, k_shot=config.TRAIN_SHOT, n_query=config.TRAIN_QUERY,
                              epoch_size=config.TRAIN_EPISODE, shuffle=True, batch_size=1, exclude_keys=[str(x) for x in b])
        
        print(len(a_dl))

# test()
# test2()
test3()