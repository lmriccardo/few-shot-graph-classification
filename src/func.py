import torch
import sys
import os
sys.path.append(os.getcwd())

from models.asmaml.gcn4maml import GCN4MAML
from models.utils import data_filtering
from data.dataset import get_dataset, random_mapping_heuristic, motif_similarity_mapping_heuristic
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
    filtered_data = data_filtering(
        val_ds, prob_vect, support_list, 
        list(support.old_classes_mapping.keys()), 
        net, logger, augmented_data
    )

    print(filtered_data)

test()