import torch
import torch.nn.functional as F
import torch_geometric.data as pyg_data
import numpy as np

from data.dataset import GraphDataset, OHGraphDataset
from data.dataloader import FewShotDataLoader, GraphDataLoader, get_dataloader
from models.gcn4maml import GCN4MAML
from models.sage4maml import SAGE4MAML
from algorithms.asmaml.asmaml import AdaptiveStepMAML
from utils.utils import elapsed_time, data_batch_collate_edge_renamed, compute_accuracy
from typing import Optional
from tqdm import tqdm

import logging
import config


def __test_meta_step(model: AdaptiveStepMAML, support_data: pyg_data.Data, query_data: pyg_data.Data) -> float:
    """ Run a step of the meta-testing """
    accs, step, *_ = model.finetuning(support_data, query_data)
    return accs[step]


def __test_step(model: GCN4MAML | SAGE4MAML, support_data: pyg_data, query_data: pyg_data) -> float:
    """ Run a step the non-meta testing """
    merged_data = data_batch_collate_edge_renamed([support_data, query_data], False)
    logits, _, _ = model(merged_data.x, merged_data.edge_index, merged_data.batch)
    preds = F.softmax(logits, dim=1).argmax(dim=1)
    return compute_accuracy(preds, merged_data.y, False)


@elapsed_time
def test(
    test_ds: GraphDataset | OHGraphDataset, logger: logging.Logger, pre_trained_model: str,
    n_way: int=config.TEST_WAY, k_shot: int=config.VAL_SHOT, n_query: int=config.VAL_QUERY, 
    episodes: int=config.VAL_EPISODE, shuffle: bool=True, batch_size: int=config.BATCH_SIZE, 
    dl_type: FewShotDataLoader | GraphDataLoader=FewShotDataLoader, model_name: str="sage", 
    meta: Optional[AdaptiveStepMAML]=None, dataset_name: str="TRIANGLES", device: str="cpu",
    inner_lr: float=config.INNER_LR, grad_clip: int=config.GRAD_CLIP, min_step: int=config.MIN_STEP,
    batch_episode: int=config.BATCH_PER_EPISODES, max_step: int=config.MAX_STEP,
    penalty: float=config.STEP_PENALITY, outer_lr: float=config.OUTER_LR, stop_lr: float=config.STOP_LR,
    patience: float=config.PATIENCE, weight_decay: float=config.WEIGHT_DECAY,
    scis: int=config.STOP_CONTROL_INPUT_SIZE, schs: int=config.STOP_CONTROL_HIDDEN_SIZE
) -> None:
    """
    Run testing

    :param test_ds: the test dataset
    :param logger: a simple logger
    :param pre_trained_model: the absolute path of the pre-trained model
    :param n_way: the number of class to sample
    :param k_shot: the number of support sample for each class
    :param n_query: the number of query sample for each class
    :param episodes: the number of episode for testing
    :param shuffle: True if shuffle the dataset or not
    :param batch_size: the size of each batch
    :param dl_type: the type of the dataloader to use
    :param model_name: the name of the base model
    :param meta: the class of the meta-model to use, None otherwise
    """
    # 1. Create the dataloader for the test set 
    test_dl = get_dataloader(test_ds, n_way, k_shot, n_query, episodes, shuffle, batch_size, dl_type)
    logger.debug("Created dataloder for test dataset of type: " + test_dl.__class__.__name__)

    # 2. Create the base model
    models = {"sage" : SAGE4MAML, "gcn" : GCN4MAML}
    model = models[model_name](
        num_classes=n_way, paper=False,
        num_features=config.NUM_FEATURES[dataset_name]).to(device)
    logger.debug("Created model of type: " + model.__class__.__name__)

    model2use = model

    # 3. Create the meta model if necessary
    meta_model = None
    if meta is not None:
        configurations = {
            "inner_lr"           : inner_lr,
            "train_way"          : n_way,
            "train_shot"         : k_shot,
            "train_query"        : n_query,
            "grad_clip"          : grad_clip,
            "batch_per_episodes" : batch_episode,
            "flexible_step"      : config.FLEXIBLE_STEP,
            "min_step"           : min_step,
            "max_step"           : max_step,
            "step_test"          : config.STEP_TEST,
            "step_penalty"       : penalty,
            "use_score"          : config.USE_SCORE,
            "use_loss"           : config.USE_LOSS,
            "outer_lr"           : outer_lr,
            "stop_lr"            : stop_lr,
            "patience"           : patience,
            "weight_decay"       : weight_decay,
            "scis"               : scis,
            "schs"               : schs
        }

        meta_model = AdaptiveStepMAML(model, False, **configurations).to(device)
        model2use = meta_model
        logger.debug("Created Meta-model of type " + meta_model.__class__.__name__)

    # 4. Load the state dict from the saved trained model
    saved_data = torch.load(pre_trained_model)
    model2use.load_state_dict(saved_data["embedding"])
    logger.debug("Loaded StateDict from " + pre_trained_model)

    logger.debug("Starting testing " + dataset_name + " with model " + model2use.__class__.__name__)
    print("=" * 100)

    # 5. Start testing
    val_accs = []
    model2use.eval()

    for _, data in enumerate(tqdm(test_dl(0)), 1):
        support_data, _, query_data, _ = data
        
        # Send to GPU if necessary
        if device != "cpu":
            support_data = support_data.to(device)
            query_data = query_data.to(device)

        # Execute the test step
        step_fn = __test_meta_step if meta is not None else __test_step
        step_acc = step_fn(model2use, support_data, query_data)
        val_accs.append(step_acc)

    val_acc_avg = np.mean(val_accs)
    val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(episodes)
    print("Mean Test Accuracy: {:.2f} ±{:.2f}".format(val_acc_avg, val_acc_ci95))
    print("=" * 100)

    return