import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.data as pyg_data
import numpy as np

from torch.nn.modules.loss import _Loss, _WeightedLoss
from algorithms.asmaml.asmaml1 import AdaptiveStepMAML
from models.sage4maml import SAGE4MAML
from models.gcn4maml import GCN4MAML
from data.dataset import get_dataset, GraphDataset, OHGraphDataset
from data.dataloader import get_dataloader, FewShotDataLoader, GraphDataLoader
from utils.utils import configure_logger, elapsed_time
from typing import Union, Tuple, List, Optional, Dict, Any
from tqdm import tqdm

import config
import os
import sys
import logging


class Trainer(object):
    """
    A simple trainer class to train a model
    on the train and validation set
    Args:
        train_ds (GraphDataset or OHGraphDataset): the train set
        val_ds (GraphDataset or OHGraphDataset): the validation set
        model_name (str): the name of the model to use ('sage' or 'gcn')
        logger (logging.Logger): a simple logger
        paper (bool): if paper dataset is used or not
        meta_model (Optional[AdaptiveStepMAML], default=None): the meta model class to use
        epochs (int, default=200): number of total epochs to run
        dataset_name (str, default=TRIANGLES): the name of the used dataset
        dataloader_type (FewShotDataLoader | GraphDataLoader, default=FewShotDataLoader):
            the type of the dataloader to use for training and validation
        use_mevolve (bool, False): True if MEvolve should be used, false otherwise
        use_flag (bool, False): True if FLAG is used, false otherwise
        use_gmixup (bool, False): True if G-Mixup is used, false otherwise
    """
    def __init__(self, )