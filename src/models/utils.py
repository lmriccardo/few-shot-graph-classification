from torch_geometric.data import Data

from data.dataset import GraphDataset
from utils.utils import graph2data, rename_edge_indexes
from typing import Dict, List, Tuple

import math
import torch
import networkx as nx
import config
import logging


def glorot(tensor):
    """Apply the Glorot NN initialization (also called Xavier)"""
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    """Fill a tensor with zeros if it is not Null"""
    if tensor is not None:
        tensor.data.fill_(0)


#####################################################################################
############################### ML-EVOLVE FILTRATION ################################
#####################################################################################
