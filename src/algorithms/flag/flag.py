"""From the paper https://arxiv.org/pdf/2010.09891.pdf"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch_geometric.data import Data

from typing import Union


class FlagGDA:
    """
    Free-Large Scale Adversarial Augmentation on Graphs (FLAG) is a 
    data augmentationa technique developed based on the adversarial
    perturbation. While most of the perturbation are directed towards
    edges or nodes, like edge/node dropping/adding, FLAG perturbation
    are directed to node features. This perturbations are generated
    by gradient-based robust optimization techniques. The "Free" stands
    for the fact that this can be used independently from the task goal:
    graph classification, node classification or link prediction. 

    Moreover, it is the first general-purpose feature-based data 
    augmentation method on graph data, which is complementary to other
    regularizers and topological augmentation. 

    Args:

    """
    def __init__(self, ) -> None:
        ...
    
    # TODO: to implement
    @staticmethod
    def flag(gnn: nn.Module, data: Data, criterion: Union[_Loss, _WeightedLoss]) -> None:
        ...