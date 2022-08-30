# From the paper https://arxiv.org/pdf/2007.05700.pdf
# TODO: Implement Algorithm 2 of the paper
# TODO: This how to merge few-shot learning with cross-fold validation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.gcn4maml import GCN4MAML
from models.sage4maml import SAGE4MAML
from typing import Union


class MEvolve(nn.Module):
    """
    The Model Evolution framework presented in the paper
    
    Args
        model (Union[GCN4MAML, SAGE4MAML]): the pre-trained classifier
        n_iters (int): 
    """
    def __init__(self, model: Union[GCN4MAML, SAGE4MAML], n_iters: int) -> None:
        self.model = model
        self.n_iters = n_iters