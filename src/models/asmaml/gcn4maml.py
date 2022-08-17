""" For the source of this code check-out https://github.com/NingMa-AI/AS-MAML/blob/master/models/GCN4maml.py """

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from models.asmaml.pool import TopKPooling
