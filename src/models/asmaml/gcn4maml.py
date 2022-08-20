""" For the source of this code check-out https://github.com/NingMa-AI/AS-MAML/blob/master/models/GCN4maml.py """

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_add

from models.asmaml.pool import TopKPooling
from models.asmaml.conv import GCNConv
from models.asmaml.linear import LinearModel
import config


class NodeInformationScore(MessagePassing):
    """Node information score"""
    def __init__(self, improved=False, cached=False, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None
    
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        edge_index, _ = remove_self_loops(edge_index)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0), ), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes, ), dtype=dtype, device=edge_index.device)

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class GCN4MAML(nn.Module):
    """GCN for AS-MAML"""
    def __init__(self, num_features: int=1, num_classes: int=30, paper: bool=False) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_classes  = num_classes
        self.paper        = paper
        
        # Define convolutional layers
        self.conv1 = GCNConv(self.num_features, config.NHID)
        self.conv2 = GCNConv(config.NHID, config.NHID)
        self.conv3 = GCNConv(config.NHID, config.NHID)

        self.calc_information_score = NodeInformationScore()

        # Define Pooling layers
        self.pool1 = TopKPooling(config.NHID, config.POOLING_RATIO)
        self.pool2 = TopKPooling(config.NHID, config.POOLING_RATIO)
        self.pool3 = TopKPooling(config.NHID, config.POOLING_RATIO)

        # Define Linear Layers
        self.linear1 = LinearModel(config.NHID * 2, config.NHID)
        self.linear2 = LinearModel(config.NHID, config.NHID // 2)
        self.linear3 = LinearModel(config.NHID // 2, self.num_classes)

        # Define activation function
        self.relu = F.leaky_relu

    def forward(self, x, edge_index, batch):
        edge_attr = None

        if self.paper:
            edge_index = edge_index.transpose(0,1)

        x = self.relu(self.conv1(x, edge_index, edge_attr), negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self.relu(self.conv2(x, edge_index, edge_attr), negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self.relu(self.conv3(x, edge_index, edge_attr), negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, None, batch)

        x_information_score = self.calc_information_score(x, edge_index)
        score = torch.sum(torch.abs(x_information_score), dim=1)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self.relu(x1, negative_slope=0.1) + \
            self.relu(x2, negative_slope=0.1) + \
            self.relu(x3, negative_slope=0.1)
        
        x = self.relu(self.linear1(x), negative_slope=0.1)
        x = self.relu(self.linear2(x), negative_slope=0.1)
        x = self.linear3(x)

        return x, score.mean(), None