import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from models.conv import SAGEConv
from models.pool import SAGPool4MAML
from models.linear import LinearModel
from models.nis import NodeInformationScore
import config


class SAGE4MAML(nn.Module):
    """SAGE Model 4 MAML"""
    def __init__(self, num_features: int=1, num_classes: int=30, paper: bool=False) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_classes  = num_classes
        self.paper        = paper
        
        # Define convolutional layers
        self.conv1 = SAGEConv(self.num_features, config.NHID)
        self.conv2 = SAGEConv(config.NHID, config.NHID)
        self.conv3 = SAGEConv(config.NHID, config.NHID)

        self.calc_information_score = NodeInformationScore()

        # Define Pooling layers
        self.pool1 = SAGPool4MAML(config.NHID, config.POOLING_RATIO)
        self.pool2 = SAGPool4MAML(config.NHID, config.POOLING_RATIO)
        self.pool3 = SAGPool4MAML(config.NHID, config.POOLING_RATIO)

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

        x = self.relu(self.conv1(x, edge_index, edge_attr),negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x =self.relu(self.conv2(x, edge_index, edge_attr),negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self.relu(self.conv3(x, edge_index, edge_attr), negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x_information_score = self.calc_information_score(x, edge_index)
        score = torch.sum(torch.abs(x_information_score), dim=1)

        x = self.relu(x1,negative_slope=0.1) + \
            self.relu(x2,negative_slope=0.1) + \
            self.relu(x3,negative_slope=0.1)

        graph_emb = x

        x = self.relu(self.linear1(x),negative_slope=0.1)
        x = self.relu(self.linear2(x),negative_slope=0.1)
        x = self.linear3(x)

        return x, score.mean(), graph_emb