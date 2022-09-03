import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, global_max_pool
from copy import deepcopy

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

        old_batch = batch.tolist()
        old_x = x.tolist()

        x = self.relu(self.conv1(x, edge_index, edge_attr),negative_slope=0.1)
        if x.isnan().sum() != 0:
            print("Conv1 Is NAN: ", x)
            print(old_x)
            print(batch)
            print(old_batch)

        old_x = x.tolist()
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, None, batch)
        if x.isnan().sum() != 0:
            print("Pool1, Is NAN: ", x)
            print(old_x)
            print(batch)
            print(old_batch)
        
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        if x1.isnan().sum() != 0:
            print(x)
            print("Cat Is NAN: ", x1)
            print(batch)
            print(old_batch)
        
        old_x = x.tolist()
        x = self.relu(self.conv2(x, edge_index, edge_attr),negative_slope=0.1)
        if x.isnan().sum() != 0:
            print("Conv2 Is NAN: ", x)
            print(old_x)
            print(batch)
            print(old_batch)
        
        old_x = x.tolist()
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, None, batch)
        if x.isnan().sum() != 0:
            print("Poll2, Is NAN: ", x)
            print(old_x)
            print(batch)
            print(old_batch)
        
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        if x2.isnan().sum() != 0:
            print(x)
            print("Cat2 Is NAN: ", x2)
            print(batch)
            print(old_batch)

        old_x = x.tolist()
        x = self.relu(self.conv3(x, edge_index, edge_attr), negative_slope=0.1)
        if x.isnan().sum() != 0:
            print(" Conv3,Is NAN: ", x)
            print(old_x)
            print(batch)
            print(old_batch)
        
        old_x = x.tolist()
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, None, batch)
        if x.isnan().sum() != 0:
            print("Pool3, Is NAN: ", x)
            print(old_x)
            print(batch)
            print(old_batch)
        
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        if x3.isnan().sum() != 0:
            print(x)
            print("Cat3 Is NAN: ", x3)
            print(batch)
            print(old_batch)

        x_information_score = self.calc_information_score(x, edge_index)
        score = torch.sum(torch.abs(x_information_score), dim=1)

        old_x = x.tolist()
        x = self.relu(x1,negative_slope=0.1) + \
            self.relu(x2,negative_slope=0.1) + \
            self.relu(x3,negative_slope=0.1)
        
        if x.isnan().sum() != 0:
            print("RELU Is NAN: ", x)
            print(old_x)
            print(batch)
            print(old_batch)

        graph_emb = x

        old_x = x.tolist()
        x = self.relu(self.linear1(x),negative_slope=0.1)
        if x.isnan().sum() != 0:
            print("Linear1 Is NAN: ", x)
            print(old_x)
            print(self.linear1.weight.fast, self.linear1.bias.fast)
            print(batch)
            print(old_batch)
        
        old_x = x.tolist()
        x = self.relu(self.linear2(x),negative_slope=0.1)
        if x.isnan().sum() != 0:
            print("Linear2 Is NAN: ", x)
            print(old_x)
            print(self.linear2.weight.fast, self.linear2.bias.fast)
            print(batch)
            print(old_batch)
        
        old_x = x.tolist()
        x = self.linear3(x)
        if x.isnan().sum() != 0:
            print("Linear3 Is NAN: ", x)
            print(old_x)
            print(self.linear3.weight.fast, self.linear3.bias.fast)
            print(batch)
            print(old_batch)

        return x, score.mean(), graph_emb