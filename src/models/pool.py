""" For the source of this code check-out https://github.com/NingMa-AI/AS-MAML/blob/master/models/TopKPoolfw.py """
# NOTE:
# - edge_index as input to torch_geometric.utils.num_nodes.maybe_num_nodes must be a torch.Tensor
# - value as input to torch_geometric.nn.inits.uniform must be a not Null torch.Tensor
# - ratio as input to torch_geometric.nn.pool.topk_pool.topk must not be a int


import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn.pool.sag_pool import SAGPooling
from models.conv import GraphConv
from models.functions import uniform

from typing import Optional


class TopKPooling(nn.Module):
    """
    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_channels, 
                       ratio: float=0.5,
                       min_score: Optional[float]=None,
                       multiplier: int=1,
                       nonlinearity=torch.tanh) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = nn.Parameter(torch.Tensor(1, in_channels))
        self.weight.fast = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters"""
        size = self.in_channels
        uniform(size, self.weight)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """The forward method"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn

        if self.weight.fast is not None:
            score = (attn * self.weight.fast).sum(dim=-1)
        else:
            score = (attn * self.weight).sum(dim=-1)
        
        if self.min_score is None:
            if self.weight.fast is not None:
                score = self.nonlinearity(score / self.weight.fast.norm(p=2, dim=-1))
            else:
                score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)
        
        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm,
            num_nodes=score.size(0)
        )

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self):
        return '{}({}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)


# class SAGPool4MAML(SAGPooling):
#     """SAGPooling for MAML. Change only the __repr__ method"""
#     def __init__(self, in_channels: int, ratio: float=0.5,
#                        GNN: nn.Module=GraphConv, min_score: Optional[float]=None,
#                        multiplier: int=1, nonlinearity=torch.tanh, **kwargs) -> None:
#         super().__init__(
#             in_channels=in_channels, ratio=ratio,
#             GNN=GNN, min_score=min_score, multiplier=multiplier,
#             nonlinearity=nonlinearity, **kwargs
#         )

#     def __repr__(self) -> str:
#         return '{}({}, {}, {}={}, multiplier={})'.format(
#             self.__class__.__name__, self.gnn.__class__.__name__,
#             self.in_channels,
#             'ratio' if self.min_score is None else 'min_score',
#             self.ratio if self.min_score is None else self.min_score,
#             self.multiplier)

class SAGPool4MAML(nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers
    if :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`:
        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})
            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})
            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}
            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}
    if :obj:`min_score` :math:`\tilde{\alpha}` is a value in [0, 1]:
        .. math::
            \mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))
            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}
            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}
            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},
    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Projections scores are learned based on a graph neural network layer.
    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            calculating projection scores (one of
            :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv`,
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.SAGEConv`). (default:
            :class:`torch_geometric.nn.conv.GraphConv`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """

    def __init__(self, in_channels, ratio=0.5, GNN=GraphConv, min_score=None,
                 multiplier=1, nonlinearity=torch.tanh, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = GNN(in_channels, 1, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn, edge_index).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self):
        return '{}({}, {}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.gnn.__class__.__name__,
            self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)