""" For the source of this code check-out https://github.com/NingMa-AI/AS-MAML/blob/master/models/GcnConv.py """


import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import add_remaining_self_loops
from models.utils import glorot, zeros
from models.linear import LinearModel


class GCNConv(MessagePassing):
    """
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int,
                       out_channels: int,
                       improved: bool=False,
                       cached: bool=False,
                       bias: bool=True,
                       **kwargs):
        super().__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight.fast = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias.fast = None
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        """Compute the Norm"""
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

        row, col = edge_index

        # src = edge_weight
        # index = row
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    def forward(self, x, edge_index, edge_weight=None):
        """The forward method"""
        x = x @ (self.weight if self.weight.fast is None else self.weight.fast)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(
                edge_index, x.size(self.node_dim), edge_weight,
                self.improved, x.dtype
            )
            self.cached_result = edge_index, norm
        
        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        if self.bias is not None:
            if self.bias.fast is not None:
                aggr_out += self.bias.fast
            else:
                aggr_out += self.bias
        
        return aggr_out
    
    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__, 
            self.in_channels, 
            self.out_channels
        )


class SAGEConv(MessagePassing):
    """
    The GraphSAGE operator, modified for the fast weight adaptation

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                       normalize: bool=False, bias: bool=True,
                       **kwargs) -> None:
        super().__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.weight.fast = None

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
            self.bias.fast = None
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)
    
    def forward(self, x, edge_index, edge_weight=None, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(0)
            )

        if self.weight.fast is not None:
            weight = self.weight.fast
        else:
            weight = self.weight

        if torch.is_tensor(x):
            x = x @ weight
        else:
            x0 = None if x[0] is None else x[0] @ weight
            x1 = None if x[1] is None else x[1] @ weight
            x = (x0, x1)
    
        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            if self.bias.fast is not None:
                aggr_out = aggr_out + self.bias.fast
            else:
                aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphConv(MessagePassing):
    """
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, aggr='add', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = LinearModel(in_channels, out_channels)
        self.weight.fast = None
        self.lin.weight.fast = None
        self.lin.bias.fast = None

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        if self.weight.fast is not None:
            h = x @ self.weight.fast
        else:
            h = x @ self.weight

        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return aggr_out + self.lin(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)