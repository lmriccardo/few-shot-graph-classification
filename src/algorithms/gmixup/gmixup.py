"""From the paper https://arxiv.org/pdf/2202.07179.pdf"""
import torch
import torch_geometric.data as pyg_data

from torch_geometric.utils import dense_to_sparse
from utils.utils import build_adjacency_matrix
from data.dataset import GraphDataset
from typing import List, Tuple
from copy import deepcopy

import math


class GMixupGDA:
    """
    G-Mixup from the paper https://arxiv.org/pdf/2202.07179.pdf
    This is the base idea: instead of directly manipulate graphs, 
    G-Mixup interpolate different graphons of different classes 
    in the Eucliden Space to get mixed graphons with it is possible
    to sample new training data. 

    It is based on Mixup, i.e., given two data sample (xi, yi) and
    (xj, yj) with yj != yi anda learning augmentation la, the resulting
    mixed new data is computed as follow: 

                xnew = la * xi + (1 - la) * xj
                ynew = la * yi + (1 - la) * yj

    However, since in this case we are talking about graphs, i.e. 
    x = G(V, E) that lives in a non-Euclidean Space, we cannot 
    directly use the above formulas. So, the idea is to consider
    graphons, i.e., symmetric, continouos and bounded function
    W s.t. given two nodes ui, uj -> W(ui,uj) is the probability
    that there exists the edge (ui, uj). Note that W lives in the 
    Euclidean Space since it is just a matrix. Then, given two
    graphs set we:
        
        1. Compute the respective graphons Wg and Wh
        2. Mixup graphons Wi = la * Wg + (1 - la) * Wh
        3. Mixup labels yi = la * yg + (1 - la) * yh
        4. Mixup node features Xi = la * Xg + (1 - la) * Xh
        5. Sample new graphs with K nodes using Wi, with yi and Xi

    Args:
        dataset (GraphDataset): the training set to augment
    """
    def __init__(self, dataset: GraphDataset) -> None:
        self.dataset = dataset

    @staticmethod
    def align_graphs(graphs : List[pyg_data.Data], 
                     padding: bool=True) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, int]:
        """
        Align multiple graphs by sorting their nodes by descending node degrees

        :param graphs: a list of graph data
        :return: aligned graphs (as adjacency matrix) and normalized node degrees
        """
        num_nodes = [data.x.shape[0] for data in graphs]
        max_num = max(num_nodes)
        min_num = min(num_nodes)

        aligned_graphs = []
        normalized_node_degrees = []
        aligned_node_xs = []

        for graph_i, data in enumerate(graphs):
            n_nodes = num_nodes[graph_i]
            g = build_adjacency_matrix(data)

            node_degree = 0.5 * (g.sum(dim=1) + g.sum(dim=1))
            node_degree = node_degree / node_degree.sum()
            idx = torch.argsort(node_degree, descending=True)

            sorted_node_degree = node_degree[idx]
            sorted_node_degree = sorted_node_degree.reshape(-1, 1)

            sorted_graph = deepcopy(g)
            sorted_graph = sorted_graph[idx, :]
            sorted_graph = sorted_graph[:, idx]

            node_x = deepcopy(data.x)
            sorted_node_x = node_x[idx, :]

            if padding:
                normalized_node_degree = torch.zeros((max_num, 1))
                normalized_node_degree[:n_nodes, :] = sorted_node_degree
                sorted_node_degree = normalized_node_degree

                aligned_graph = torch.zeros((max_num, max_num))
                aligned_graph[:n_nodes, :n_nodes] = sorted_graph
                sorted_graph = aligned_graph

                aligned_node_x = torch.zeros((max_num, data.x.shape[1]))
                aligned_node_x[:n_nodes, :] = sorted_node_x
                sorted_node_x = aligned_node_x

            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)
            aligned_node_xs.append(sorted_node_x)

        return aligned_graphs, aligned_node_xs, normalized_node_degrees, max_num, min_num

    @staticmethod
    def usvd(aligned_graphs: List[torch.Tensor], threshold: float=2.02) -> torch.Tensor:
        """
        Estimate a graphon by Universal Singular Value Threshold

        Reference: "Matrix estimation by universal singular value thresholding."
        Link Paper PDF: https://arxiv.org/pdf/1212.1247.pdf

        :param aligned_graphs: a list of adjacency matrices
        :param threshold: the threshold for singular values
        :return: the estimated (r, r) graphon model
        """
        aligned_graphs = torch.tensor([t.tolist() for t in aligned_graphs]).float()
        sum_graphs = torch.mean(aligned_graphs, dim=0)
        num_nodes = sum_graphs.size(0)

        u, s, v = torch.linalg.svd(sum_graphs)
        singular_threshold = threshold * math.sqrt(num_nodes)
        binary_s = torch.lt(s, singular_threshold)
        s[binary_s] = 0
        graphon = u @ s.diag() @ v
        graphon[graphon > 1] = 1
        graphon[graphon < 0] = 0
        
        return graphon
    
    @staticmethod
    def two_graphons_mixup(two_graphons: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                           la: float=0.5,
                           num_sample: int=20) -> List[pyg_data.Data]:
        """
        Mixup two graphons and generate a number of samples. 

        """
        first_graphon, first_x, first_label = two_graphons[0]
        second_graphon, second_x, second_label = two_graphons[1]

        mixup_label = la * first_label + (1 - la) * second_label
        mixup_graphon = la * first_graphon + (1 - la) * second_graphon
        mixup_x = la * first_x + (1 - la) * second_x

        sample_graph_label = torch.tensor([mixup_label], dtype=torch.float)
        sample_graph_x = mixup_x

        sample_graphs = []
        
        for _ in range(num_sample):
            sample_graph = (torch.rand(*mixup_graphon.shape) <= mixup_graphon).type(torch.int)
            sample_graph = torch.triu(sample_graph)
            sample_graph = sample_graph + sample_graph.t() - torch.diag(sample_graph.diag())
            sample_graph = sample_graph[sample_graph.sum(dim=1) != 0]
            sample_graph = sample_graph[:, sample_graph.sum(dim=0) != 0]
            edge_index, _ = dense_to_sparse(sample_graph)

            sample_graphs.append(pyg_data.Data(x=sample_graph_x, y=sample_graph_label, edge_index=edge_index))
        
        return sample_graphs