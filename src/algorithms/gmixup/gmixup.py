"""From the paper https://arxiv.org/pdf/2202.07179.pdf"""
import torch
import torch.nn.functional as F
import torch_geometric.data as pyg_data

from torch.nn.modules.loss import _Loss
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.pool import global_mean_pool
from utils.utils import build_adjacency_matrix, to_pygdata, to_datadict
from data.dataset import GraphDataset
from typing import List, Tuple
from copy import deepcopy

import math
import random


def align_graphs(graphs : List[pyg_data.Data], padding: bool=True) -> Tuple[
    List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor], int, int
]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of graph data
    :return: aligned graphs (as adjacency matrix) and normalized node degrees
    """
    num_nodes = [data.x.shape[0] for data in graphs]
    num_features = graphs[0].x.shape[1]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    aligned_node_xs = torch.empty((max_num * len(num_nodes), num_features))
    current_node_feature_idx = 0
    batches = []

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

            aligned_node_x = torch.zeros((max_num, num_features))
            aligned_node_x[:n_nodes, :] = sorted_node_x
            sorted_node_x = aligned_node_x

        sorted_node_x_shape = current_node_feature_idx + sorted_node_x.shape[0]
        normalized_node_degrees.append(sorted_node_degree)
        aligned_graphs.append(sorted_graph)
        aligned_node_xs[current_node_feature_idx : sorted_node_x_shape, :] = sorted_node_x
        current_node_feature_idx = sorted_node_x_shape
        batches.extend([graph_i] * sorted_node_x.shape[0])

    return aligned_graphs, aligned_node_xs, torch.tensor(batches),\
           normalized_node_degrees, max_num, min_num


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

    sample_graph_label = mixup_label
    sample_graph_x = mixup_x

    sample_graphs = []
    
    for _ in range(num_sample):
        sample_graph = (torch.rand(*mixup_graphon.shape) <= mixup_graphon).type(torch.int)
        sample_graph = torch.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.t() - torch.diag(sample_graph.diag())
        sample_graph = sample_graph[sample_graph.sum(dim=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(dim=0) != 0]
        edge_index, _ = dense_to_sparse(sample_graph)

        sample_graphs.append(pyg_data.Data(
                                x=sample_graph_x,
                                y=sample_graph_label,
                                edge_index=edge_index
                            ))
    
    return sample_graphs


class OHECrossEntropy(_Loss):
    """Cross Entropy function for one-hot encoded labels"""
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str="mean") -> None:
        super(OHECrossEntropy, self).__init__(size_average, reduce, reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()


class GMixupGDA:
    """
    G-Mixup from the paper https://arxiv.org/pdf/2202.07179.pdf
    This is the base idea: instead of directly manipulate graphs, 
    G-Mixup interpolate different graphons of different classes 
    in the Eucliden Space to get mixed graphons with it is possible
    to sample new training data. 

    It is based on Mixup, i.e., given two data sample (xi, yi) and
    (xj, yj) with yj != yi and a lambda la, the resulting
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
    
    def __call__(self, lam_range: Tuple[float]=[0.05, 0.1], 
                       aug_ratio: float=0.15,
                       aug_num  : int=10) -> None:
        """
        Run the data augmentation technique.

        **NOTE**: labels for mixup must be one-hot encoded
        
        :param lam_range: the low and high value for a uniform distribution
        :param aug_ratio: the augmentation ratio, used to compute the
                          total number of sample to generate.
        :param aug_num: the augmentation number
        :return: None
        """
        class_graphs = self.dataset.get_graphs_per_label()
        num_classes = self.dataset.number_of_classes
        classes_mapping = dict(zip(self.dataset.classes, range(num_classes)))
        graphons = []

        for label, graph_list in class_graphs.items():
            one_hot_label = F.one_hot(torch.tensor([classes_mapping[label]]).long(), num_classes=num_classes)[0]
            print(f"Estimating graphons for label: {label} -> {one_hot_label}")

            graph_list = [to_pygdata(graph, label) for graph in graph_list]
            aligned_graphs, aligned_nodes_features, batches, _, _, _ = align_graphs(graph_list)
            pooled_x = global_mean_pool(aligned_nodes_features, batches)
            graphon = usvd(aligned_graphs, 0.2)
            graphons.append((graphon, pooled_x, one_hot_label))

        num_sample = int( len(self.dataset) * aug_ratio / aug_num )
        lam_list = torch.Tensor(aug_num,).uniform_(lam_range[0], lam_range[1])

        print(f"=======================================\nNumber of samples: {num_sample}")
        print("Lambdas for Mixup: " + ",".join(["{:.4f}".format(x) for x in lam_list.tolist()]))
        print("=======================================")
        print("Executing Mixup for generating new sample data")

        augmented_graphs = []
        for lam in lam_list.tolist():
            print(f"--> Using lambda: {lam}")
            
            two_graphons = random.sample(graphons, k=2)
            augmented_graphs += two_graphons_mixup(two_graphons, la=lam, num_sample=num_sample)
            print(f"--> Generated new graphs for One-Hot Encoded label: {augmented_graphs[-1].y}")

        print("=======================================")
        print("Augmenting original dataset with new data")

        augmented_graphs = to_datadict(augmented_graphs)
        new_dataset = self.dataset + augmented_graphs

        print("Resulting new dataset has size of: ", len(new_dataset))
        
        return new_dataset