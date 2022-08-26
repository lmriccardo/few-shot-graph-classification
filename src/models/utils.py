from data.dataset import GraphDataset, \
                         random_mapping_heuristic, \
                         motif_similarity_mapping_heuristic

from typing import Dict, List, Tuple
from torch_geometric.data import Data

import math
import torch
import networkx as nx
import config


def glorot(tensor):
    """Apply the Glorot NN initialization (also called Xavier)"""
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    """Fill a tensor with zeros if it is not Null"""
    if tensor is not None:
        tensor.data.fill_(0)


#####################################################################################
############################### ML-EVOLVE FILTRATION ################################
#####################################################################################

def compute_threshold(graph_probability_vector: Dict[Data, torch.Tensor], 
                      label_reliabilities     : Dict[Data, float]) -> float:
    """
    Compute the threshold for the label reliability acceptation.
    This threshold is the result of the following expression

        t = arg min_t sum(T[(t - ri) * g(G_i, y_i)], (G_i, y_i) in D_val)
    
    Thus, we have to compute the derivative equal to 0

    :param graph_probability_vector: the probability vector for each graph
    :param label_reliabilities: label reliability value for each graph
    :return: the optimal threshold
    """
    ...

def data_filtering(train_ds                : GraphDataset, 
                   validation_ds           : GraphDataset,
                   graph_probability_vector: Dict[Data, torch.Tensor],
                   classes                 : List[int]) -> List[Tuple[nx.Graph, str]]:
    """
    After applying the heuristic for data augmentation, we have a
    bunch of new graphs and respective labels that needs to be
    added to the training set. Before this, we have to filter this data
    by label reliability. For further information look at the paper
    https://arxiv.org/pdf/2007.05700.pdf by Zhou et al.
    
    :param train_ds: the train dataset
    :param validation_ds: the validation dataset
    :param classes: the list of targets label
    :param graph_probability_vector: a dictionary mapping for each graph the
                                     probability vector obtained after running
                                     the pre-trained classifier.

    :return: the list of graphs and labels that are reliable to be added
    """
    # Get augmented data
    heuristics = {
        "random_mapping" : random_mapping_heuristic,
        "motif_similarity_mapping" : motif_similarity_mapping_heuristic
    }

    chosen_heuristic = heuristics[config.HEURISTIC]
    augmented_data = chosen_heuristic(train_ds)
    count_per_labels = validation_ds.count_per_class

    # Compute the confusion matrix Q
    n_classes = len(classes)
    confusion_matrix = torch.zeros((n_classes, n_classes))
    for idx, target in enumerate(classes):
       
        # Get graphs with a specific target
        prob_vector_target = [v.tolist() for g, v in graph_probability_vector.items() if g.y.item() == target]
        prob_vector_tensor = torch.tensor(prob_vector_target)
        confusion_matrix[idx] = 1 / count_per_labels[target] *  prob_vector_tensor.sum(dim=0)
    
    # Now, compute the label reliability for all graphs in the validation set
    label_reliabilities = dict()
    for graph, prob_vect in graph_probability_vector.items():
        label = graph.y.items()
        label_idx = classes.index(label)
        label_reliabilities[graph] = prob_vect @ confusion_matrix[label_idx]
    
