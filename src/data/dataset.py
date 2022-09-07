import torch
import torch_geometric.data as pyg_data

from torch.utils.data import Dataset
from utils.utils import rename_edge_indexes,    \
                        data_batch_collate,     \
                        load_with_pickle,       \
                        cartesian_product,      \
                        build_adjacency_matrix, \
                        to_nxgraph

from typing import Dict, Any, List, Union, Tuple, Optional
from collections import defaultdict
from copy import deepcopy
from numpy.linalg import matrix_power
from networkx.algorithms.link_prediction import resource_allocation_index

import logging
import os
import config
import numpy as np
import math
import random


class GraphDataset(Dataset):
    """Graph dataset"""
    def __init__(self, graph_ds: Dict[str, Dict[str, Any]]) -> None:
        super(GraphDataset, self).__init__()
        self.graph_ds = graph_ds
        self.count_per_class = dict()
    
    @classmethod
    def get_dataset(cls, attributes  : List[Any], 
                         data        : Dict[str, Any], 
                         num_features: int=1) -> 'GraphDataset':
        """
        """
        graphs = dict()

        label2graphs = data["label2graphs"]
        graph2nodes  = data["graph2nodes"]
        graph2edges  = data["graph2edges"]

        count_per_class = defaultdict(int)
        graph_counter = 0
        for label, graph_list in label2graphs.items():
            for graph_id in graph_list:
                graph_nodes = graph2nodes[graph_id]
                graph_edges = graph2edges[graph_id]
                nodes_attributes = []
                for node_id in graph_nodes:
                    attribute = attributes[node_id]
                    if num_features == 1:
                        attribute = [attribute]
                    
                    nodes_attributes.append(attribute)
                
                graph_data = {
                    "nodes" : graph_nodes,
                    "edges" : graph_edges,
                    "attributes" : [list(map(float, x)) for x in nodes_attributes]
                }

                graphs[graph_counter] = (graph_data, label)
                graph_counter += 1
            
            count_per_class[label] += len(graph_list)
        
        graphs = dict(sorted(graphs.items(), key=lambda x: x[0]))
        graph_dataset = super(GraphDataset, cls).__new__(cls)
        graph_dataset.__init__(graphs)
        graph_dataset.count_per_class = count_per_class

        return graph_dataset

    def __len__(self) -> int:
        """Return the lenght of the dataset as a number of graphs"""
        return len(self.graph_ds)
    
    def __repr__(self) -> str:
        """Return a description of the dataset"""
        return f"{self.__class__.__name__}(classes={set(self.classes)},n_graphs={self.__len__()})"

    def __add__(self, other: Union['GraphDataset', List[Tuple[Dict[str, Any], str]]]) -> 'GraphDataset':
        """Create a new graph dataset as the sum of the current and the input given dataset"""
        last_id = self.__len__() + 1
        if isinstance(other, GraphDataset):
            other = list(other.graph_ds.values())
        
        data_dict = deepcopy(self.graph_ds)
        for elem in other:
            data_dict[last_id] = elem
            last_id += 1
        
        new_ds = GraphDataset(data_dict)
        new_ds.count_per_class = GraphDataset._count_per_class(new_ds)
        return new_ds

    def __iadd__(self, other: Union['GraphDataset', List[Tuple[Dict[str, Any], str]]]) -> 'GraphDataset':
        return self.__add__(other)

    def __getitem__(self, idx: int | str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(idx, str):
            idx = int(idx)

        g_data, label = self.graph_ds[idx]
        x = torch.tensor(g_data["attributes"], dtype=torch.float)
        edge_index = torch.tensor(g_data["edges"], dtype=torch.long).t().contiguous()
        y = torch.tensor([label], dtype=torch.long)

        return (x, edge_index, y)
    
    @staticmethod
    def _count_per_class(graph_ds: 'GraphDataset') -> Dict[str, int]:
        """Create a dictionary for count_per_class attribute"""
        count_per_class = defaultdict(int)
        for _, (_, label) in graph_ds.graph_ds.items():
            count_per_class[label] += 1
        
        return count_per_class
    
    @property
    def classes(self) -> List[int | str]:
        """Return the total list of classes in the dataset"""
        return list(self.count_per_class.keys())

    @property
    def number_of_classes(self) -> int:
        """Return the total number of classes"""
        return len(self.count_per_class)
    
    def targets(self) -> torch.Tensor:
        """Returns for each graph its correspective class (with duplicate)"""
        return torch.tensor([int(label) for _, (_, label) in self.graph_ds.items()], dtype=torch.long)

    def get_graphs_per_label(self) -> Dict[str, List[Any]]:
        """Return a dictionary (label, list_of_graph with that label)"""
        graphs_per_label = {target : [] for target in self.classes}
        for _, (g_data, label) in self.graph_ds.items():
            graphs_per_label[label].append(g_data)
        
        return graphs_per_label

    def to_data(self) -> Tuple[pyg_data.Data, List[pyg_data.Data]]:
        """Return the entire dataset as a torch_geometric.data.Data"""
        data_list = []
        for _, (g_data, label) in self.graph_ds.items():
            x = torch.tensor(g_data["attributes"], dtype=torch.long)
            edge_index = torch.tensor(g_data["edges"], dtype=torch.long).t().contiguous()
            data = pyg_data.Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
            data_list.append(data)
        
        dataset_data = data_batch_collate(rename_edge_indexes(data_list))
        return dataset_data, data_list


def generate_train_val_test(dataset_name: str,
                            logger: logging.Logger,
                            data_dir: Optional[str]=None, 
                            download: bool=True
) -> Tuple[GraphDataset, GraphDataset, GraphDataset]:
    """ Return dataset for training, validation and testing """
    logger.debug("--- Generating Train, Test and Validation datasets --- ")
    
    assert download or data_dir is not None, "At least one between: data_dir and download must be given"

    node_attribute = None
    test_file = None
    train_file = None
    val_file = None

    if data_dir is not None:
        node_attribute = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_node_attributes.pickle")
        test_file = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_test_set.pickle")
        train_file = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_train_set.pickle")
        val_file = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_val_set.pickle")

    node_attribute_data = load_with_pickle(node_attribute)
    if isinstance(node_attribute_data, np.ndarray | torch.Tensor):
        node_attribute_data = node_attribute_data.tolist()

    test_data = load_with_pickle(test_file)
    train_data = load_with_pickle(train_file)
    val_data = load_with_pickle(val_file)

    train_ds = GraphDataset.get_dataset(node_attribute_data, train_data, num_features=config.NUM_FEATURES[dataset_name])
    test_ds  = GraphDataset.get_dataset(node_attribute_data,  test_data, num_features=config.NUM_FEATURES[dataset_name])
    val_ds   = GraphDataset.get_dataset(node_attribute_data,   val_data, num_features=config.NUM_FEATURES[dataset_name])

    return train_ds, test_ds, val_ds, data_dir


def get_dataset(logger: logging.Logger, 
                download: bool=False, 
                dataset_name: str="TRIANGLES", 
                data_dir: str="../data") -> Tuple[GraphDataset, GraphDataset, GraphDataset, str]:
    """Generate the train, test and validation dataset"""
    data_dir = data_dir if not download else None
    train_ds, test_ds, val_ds, data_dir = generate_train_val_test(
        data_dir=data_dir,
        download=download,
        dataset_name=dataset_name,
        logger=logger
    )
    return train_ds, test_ds, val_ds, data_dir


def split_dataset(dataset: GraphDataset, n_sample: int=2, **kwargs) -> List[GraphDataset]:
    """
    Split a single dataset into a number of smaller dataset

    :param dataset: the full dataset
    :param n_sample (default=2): the total number of smaller dataset to return
    :param ratios (optional, Dict[str, float]): a dictionary where specifying sample ratio
                                                for each of the N number of smaller dataset
                                                the sum of all ratio must be <= 1.
    """
    # FIXME: now implemented only for a 2-split of the original dataset. Extends to N-split
    graphs_per_label = dataset.get_graphs_per_label()
    count_per_label = dataset.count_per_class

    percs = [.8, .2]
    ds_list = []
    start_sampling_idx = {k : 0 for k in count_per_label}

    for sample_number in range(n_sample):
        graph_dict = dict()
        graph_id = 0
        graph_list = []
        perc = percs[sample_number]

        for label, gs in graphs_per_label.items():
            to_sample = math.ceil(perc * count_per_label[label])
            labels = [label] * to_sample
            graph_elem = list(zip(gs[start_sampling_idx[label]:start_sampling_idx[label] + to_sample], labels))
            graph_list += graph_elem
            start_sampling_idx[label] += to_sample
        
        for g_data, label in graph_list:
            graph_dict[graph_id] = (g_data, label)
            graph_id += 1
        
        new_ds = GraphDataset(graph_dict)
        new_ds.count_per_class = GraphDataset._count_per_class(new_ds)
        ds_list.append(new_ds)
    
    return ds_list


def get_dataset_from_labels(dataset: GraphDataset, labels: List[str | int]) -> GraphDataset:
    """
    Starting from the original dataset it returns a subset of it.
    Moreover, it returns a new dataset which classes 'labels'.

    :param dataset: the original dataset
    :param labels: classes to consider
    :return: a new dataset
    """
    graph_ds = dict()
    for graph_id, (g, label) in dataset.graphs_ds.items():
        if label in labels or str(label) in labels:
            graph_ds[graph_id] = (g, label)
    
    new_dataset = GraphDataset(graph_ds)
    new_dataset.count_per_class = GraphDataset._count_per_class(new_dataset)
    return new_dataset


#####################################################################################
############################### ML-EVOLVE HEURISTICS ################################
#####################################################################################

def random_mapping_heuristic(graphs: GraphDataset) -> List[Tuple[Dict[str, Any], str]]:
    """
    Random mapping is the first baseline heuristics used in the
    ML-EVOLVE graph data augmentation technique, shown in the 
    https://dl.acm.org/doi/pdf/10.1145/3340531.3412086 paper by Zhou et al.

    The idea is the followind (for a single graph): we have to create E_cdel 
    and E_cadd. First of all they set E_cdel = E (i.e., the entire set of 
    existing edges) and E_cadd = all non existing edges. Then to construct
    E_add and E_del they sample from the respective set.

            E_add = random.sample(E_cadd, size=ceil(m * beta)) and
            E_del = random.sample(E_cdel, size=ceil(m * beta))

    where m = |E| and beta is a number setted to 0.15 in the paper.

    :param graphs: the entire dataset of graphs
    :return: the new graph G' = (V, (E + E_add) - E_del)
    """
    new_graphs = []
    
    # Iterate over all graphs
    for _, ds_element in graphs.graph_ds.items():
        current_graph_data, label = ds_element

        # Takes all edges
        e_cdel = current_graph_data["edges"]

        # Takes every pair of nodes that is not an edge
        e_cadd = []
        for node_x, node_y in cartesian_product(current_graph_data["nodes"]):
            if node_x != node_y and (node_x, node_y) not in e_cdel:
                e_cadd.append([node_x, node_y])
                e_cadd.append([node_y, node_x])
        
        if not e_cadd:
            continue
        
        # Then we have to sample
        number_of_edges = len(current_graph_data["edges"])
        e_add = random.sample(e_cadd, k=math.ceil(number_of_edges * config.BETA))
        e_del = random.sample(e_cdel, k=math.ceil(number_of_edges * config.BETA))

        # Remove and add edges
        for e in e_del:
            current_graph_data["edges"].remove(e)
        
        current_graph_data["edges"].extend(e_add)
        
        # Let's do a deepcopy to not modify the original graph
        new_graphs.append((current_graph_data, label))
    
    return new_graphs


def motif_similarity_mapping_heuristic(graphs: GraphDataset) -> List[Tuple[Dict[str, Any], str]]:
    """
    Motif-similarity mapping is the second heuristics for new 
    data generation, presented in https://dl.acm.org/doi/pdf/10.1145/3340531.3412086 
    paper by Zhou et al. The idea is based on the concept of graph motifs: 
    sub-graphs that repeat themselves in a specific graph or even among various
    graphs. Each of these sub-graphs, defined by a particular pattern of
    interactions between vertices, may describe a framework in which particular
    functions are achieved efficiently.

    In this case they consider the so-called open-triad, equivalent to a lenght-2
    paths emanating from the head vertex v that induce a triangle. That is, an
    open-triad is for instance a sub-graph composed of three vertices v1, v2, v3
    and this edges (v1, v2) and (v1, v3). This induce a triangle since we can swap
    edges like (v1, v2) becomes (v2, v3) or (v1, v3) becomes (v3, v2). 

    This is the base idea: for all open-triad with vertex head v and tail u we
    construct E_cadd = {(v, u) | A(u,v)=0 and A^2(u,v)=0 and v != u} where
    A(u,v) is the value of the adjacency matrix for the edge (v,u). Then to construct
    E_add we do a weighted random sampling from E_cadd, where the weight depends on
    an index called the Resource Allocation Index (The formula can be found in the
    'networkx' python module under networkx.algorithms.link_prediction.resource_allocation_index).
    Similarly, we compute the deletation probability as w_del = 1 - w_add, and finally
    for each open-triad involving (v, u) we weighted random sample edges to remove.
    This removed edges will construct the E_del set.

    You can have a better look of the algorithm (Algorithm 1) in this paper
    https://arxiv.org/pdf/2007.05700.pdf (by Zhou et al)


    :param graphs: the dataset with all graphs
    :return: the new graph G' = (V, (E + E_add) - E_del)
    """
    new_graphs = []

    # Iterate over all graphs
    for _, ds_element in graphs.graph_ds.items():
        current_graph_data, label = ds_element
        number_of_nodes = len(current_graph_data["nodes"])
        number_of_edges = len(current_graph_data["edges"])

        # First of all let's define a mapping from graph's nodes and
        # their indexes in the adjacency matrix, so also for a the reverse mapping
        node_mapping = dict(zip(current_graph_data["nodes"], range(number_of_nodes)))
        reverse_node_mapping = {v : k for k, v in node_mapping.items()}

        # Then we need the adjancency matrix and its squared power
        # Recall that the square power of A, A^2, contains for all
        # (i, j): A[i,j] = deg(i) if i = j, otherwise it shows if
        # there exists a path of lenght 2 that connect i with j. In
        # this case the value of the cell is A[i,j] = 1, 0 otherwise.
        adj_matrix = build_adjacency_matrix(current_graph_data).numpy()
        power_2_adjancency = matrix_power(adj_matrix, 2)
        
        # The first step of the algorithm is to compute E_cadd
        e_cadd = []
        for node_x, node_y in cartesian_product(current_graph_data["nodes"]):

            # mapping is needed to index the adjancency matrix
            node_x, node_y = node_mapping[node_x], node_mapping[node_y]

            # In this case, what we wanna find are all that edges that
            # does not exists in the graph, but if they would exists than
            # no open-triad could be present inside the graph.
            if adj_matrix[node_x,node_y] == 0 and power_2_adjancency[node_x,node_y] != 0 \
                and node_x != node_y and (node_y, node_x) not in e_cadd:
                e_cadd.append((reverse_node_mapping[node_x], reverse_node_mapping[node_y]))
                e_cadd.append((reverse_node_mapping[node_y], reverse_node_mapping[node_x]))

        possible_triads = dict()
        rai_dict = dict()
        total_rai_no_edges = 0.0

        # In this step, we search for all edges inside E_cadd
        # the other two edges that constitute the triad. The search
        # look only for the first pair, no furthermore. 
        # Here we can also compute the Resource Allocation Index.
        for (node_x, node_y) in e_cadd:
            node_x, node_y = node_mapping[node_x], node_mapping[node_y]
            for node in current_graph_data["nodes"]:
                node = node_mapping[node]
                if adj_matrix[node_x, node] + adj_matrix[node, node_y] == 2:
                    node_x = reverse_node_mapping[node_x]
                    node = reverse_node_mapping[node]
                    node_y = reverse_node_mapping[node_y]
                    possible_triads[(node_x, node_y)] = [(node_x, node), (node, node_y)]
                    break
            
            nx_graph = to_nxgraph(current_graph_data)

            rai = resource_allocation_index(nx_graph, [(node_x, node_y)])
            _, _, rai = next(iter(rai))
            rai_dict[(node_x, node_y)] = rai
            total_rai_no_edges += rai

            edges = possible_triads[(node_x, node_y)]
            for u, v, p in resource_allocation_index(nx_graph, edges):
                if (u, v) not in rai_dict:
                    rai_dict[(u, v)] = p

        # In this step of the algorithm, we have to construct the W_add set.
        # Then, we can sample some edges from E_cadd and construct E_add.
        w_add = dict()
        for (node_x, node_y) in rai_dict:
            if (node_x, node_y) in e_cadd:
                w_add[(node_x, node_y)] = rai_dict[(node_x, node_y)] / total_rai_no_edges

        e_add_sample_number = math.ceil(number_of_edges * config.BETA)

        idxs = list(range(0, len(e_cadd)))
        p_distribution = list(w_add.values())

        choices = np.random.choice(idxs, size=e_add_sample_number, p=p_distribution, replace=False)
        e_add = np.array(e_cadd)[choices].tolist()
        e_add = list(map(tuple, e_add))

        # Finally, the second to the last step is to fill the E_del set.
        # In this step we compute the deletation weights, only for those
        # edges that belongs to the same triad of the previously chosen
        # edges. Essentially, those edges that belongs to E_add
        e_del = []
        for edge in e_add:
            left, right = possible_triads[edge]
            if (left, right) in e_del:
                continue

            w_del_left = 1 - rai_dict[left] / total_rai_no_edges
            w_del_right = 1 - rai_dict[right] / total_rai_no_edges
            p_distribution = [w_del_left, w_del_right]
            ch = random.choices([left, right], k=1, weights=p_distribution)[0]

            e_del.append(ch)

        # Remove and add edges
        for e in e_del:
            current_graph_data["edges"].remove(e)
        
        current_graph_data["edges"].extend(e_add)

        # The last step is to construct the new graph
        new_graphs.append((current_graph_data, label))
    
    return new_graphs


def augment_dataset(dataset: GraphDataset, heuristic: str="random mapping") -> List[Tuple[Dict[str,Any], str]]:
    """Apply the augmentation to the dataset"""
    heuristics = {
        "random_mapping"           : random_mapping_heuristic,
        "motif_similarity_mapping" : motif_similarity_mapping_heuristic
    }

    chosen_heuristic = heuristics[heuristic]
    augmented_data = chosen_heuristic(dataset)

    return augmented_data