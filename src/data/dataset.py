import torch
import torch.nn.functional as F
import torch_geometric.data as pyg_data

from torch.utils.data import Dataset
from utils.utils import rename_edge_indexes,    \
                        data_batch_collate,     \
                        load_with_pickle

from typing import Dict, Any, List, Union, Tuple, Optional
from collections import defaultdict
from copy import deepcopy

import logging
import os
import config
import numpy as np
import math


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
        Returns a new instance of GraphDataset filled with graphs inside data. 'attributes'
        is the list with all the attributes (not only those beloging to nodes in 'data').

        :param data: a dictionary with label2graphs, graph2nodes and graph2edges
        :param attributes: a list with node attributes
        :return: a new instance of GraphDataset
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

    def __add__(self, other: Union['GraphDataset', Dict[int, Tuple[Dict[str, Any], str]]]) -> 'GraphDataset':
        """Create a new graph dataset as the sum of the current and the input given dataset"""
        last_id = self.__len__() + 1
        if isinstance(other, GraphDataset):
            other = other.graph_ds
        
        data_dict = deepcopy(self.graph_ds)
        for elem in other.values():
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

    def get_graphs_per_label(self) -> Dict[int | str, List[int]]:
        """Return a dictionary (label, list_of_graph with that label)"""
        graphs_per_label = {target : [] for target in self.classes}
        for graph_id, (_, label) in self.graph_ds.items():
            graphs_per_label[label].append(graph_id)
        
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


class OHGraphDataset(torch.utils.data.Dataset):
    """Dataset for One-Hot Encoded labels"""
    def __init__(self, graph_ds: Optional[Dict[int, Dict[str, Any]] | GraphDataset]=None) -> None:
        self.is_oh_labels = False

        if graph_ds is not None:
            if isinstance(graph_ds, dict):
                graph_ds = GraphDataset(graph_ds)
                graph_ds.count_per_class = GraphDataset._count_per_class(graph_ds)

            self.old_graph_ds = graph_ds
            self.graph_ds = graph_ds.graph_ds
            self._to_onehot()
            self.is_oh_labels = True
        else:
            self.old_graph_ds = None
            self.graph_ds = None
            
        self.from_dict = False
        
    @classmethod
    def from_dict(cls, data_dict: Dict[int, Dict[str, Any]]) -> 'OHGraphDataset':
        oh_ds = super(OHGraphDataset, cls).__new__(cls)
        oh_ds.__init__()
        oh_ds.graph_ds = data_dict
        oh_ds.from_dict = True
        
        return oh_ds
            
    @property
    def num_classes(self) -> int:
        """Return the number of total classes"""
        return list(self.graph_ds.values())[0][-1].shape[0]

    def get_graphs_per_label(self) -> Dict[int | str, List[int]]:
        """Return a dictionary (label, list_of_graph with that label)"""
        label_to_graphs = defaultdict(list)
        arange = torch.arange(end=self.num_classes, start=0, step=1)
        for graph_id, (_, label) in self.graph_ds.items():
            graph_classes = arange[label > 0].tolist()
            for cls in graph_classes:
                label_to_graphs[cls].append(graph_id)
        
        return label_to_graphs
    
    def _to_onehot(self) -> None:
        """Tranform each label into a one-hot-encoded label"""
        # If labels are yet one-hot-encode, we don't
        # need to do anything of special
        if self.is_oh_labels:
            return 
        
        # First I need to take the total number of labels
        num_classes = max(self.old_graph_ds.classes) + 1
        ohe_mapping = dict(zip(
            self.old_graph_ds.classes, 
            F.one_hot(
                torch.tensor(self.old_graph_ds.classes), 
                num_classes=num_classes
            )
        ))
        
        # Map each label to its correspective OH encoding
        for key in self.graph_ds.keys():
            self.graph_ds[key] = (
                self.graph_ds[key][0], 
                ohe_mapping[self.graph_ds[key][-1]]
            )
        
        # Avoid possible future re-run of this method
        self.is_oh_labels = True
        
    def _add_from_dict(self, other: 'OHGraphDataset') -> 'OHGraphDataset':
        """Create a new summed dataset where other is a from_dict dataset"""
        ds = list(self.graph_ds.values()) + list(other.graph_ds.values())
        
        # Take the max overall the shape of the labels
        max_shape = max(self.num_classes, other.num_classes)
        zeros_matrix = torch.zeros((self.__len__() + len(other), max_shape))
        
        # Create a new dictionary and adjust the one-hot encoded labels
        graph_ds = dict()
        for ds_i, (g, label) in enumerate(ds):
            oh_label = zeros_matrix[ds_i]
            oh_label[:label.shape[0]] = label
            graph_ds[ds_i] = (g, oh_label)
        
        new_ds = OHGraphDataset.from_dict(graph_ds)
        return new_ds
        
    def __add__(self, other: Union['OHGraphDataset', GraphDataset, Dict[int, Dict[str, Any]]]) -> 'OHGraphDataset':
        """Create a new graph dataset as the sum of the current and the input given dataset"""
        if isinstance(other, OHGraphDataset) and other.from_dict:
            return self._add_from_dict(other)

        if isinstance(other, GraphDataset | dict):
            if isinstance(other, dict):
                # Take the first Item and check if its label
                # is a tensor with lenght grater than one.
                # In this case I assume that this is a one-hot
                # encoded class. 
                first_item_key = min(other.keys())
                first_item = other[first_item_key][-1]
                if isinstance(first_item, torch.Tensor) and first_item.shape[0] > 1:
                    other = OHGraphDataset.from_dict(other)
                    return self._add_from_dict(other)
                    
            other = OHGraphDataset(other)
            return self.__add__(other)
    
        graph_ds = self.old_graph_ds + other.old_graph_ds
        return OHGraphDataset(graph_ds)

    def __iadd__(self, other: Union['OHGraphDataset', GraphDataset, Dict[int, Dict[str, Any]]]) -> 'OHGraphDataset':
        return self.__add__(other)
    
    def __len__(self) -> int:
        """Return the total number of graphs in the dataset"""
        return len(self.graph_ds)
    
    def __repr__(self) -> str:
        """Return a description of the dataset"""
        return f"{self.__class__.__name__}(n_graphs={self.__len__()},n_classes={self.num_classes})"
            
    def __getitem__(self, idx: int | str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(idx, str):
            idx = int(idx)
        
        g_data, label = self.graph_ds[idx]
        x = torch.tensor(g_data["attributes"], dtype=torch.float)
        edge_index = torch.tensor(g_data["edges"], dtype=torch.long).t().contiguous()
        
        return (x, edge_index, label)


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

        for label, graph_ids in graphs_per_label.items():
            gs = [dataset.graph_ds[gid][0] for gid in graph_ids]
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


def get_dataset_from_indices(dataset: GraphDataset, indices: List[int]) -> GraphDataset:
    ...