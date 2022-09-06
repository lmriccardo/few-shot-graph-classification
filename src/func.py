import torch

from torch.utils.data import Dataset, DataLoader
from data.sampler import FewShotSampler, TaskBatchSampler
from utils.utils import load_with_pickle, data_batch_collate, rename_edge_indexes, configure_logger
from typing import Any, Dict, List, Tuple, Union, Optional
from collections import defaultdict
from copy import deepcopy
from functools import partial

from typing import List, Generic, Optional
from config import T
import torch_geometric.data as gdata

import numpy as np

import config
import os
import logging


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
        last_id = max(self.__len__()) + 1
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
        edge_index = torch.tensor(g_data["edges"], dtype=torch.float).t().contiguous()
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

    def to_data(self) -> Tuple[gdata.Data, List[gdata.Data]]:
        """Return the entire dataset as a torch_geometric.data.Data"""
        data_list = []
        for _, (g_data, label) in self.graph_ds.items():
            x = torch.tensor(g_data["attributes"], dtype=torch.float)
            edge_index = torch.tensor(g_data["edges"], dtype=torch.float).t().contiguous()
            data = gdata.Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
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


def task_sampler_uncollate(task_sampler: TaskBatchSampler, 
                           data_batch  : List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] ) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
]:
    """
    Takes as input the task sampler and a batch containing both the 
    support and the query set. It returns two different DataBatch
    respectively for support and query_set.
    Assume L = [x1, x2, x3, ..., xN] is the data_batch
    each xi is a graph. Moreover, we have that
    L[0:K] = support sample for the first class
    L[K+1:K+Q] = query sample for the first class
    In general, we have that 
            L[i * (K + Q) : (i + 1) * (K + Q)]
    is the (support, query) pair for the i-th class
    Finally, the first batch is the one that goes from
    L[0 : N * (K + Q)], so
            L[i * N * (K + Q) : (i + 1) * N * (K + Q)]
    is the i-th batch.
    :param task_sampler: The task sampler
    :param data_batch: a batch with support and query set
    :return: support batch, query batch
    """
    n_way = task_sampler.task_sampler.n_way
    k_shot = task_sampler.task_sampler.k_shot
    n_query = task_sampler.task_sampler.n_query
    task_batch_size = task_sampler.task_batch_size

    total_support_query_number = n_way * (k_shot + n_query)
    support_plus_query = k_shot + n_query

    # Initialize batch list for support and query set
    support_data_batch = []
    query_data_batch = []

    # I know how many batch do I have, so
    for batch_number in range(task_batch_size):

        # I also know how many class do I have in a task
        for class_number in range(n_way):

            # First of all let's take the i-th batch
            data_batch_slice = slice(
                batch_number * total_support_query_number,
                (batch_number + 1) * total_support_query_number
            )
            data_batch_per_batch = data_batch[data_batch_slice]

            # Then let's take the (support, query) pair for a class
            support_query_slice = slice(
                class_number * support_plus_query,
                (class_number + 1) * support_plus_query
            )
            support_query_data = data_batch_per_batch[support_query_slice]

            # Divide support from query
            support_data = support_query_data[:k_shot]
            query_data = support_query_data[k_shot:support_plus_query]

            support_data_batch += support_data
            query_data_batch += query_data
    
    return support_data_batch, query_data_batch


def _collate(_batch: Generic[T]) -> Tuple[gdata.Data, List[gdata.Data]]:
    data_list = []

    for element in _batch:
        element_x = element[0]
        element_edge_index = element[1]
        element_label = element[2]

        # Create single data for datalist
        single_data = gdata.Data(x=element_x, edge_index=element_edge_index, y=element_label)
        data_list.append(single_data)

    return data_batch_collate(rename_edge_indexes(data_list)), data_list


def graph_collator(batch: Generic[T], sampler: TaskBatchSampler) -> Generic[T]:
    """collator"""
    support_data_batch, query_data_batch = task_sampler_uncollate(sampler, batch)
    return _collate(support_data_batch), _collate(query_data_batch)


class GraphDataLoader(DataLoader):
    def __init__(self, dataset     : GraphDataset,
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None,
                       **kwargs) -> None:

        if 'collate_fn' in kwargs:
            del kwargs["collate_fn"]

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.batch_sampler = kwargs["batch_sampler"]

        super().__init__(
            dataset,
            collate_fn=partial(graph_collator, sampler=self.batch_sampler),
            **kwargs,
        )


def get_dataloader(
    ds: GraphDataset, n_way: int, k_shot: int, n_query: int, 
    epoch_size: int, shuffle: bool, batch_size: int, 
    exclude_keys: Optional[List[str]] = None
) -> GraphDataLoader:
    """Return a dataloader instance"""
    return GraphDataLoader(
        dataset=ds,
        exclude_keys=exclude_keys,
        batch_sampler=TaskBatchSampler(
            dataset_targets=ds.targets(),
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            epoch_size=epoch_size,
            shuffle=shuffle,
            batch_size=batch_size
        )
    )


if __name__ == "__main__":
    torch.set_printoptions(edgeitems=config.EDGELIMIT_PRINT)
    logger = configure_logger(file_logging=config.FILE_LOGGING, logging_path=config.LOGGING_PATH)

    dataset_name = "TRIANGLES"
    train_ds, test_ds, val_ds, _ = get_dataset(
        download=config.DOWNLOAD_DATASET, 
        data_dir=config.DATA_PATH, 
        logger=logger,
        dataset_name=dataset_name
    )

    train_dataloader = get_dataloader(
        ds=train_ds, n_way=config.TRAIN_WAY, k_shot=config.TRAIN_SHOT,
        n_query=config.TRAIN_QUERY, epoch_size=config.TRAIN_EPISODE,
        shuffle=True, batch_size=1
    )

    print(next(iter(train_dataloader)))