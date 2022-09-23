import torch
import torch_geometric.data as pyg_data


from torch.utils.data import DataLoader, default_collate
from utils.utils import data_batch_collate, rename_edge_indexes, setup_seed
from data.sampler import TaskBatchSampler
from data.dataset import GraphDataset, OHGraphDataset
from typing import Generic, Tuple, List, Optional, Iterator
from config import T
from functools import partial

import torchnet as tnt


def uncollate(sampler, batch):
    n_way = sampler.task_sampler.n_way
    k_shot = sampler.task_sampler.k_shot
    batch_size = sampler.task_batch_size

    idxs = torch.arange(start=0, end=len(batch), step=1).view(batch_size, -1)
    support_batch = []
    query_batch = []
    
    for support_query_idxs in idxs:
        support_idx, query_idx = support_query_idxs[: (n_way * k_shot)], support_query_idxs[(n_way * k_shot):]
        support_batch.extend([batch[x] for x in support_idx.tolist()])
        query_batch.extend([batch[x] for x in query_idx.tolist()])
        
    return support_batch, query_batch


def _collate(_batch: Generic[T], oh_labels: bool=False) -> Tuple[pyg_data.Data, List[pyg_data.Data]]:
    data_list = []

    for element in _batch:
        element_x = element[0]
        element_edge_index = element[1]
        element_label = element[2]

        # Create single data for datalist
        single_data = pyg_data.Data(x=element_x, edge_index=element_edge_index, y=element_label)
        data_list.append(single_data)

    return data_batch_collate(rename_edge_indexes(data_list), oh_labels=oh_labels)


def graph_collator(batch: Generic[T], sampler: TaskBatchSampler, oh_labels: bool=False) -> Generic[T]:
    """collator"""
    # support_data_batch, query_data_batch = task_sampler_uncollate(sampler, batch)
    support_data_batch, query_data_batch = uncollate(sampler, batch)
    return _collate(support_data_batch, oh_labels), _collate(query_data_batch, oh_labels)


# class FewShotDataLoader(DataLoader):
#     def __init__(self, dataset     : GraphDataset | OHGraphDataset,
#                        follow_batch: Optional[List[str]] = None,
#                        exclude_keys: Optional[List[str]] = None,
#                        **kwargs) -> None:

#         if 'collate_fn' in kwargs:
#             del kwargs["collate_fn"]

#         self.follow_batch = follow_batch
#         self.exclude_keys = exclude_keys

#         self.batch_sampler = kwargs["batch_sampler"]

#         super().__init__(
#             dataset,
#             collate_fn=partial(graph_collator, 
#                                sampler=self.batch_sampler, 
#                                oh_labels=self.batch_sampler.oh_labels),
#             **kwargs,
#         )
    
#     def __iter__(self) -> Iterator[Tuple[
#         pyg_data.Data, List[pyg_data.Data], pyg_data.Data, List[pyg_data.Data]
#     ]]:
#         for data in super().__iter__():
#             (data1, data2), (data3, data4) = data
#             yield data1, data2, data3, data4


import random
import torch


# class FewShotDataLoader(object):
#     """A simple few-shot dataloader"""
#     def __init__(self, dataset: GraphDataset | OHGraphDataset, **kwargs) -> None:
#         self.dataset = dataset
#         self.setattr(**kwargs)

#         self.targets = self.dataset.get_graphs_per_label()
        
#     def setattr(self, **attrs) -> None:
#         for k, v in attrs.items():
#             self.__setattr__(k, v)

#     def _uncollate(self, batch):
#         idxs = torch.arange(start=0, end=len(batch), step=1).view(self.batch_size, -1)
#         support_batch = []
#         query_batch = []
        
#         for support_query_idxs in idxs:
#             support_idx, query_idx = support_query_idxs[: (self.n_way * self.k_shot)], support_query_idxs[(self.n_way * self.k_shot):]
#             support_batch.extend([batch[x] for x in support_idx.tolist()])
#             query_batch.extend([batch[x] for x in query_idx.tolist()])
            
#         return support_batch, query_batch

#     def __iter__(self) -> Iterator[Tuple[pyg_data.Data, List[pyg_data.Data], pyg_data.Data, List[pyg_data.Data]]]:
#         target_classes = random.sample(list(self.targets.keys()), k=self.n_way)
#         for _ in range(self.epoch_size):

#             n_way_k_shot_query = []
#             for cl in target_classes:

#                 graphs = self.targets[cl]
#                 assert len(graphs) >= self.k_shot + self.n_query, "Not enough graphs for sampling"
#                 selected_data = random.sample(graphs, k=self.k_shot + self.n_query)
#                 n_way_k_shot_query += selected_data
            
#             n_way_k_shot_query = torch.tensor(n_way_k_shot_query)
#             perm = torch.randperm(n_way_k_shot_query.shape[0])
#             n_way_k_shot_query = n_way_k_shot_query[perm]
            
#             batch = [self.dataset[i.item()] for i in n_way_k_shot_query]
#             support_data_batch, query_data_batch = self._uncollate(batch)
#             support_data, support_data_list = _collate(support_data_batch, False)
#             query_data, query_data_list = _collate(query_data_batch, False)

#             yield support_data, support_data_list, query_data, query_data_list


def coll(batch):
    return batch[0]


# TODO: Handle one-hot encoded labels
class FewShotDataLoader(object):
    def __init__(self, dataset: GraphDataset | OHGraphDataset, **kwargs) -> None:
        self.dataset = dataset
        self.num_workers = 2
        self.setattr(**kwargs)
        
        self.labels = self.dataset.get_graphs_per_label()
        
    def setattr(self, **attrs) -> None:
        for k, v in attrs.items():
            self.__setattr__(k, v)
    
    def sample_class(self) -> List[int]:
        """Sample N classes"""
        return random.sample(list(self.labels.keys()), k=self.n_way)
    
    def sample_graphs(self, classes: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Sample K + Q graphs
        
        :param: N classes
        :return: a tuple with (support graphs, support labels, query graphs, query labels)
        """
        support_graphs, support_labels = [], []
        query_graphs,   query_labels   = [], []
        
        for index, label in enumerate(classes):
            graphs = self.labels[label]
            assert len(graphs) >= self.k_shot + self.n_query, "Not enough graphs to sample"
            
            selected_graphs = random.sample(graphs, k=self.k_shot + self.n_query)
            support_graphs.extend(selected_graphs[0:self.k_shot])
            query_graphs.extend(selected_graphs[self.k_shot:])
            
            support_labels.extend([index] * self.k_shot)
            query_labels.extend([index] * self.n_query)
        
        support_perm   = torch.randperm(len(support_graphs))
        support_graphs = torch.tensor(support_graphs)[support_perm].tolist()
        support_labels = torch.tensor(support_labels)[support_perm].tolist()
        
        query_perm   = torch.randperm(len(query_graphs))
        query_graphs = torch.tensor(query_graphs)[query_perm].tolist()
        query_labels = torch.tensor(query_labels)[query_perm].tolist()
        
        return support_graphs, support_labels, query_graphs, query_labels

    def get_graph_data(self, graph_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[pyg_data.Data]]:
        """
        Reorder and align graph data, i.e., node attributes,
        edge indices and graph indicators.
        
        :param: the IDs of the sampled graphs
        :return: (attributes, edge_index, graph indicators)
        """
        node_attributes = None
        edge_indices    = None
        graph_indicator = []
        data_list       = []
        node_number     = 0
        
        for index, graph_id in enumerate(graph_ids):
            graph_data = self.dataset[graph_id]
            x, edge_index, label = graph_data

            # First generate single data for the datalist
            data_list.append(pyg_data.Data(x=x, edge_index=edge_index, y=label))

            row1, row2 = edge_index.tolist()
            nodes = edge_index[0].unique(sorted=True).tolist()
            
            # Append node attributes
            if node_attributes is None:
                node_attributes = x
            else:
                node_attributes = torch.vstack((node_attributes, x))
            
            # Rename edges
            new_nodes = list(range(node_number, node_number + x.shape[0]))
            node_number = node_number + x.shape[0]
            node2new_nodes = dict(zip(nodes, new_nodes))
            row1 = torch.tensor(list(map(lambda x: node2new_nodes[x], row1)))
            row2 = torch.tensor(list(map(lambda x: node2new_nodes[x], row2)))
            edge_index = torch.vstack((row1, row2))
            
            if edge_indices is None:
                edge_indices = edge_index
            else:
                edge_indices = torch.hstack((edge_indices, edge_index))
            
            # Fill graph indicators
            graph_indicator.extend([index] * x.shape[0])
        
        return node_attributes, edge_indices, torch.tensor(graph_indicator), data_list
    
    def sample_episode(self) -> Tuple[pyg_data.Data, List[pyg_data.Data], pyg_data.Data, List[pyg_data.Data]]:
        """Sample a pair (support, query) set"""
        
        # 1. First sample classes
        classes = self.sample_class()
        
        # 2. Sample support and query ids
        support_graphs, support_labels, query_graphs, query_labels = self.sample_graphs(classes)
        
        # 3. Get support graph data
        support_data = self.get_graph_data(support_graphs)
        support_node_attributes, support_edge_indices, support_graph_indicators, support_data_list = support_data
        support_data = pyg_data.Data(
            x=support_node_attributes,
            edge_index=support_edge_indices,
            y=torch.tensor(support_labels).long(),
            batch=support_graph_indicators
        )
        
        # 4. Get query graph data
        query_data = self.get_graph_data(query_graphs)
        query_node_attributes, query_edge_indices, query_graph_indicators, query_data_list = query_data
        query_data = pyg_data.Data(
            x=query_node_attributes,
            edge_index=query_edge_indices,
            y=torch.tensor(query_labels).long(),
            batch=query_graph_indicators
        )
        
        return support_data, support_data_list, query_data, query_data_list
    
    def load_function(self, iter_idx: int) -> Tuple[pyg_data.Data, pyg_data.Data]:
        support_data, support_data_list, query_data, query_data_list = self.sample_episode()
        return support_data, support_data_list, query_data, query_data_list
    
    def get_iterator(self, epoch: int=0):
        setup_seed(epoch)
        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size), load=self.load_function)
        return tnt_dataset.parallel(
            batch_size=self.batch_size,
            collate_fn=coll,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def __call__(self, epoch: int):
        return self.get_iterator(epoch)
    
    def __len__(self) -> int:
        return int( self.epoch_size / self.batch_size )


class GraphDataLoader(DataLoader):
    """Custom simple DataLoader"""
    def __init__(self, dataset     : GraphDataset | OHGraphDataset,
                       batch_size  : int=1,
                       shuffle     : bool=True,
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None,
                       **kwargs
    ) -> None:

        if 'collate_fn' in kwargs:
            del kwargs["collate_fn"]

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.oh_labels = False
        if isinstance(OHGraphDataset, dataset):
            self.oh_labels = True

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=partial(_collate, oh_labels=self.oh_labels),
            **kwargs,
        )

    def __iter__(self) -> Iterator[Tuple[pyg_data.Data, List[pyg_data.Data]]]:
        for data in super().__iter__():
            data1, data2 = data
            yield data1, data2


def get_dataloader(
    ds: GraphDataset | OHGraphDataset, n_way: int, k_shot: int, n_query: int, 
    epoch_size: int, shuffle: bool, batch_size: int, exclude_keys: Optional[List[str]] = None,
    oh_labels: bool=False, dl_type: FewShotDataLoader | GraphDataLoader=FewShotDataLoader
) -> FewShotDataLoader:
    """Return a dataloader instance"""
    if dl_type.__name__ == FewShotDataLoader.__name__:
        # return FewShotDataLoader(
        #     dataset=ds,
        #     exclude_keys=exclude_keys,
        #     batch_sampler=TaskBatchSampler(
        #         dataset_targets=ds.get_graphs_per_label(),
        #         n_way=n_way, k_shot=k_shot, n_query=n_query,
        #         epoch_size=epoch_size, shuffle=shuffle,
        #         batch_size=batch_size, oh_labels=oh_labels
        #     )
        # )

        return FewShotDataLoader(
            dataset=ds, n_way=n_way, k_shot=k_shot, n_query=n_query,
            epoch_size=epoch_size, batch_size=batch_size
        )
    
    return GraphDataLoader(ds, batch_size=batch_size)