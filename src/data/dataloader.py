import torch
import torch.nn.functional as F
import torch_geometric.data as pyg_data
import torchnet as tnt

from torch.utils.data import DataLoader
from data.sampler import TaskBatchSampler
from data.dataset import GraphDataset, OHGraphDataset
from utils.utils import setup_seed, data_batch_collate, rename_edge_indexes
from config import T
from typing import Tuple, List, Generic, Optional
from functools import partial
from collections.abc import Iterable


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


class FewShotDataLoader(object):
    """ DataLoader documentation """
    def __init__(self, dataset: GraphDataset | OHGraphDataset, **kwargs) -> None:
        self.dataset = dataset
        self.num_workers = 4
        self.setattr(**kwargs)

        # Configure the task batch sampler
        self.oh_labels = isinstance(self.dataset, OHGraphDataset)
        self.task_sampler = self._configure_sampler()

    def setattr(self, **attrs) -> None:
        for name, value in attrs.items():
            self.__setattr__(name, value)

    def _configure_sampler(self) -> TaskBatchSampler:
        """ Return a TaskBatchSampler instance """
        tb_sampler = TaskBatchSampler(
            self.dataset.get_graphs_per_label(), 
            self.batch_size, self.n_way, self.k_shot, 
            self.n_query, self.epoch_size, 
            shuffle=self.shuffle, oh_labels=self.oh_labels
        )

        return tb_sampler

    def _collate_graphdata(self, graph_ids: List[int]) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[pyg_data.Data]
    ]:
        """
        Rename and mix edges, nodes and labels of different graphs

        :param graph_ids: the IDs of the sampled graphs
        :return: (attributes, edge indices, graph indicators, list of Data)
        """
        node_attributes = None
        edge_indices    = None
        graph_indicator = []
        data_list       = []
        labels          = []
        node_number     = 0

        for index, graph_id in enumerate(graph_ids):
            graph_data = self.dataset[graph_id]
            x, edge_index, label = graph_data

            # Insert the label
            labels.append(label.item())

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
        
        # Label remapping
        if not self.oh_labels:
            label_mapping = dict(zip(sorted(set(labels)), range(len(set(labels)))))
            labels = list(map(lambda x: label_mapping[x], labels))
        
        return (
            node_attributes, edge_indices, torch.Tensor(labels).long(),
            torch.tensor(graph_indicator), data_list
        )


    def _sample_episode(self) -> Tuple[
            pyg_data.Data, List[pyg_data.Data], pyg_data.Data, List[pyg_data.Data]
    ]:
        """ Sample a pair of support and query set """
        # 1. First sample graph indices
        support_ids, query_ids = uncollate(self.task_sampler, next(iter(self.task_sampler)))

        # 2. Get support graph data
        support_data = self._collate_graphdata(support_ids)
        support_na, support_ei, support_y, support_gi, support_dl = support_data
        support_data = pyg_data.Data(
            x=support_na, edge_index=support_ei,
            y=support_y, batch=support_gi
        )

        # 3. Get query graph data
        query_data = self._collate_graphdata(query_ids)
        query_na, query_ei, query_y, query_gi, query_dl = query_data
        query_data = pyg_data.Data(
            x=query_na, edge_index=query_ei,
            y=query_y, batch=query_gi
        )

        return support_data, support_dl, query_data, query_dl

    def _load_function(self, idx: int) -> Tuple[
            pyg_data.Data, List[pyg_data.Data], pyg_data.Data, List[pyg_data.Data]
    ]:
        """ Load function for the torchnet dataloader """
        support_data, support_dl, query_data, query_dl = self._sample_episode()
        return support_data, support_dl, query_data, query_dl

    @staticmethod
    def _coll(batch):
        return batch[0]

    def _get_iterator(self, epoch: int=0) -> DataLoader:
        """ Return a Torch DataLoader for the graph data """
        # Setup the seed
        setup_seed(epoch)

        # Create a new dataset using torchnet
        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=self._load_function
        )

        # Finally return the dataloader
        return tnt_dataset.parallel(
            batch_size=self.batch_size,
            collate_fn=FewShotDataLoader._coll,
            num_workers=self.num_workers,
            shuffle=self.shuffle
        )
    
    def __call__(self, epoch: int=0) -> DataLoader:
        return self._get_iterator(epoch)


class GraphDataLoader(DataLoader):
    """ Custom simple DataLoader """
    def __init__(self, dataset     : GraphDataset | OHGraphDataset,
                       batch_size  : int=1,
                       shuffle     : bool=True,
                       follow_batch: Optional[List[str]]=None,
                       exclude_keys: Optional[List[str]]=None,
                       **kwargs
    ) -> None:

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.oh_labels = isinstance(dataset, OHGraphDataset)

        super(GraphDataLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=partial(_collate, oh_labels=self.oh_labels), 
            **kwargs
        )

    def __iter__(self) -> Iterable[Tuple[pyg_data.Data, List[pyg_data.Data]]]:
        for data in super().__iter__():
            data1, data2 = data
            yield data1, data2

    def __call__(self, epoch: int) -> 'GraphDataLoader':
        return self.__iter__()


def get_dataloader(
    ds: GraphDataset | OHGraphDataset, n_way: int, k_shot: int, 
    n_query: int, epoch_size: int, shuffle: bool, batch_size: int,
    dl_type: FewShotDataLoader | GraphDataLoader=FewShotDataLoader
) -> FewShotDataLoader | GraphDataLoader:
    """ Return a dataloader instance """
    if dl_type.__name__ == FewShotDataLoader.__name__:
        return FewShotDataLoader(
            dataset=ds, n_way=n_way, k_shot=k_shot, n_query=n_query,
            epoch_size=epoch_size, batch_size=batch_size, shuffle=shuffle
        )

    return GraphDataLoader(ds, batch_size=batch_size, shuffle=shuffle)