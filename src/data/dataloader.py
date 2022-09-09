import torch_geometric.data as pyg_data

from torch.utils.data import DataLoader
from utils.utils import data_batch_collate, rename_edge_indexes, task_sampler_uncollate
from data.sampler import TaskBatchSampler
from data.dataset import GraphDataset
from typing import Generic, Tuple, List, Optional, Iterator
from config import T
from functools import partial


def _collate(_batch: Generic[T]) -> Tuple[pyg_data.Data, List[pyg_data.Data]]:
    data_list = []

    for element in _batch:
        element_x = element[0]
        element_edge_index = element[1]
        element_label = element[2]

        # Create single data for datalist
        single_data = pyg_data.Data(x=element_x, edge_index=element_edge_index, y=element_label)
        data_list.append(single_data)

    return data_batch_collate(rename_edge_indexes(data_list))


def graph_collator(batch: Generic[T], sampler: TaskBatchSampler) -> Generic[T]:
    """collator"""
    support_data_batch, query_data_batch = task_sampler_uncollate(sampler, batch)
    return _collate(support_data_batch), _collate(query_data_batch)


class FewShotDataLoader(DataLoader):
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
    
    def __iter__(self) -> Iterator[Tuple[
        pyg_data.Data, List[pyg_data.Data], pyg_data.Data, List[pyg_data.Data]
    ]]:
        for data in super().__iter__():
            (data1, data2), (data3, data4) = data
            yield data1, data2, data3, data4



class GraphDataLoader(DataLoader):
    """Custom simple DataLoader"""
    def __init__(self, dataset     : GraphDataset,
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

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=_collate,
            **kwargs,
        )

    def __iter__(self) -> Iterator[Tuple[pyg_data.Data, List[pyg_data.Data]]]:
        for data in super().__iter__():
            data1, data2 = data
            yield data1, data2


def get_dataloader(
    ds: GraphDataset, n_way: int, k_shot: int, n_query: int, 
    epoch_size: int, shuffle: bool, batch_size: int, 
    exclude_keys: Optional[List[str]] = None
) -> FewShotDataLoader:
    """Return a dataloader instance"""
    return FewShotDataLoader(
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