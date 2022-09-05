from typing import List, Generic, Optional
from config import T
from torch.utils.data import DataLoader
import torch_geometric.loader as gloader
from data.dataset import GraphDataset
from data.sampler import TaskBatchSampler
from utils.utils import data_batch_collate, rename_edge_indexes


class GraphCollater(gloader.dataloader.Collater):
    def __init__(self, *args) -> None:
        super(GraphCollater, self).__init__(*args)

    def __call__(self, batch: Generic[T]) -> Generic[T]:
        elem = batch[0]
        if isinstance(elem, GraphDataset):
            return self(elem)

        return super(GraphCollater, self).__call__(batch)
        

class FewShotDataLoader(DataLoader):
    """Custom few-shot sampler DataLoader for GraphDataset"""

    def __init__(self, dataset: GraphDataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 follow_batch: Optional[List[str]] = None,
                 exclude_keys: Optional[List[str]] = None,
                 **kwargs) -> None:

        if 'collate_fn' in kwargs:
            del kwargs["collate_fn"]

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        # Take the batch sampler
        self.batch_sampler = kwargs["batch_sampler"]

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=GraphCollater(follow_batch, exclude_keys),
            **kwargs,
        )

    def __iter__(self):
        for x in super().__iter__():
            support_batch, support_data_list, query_batch, query_data_list = self.batch_sampler.uncollate(x)
            yield support_batch, support_data_list, query_batch, query_data_list


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
            collate_fn=GraphCollater(follow_batch, exclude_keys),
            **kwargs,
        )
    
    def __iter__(self):
        for x in super().__iter__():
            sample, sample_list = data_batch_collate(
                rename_edge_indexes(
                    x.to_data_list()
                )
            )
            yield sample, sample_list


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