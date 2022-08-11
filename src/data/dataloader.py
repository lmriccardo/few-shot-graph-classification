import torch
import torch_geometric.loader as gloader

from typing import List, Generic, Optional
from config import T
from data.dataset import GraphDataset


class GraphCollater(gloader.dataloader.Collater):
    def __init__(self, *args) -> None:
        super(GraphCollater, self).__init__(*args)

    def __call__(self, batch: Generic[T]) -> Generic[T]:
        elem = batch[0]
        if isinstance(elem, GraphDataset):
            return self(elem)

        return super(GraphCollater, self).__call__(batch)


class FewShotDataLoader(torch.utils.data.DataLoader):
    """Custom DataLoader for GraphDataset"""

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
            support_batch, query_batch = self.batch_sampler.create_batches_from_data_batch(x)
            yield support_batch, query_batch
