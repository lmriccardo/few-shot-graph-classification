from os import rename
import torch
import torch_geometric.data as gdata

from utils.utils import task_sampler_uncollate

from typing import List
import random


class FewShotSampler(torch.utils.data.Sampler):
    """
    In few-shot classification, and in particular in Meta-Learning, 
    we use a specific way of sampling batches from the training/val/test 
    set. This way is called N-way-K-shot, where N is the number of classes 
    to sample per batch and K is the number of examples to sample per class 
    in the batch. The sample batch on which we train our model is also called 
    `support` set, while the one on which we test is called `query` set.

    This class is a N-way-K-shot sampler that will be used as a batch_sampler
    for the :obj:`torch_geometric.loader.DataLoader` dataloader. This sampler
    return batches of indices that correspond to support and query set batches.

    Attributes:
        labels: PyTorch tensor of the labels of the data elements
        n_way: Number of classes to sampler per batch
        k_shot: Number of examples to sampler per class in the batch
        n_query: Number of query example to sample per class in the batch
        shuffle: If True, examples and classes are shuffled at each iteration
        indices_per_class: How many indices per classes
        classes: list of all classes
        epoch_size: number of batches per epoch
    """

    def __init__(self, labels: torch.Tensor,
                 n_way: int,
                 k_shot: int,
                 n_query: int,
                 epoch_size: int,
                 shuffle: bool = True) -> None:
        super().__init__(None)
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.shuffle = shuffle
        self.epoch_size = epoch_size

        self.classes = torch.unique(self.labels).tolist()
        self.indices_per_class = dict()
        for cl in self.classes:
            self.indices_per_class[cl] = torch.where(self.labels == cl)[0]

    def shuffle_data(self) -> None:
        """
        Shuffle the examples per class

        Args:
            classes: The list of all classes
        """
        for cl in self.classes:
            perm = torch.randperm(self.indices_per_class[cl].shape[0])
            self.indices_per_class[cl] = self.indices_per_class[cl][perm]

    def __iter__(self) -> List[torch.Tensor]:
        # Shuffle the data
        if self.shuffle:
            self.shuffle_data()

        target_classes = random.sample(self.classes, self.n_way)
        for _ in range(self.epoch_size):
            n_way_k_shot_n_query = []
            for cl in target_classes:
                labels_per_class = self.indices_per_class[cl]
                assert len(labels_per_class) >= self.k_shot + self.n_query
                selected_data = random.sample(
                    labels_per_class.tolist(), self.k_shot + self.n_query)
                n_way_k_shot_n_query.append(selected_data)

            yield torch.tensor(n_way_k_shot_n_query)

    def __len__(self) -> int:
        return self.epoch_size


class TaskBatchSampler(torch.utils.data.Sampler):
    """Sample a batch of tasks"""

    def __init__(self, dataset_targets: torch.Tensor,
                 batch_size: int,
                 n_way: int,
                 k_shot: int,
                 n_query: int,
                 epoch_size: int,
                 shuffle: bool = True) -> None:

        super().__init__(None)
        self.task_sampler = FewShotSampler(
            dataset_targets,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            epoch_size=epoch_size,
            shuffle=shuffle
        )

        self.task_batch_size = batch_size

    def __iter__(self):
        mini_batches = []
        for task_idx, task in enumerate(self.task_sampler):
            mini_batches.extend(task.tolist())
            if (task_idx + 1) % self.task_batch_size == 0:
                yield torch.tensor(mini_batches).flatten().tolist()
                mini_batches = []

    def __len__(self):
        return len(self.task_sampler) // self.task_batch_size

    def uncollate(self, data_batch):
        """Invoke the uncollate from utils.utils"""
        return task_sampler_uncollate(self, data_batch)
