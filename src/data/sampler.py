import torch
import torch_geometric.data as gdata

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

    def create_batches_from_data_batch(self, data_batch):
        """
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
        """
        n_way = self.task_sampler.n_way
        k_shot = self.task_sampler.k_shot
        n_query = self.task_sampler.n_query

        total_support_query_number = n_way * (k_shot + n_query)
        support_plus_query = k_shot + n_query

        # Initialize batch list for support and query set
        support_data_batch = []
        query_data_batch = []

        # I know how many batch do I have, so
        for batch_number in range(self.task_batch_size):

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

        # Create new DataBatchs and return
        return gdata.Batch.from_data_list(support_data_batch), gdata.Batch.from_data_list(query_data_batch)
