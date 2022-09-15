import torch

from utils.utils import task_sampler_uncollate

from typing import List, Dict
from copy import deepcopy
import random


# class FewShotSampler(torch.utils.data.Sampler):
#     """
#     In few-shot classification, and in particular in Meta-Learning, 
#     we use a specific way of sampling batches from the training/val/test 
#     set. This way is called N-way-K-shot, where N is the number of classes 
#     to sample per batch and K is the number of examples to sample per class 
#     in the batch. The sample batch on which we train our model is also called 
#     `support` set, while the one on which we test is called `query` set.

#     This class is a N-way-K-shot sampler that will be used as a batch_sampler
#     for the :obj:`torch_geometric.loader.DataLoader` dataloader. This sampler
#     return batches of indices that correspond to support and query set batches.

#     Attributes:
#         labels: PyTorch tensor of the labels of the data elements
#         n_way: Number of classes to sampler per batch
#         k_shot: Number of examples to sampler per class in the batch
#         n_query: Number of query example to sample per class in the batch
#         shuffle: If True, examples and classes are shuffled at each iteration
#         indices_per_class: How many indices per classes
#         classes: list of all classes
#         epoch_size: number of batches per epoch
#     """

#     def __init__(self, labels: torch.Tensor,
#                  n_way: int,
#                  k_shot: int,
#                  n_query: int,
#                  epoch_size: int,
#                  shuffle: bool = True) -> None:
#         super().__init__(None)
#         self.labels = labels
#         self.n_way = n_way
#         self.k_shot = k_shot
#         self.n_query = n_query
#         self.shuffle = shuffle
#         self.epoch_size = epoch_size

#         self.classes = torch.unique(self.labels).tolist()
#         self.indices_per_class = dict()
#         for cl in self.classes:
#             self.indices_per_class[cl] = torch.where(self.labels == cl)[0]

#     def shuffle_data(self) -> None:
#         """
#         Shuffle the examples per class

#         Args:
#             classes: The list of all classes
#         """
#         for cl in self.classes:
#             perm = torch.randperm(self.indices_per_class[cl].shape[0])
#             self.indices_per_class[cl] = self.indices_per_class[cl][perm]

#     def __iter__(self) -> List[torch.Tensor]:
#         # Shuffle the data
#         if self.shuffle:
#             self.shuffle_data()

#         target_classes = random.sample(self.classes, self.n_way)
#         for _ in range(self.epoch_size):
#             n_way_k_shot_n_query = []
#             for cl in target_classes:
#                 labels_per_class = self.indices_per_class[cl]
#                 assert len(labels_per_class) >= self.k_shot + self.n_query
#                 selected_data = random.sample(labels_per_class.tolist(), self.k_shot + self.n_query)
#                 n_way_k_shot_n_query.append(selected_data)

#             yield torch.tensor(n_way_k_shot_n_query)

#     def __len__(self) -> int:
#         return self.epoch_size


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
        oh_labels: True, if the labels are one-hot encoded
    """
    def __init__(
            self, labels: Dict[int | str, List[int]], n_way: int, 
            k_shot: int, n_query: int, epoch_size: int, 
            shuffle: bool=True, oh_labels: bool=False
    ) -> None:       
        super(FewShotSampler, self).__init__(None)
        
        # "labels" is a dictionary (label, list of graphs indices
        # with that label). If a graph is contained in multiple 
        # list for different labels, that's mean its one-hot
        # encoded label is something like [..., x1, ..., xn, ...]
        # such that x1 + ... + xn = 1.0. 
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.shuffle = shuffle
        self.epoch_size = epoch_size
        self.oh_labels = oh_labels
        
        # Compute in how many classes a graph is
        self.num_per_class = None
        if self.oh_labels:
            self.num_per_class = self._number_per_labels()
        
        # Shuffle the data
        if self.shuffle:
            self._shuffle_data()
            
    def _number_per_labels(self) -> Dict[int, int]:
        """
        Compute for each graph the lenght of the
        one-hot encoded label. This means that if
        the returned dictionary containes per a graph
        g (g, (1, [...])) then its label is [..., x, ...], 
        or if (g, (2, [...])) then it is [..., x, ..., y, ...]
        and so on, up to N.
        
        :return: a dictionary
        """
        num_per_class = dict()
        for label, graph_ids in self.labels.items():
            for graph_id in graph_ids:
                if graph_id not in num_per_class:
                    num_per_class[graph_id] = [0, []]
                
                num_per_class[graph_id][0] += 1
                num_per_class[graph_id][1].append(label)
                
        return num_per_class
        
    def _shuffle_data(self) -> None:
        """Shuffle the data"""
        for cl in self.labels:
            perm = torch.randperm(len(self.labels[cl]))
            self.labels[cl] = torch.tensor(self.labels[cl])[perm].tolist()

    def _clear(self, classes: List[int], target_graphs: List[int]) -> List[int]:
        """
        Clear the list of target graphs with a particular class. What does it 
        mean clearing? First of all this method is used only in case one-hot
        encoded labels are used, in the other case it does not make sense. So,
        when sampling K + Q graphs for one of the N classes, in few-shot sampling,
        we need to make sure that all the graphs inside the K + Q graphs satisfy
        certain constraints. Let's take a data (x, y) where y is a one-hot encoded
        label, and assume Y = {y1,...,yN} are the sampled classes (yi is a int). 
        Then, x can be added to the K + Q graphs iff the following constraints are 
        satisfied:

                    [(E! i in Y | y[i] = 1) OR
                     (E! I = (i1, ..., iN) in (Y^n \ {(i)^N | i in Y}) 
                            | y[i1] + ... + y[iN] = 1 )] AND 
                    (not E i not in Y | y[i] != 0)
        
        :param classes: the sampled N classes
        :param target_graphs: the list of all graphs with label in class
        :return: cleared target_graphs
        """
        new_target_graphs = []
        cls = deepcopy(classes)
        cls = sorted(cls)
        
        for graph_id in target_graphs:
            if self.num_per_class[graph_id][0] == 1:
                new_target_graphs.append(graph_id)
                continue
                
            if self.num_per_class[graph_id][0] > len(classes):
                continue
            
            graph_cls = self.num_per_class[graph_id][1]
            is_ok = all(list(map(lambda x: x in cls, graph_cls)))
            if is_ok:
                new_target_graphs.append(graph_id)
                continue
        
        return new_target_graphs
        
    def __iter__(self) -> List[torch.Tensor]:
        # Take the target classes
        target_classes = random.sample(list(self.labels.keys()), k=self.n_way)
        for _ in range(self.epoch_size):

            n_way_k_shot_query = [[], []]
            for cl in target_classes:
                
                graphs = self.labels[cl]
                if self.oh_labels:
                    graphs = self._clear(target_classes, self.labels[cl])
                
                assert len(graphs) >= self.k_shot + self.n_query, "Not enough graphs for sampling"
                selected_data = random.sample(graphs, k=self.k_shot + self.n_query)
                n_way_k_shot_query[0] += selected_data[:self.k_shot]
                n_way_k_shot_query[1] += selected_data[self.k_shot:]
                # n_way_k_shot_query.append(selected_data)
            
            support_ = torch.tensor(n_way_k_shot_query[0])
            support_perm = torch.randperm(support_.shape[0])
            support_ = support_[support_perm]

            query_ = torch.tensor(n_way_k_shot_query[1])
            query_perm = torch.randperm(query_.shape[0])
            query_ = query_[query_perm]
            
            yield torch.hstack((support_, query_))
        
    def __len__(self) -> int:
        return self.epoch_size


class TaskBatchSampler(torch.utils.data.Sampler):
    """
    TaskBatchSampler is a multi-batch few-shot sampler.
    Each iterations it returns multiple N * (K + Q) graphs
    collated into a single batch given a precise batch_size.

    Args
        dataset_targets (Dict[int | str, List[int]]): a python dictionary
            with keys all the labels of the dataset, and values for each
            label, those graphs that should be classified with that target class
        batch_size (int): the size of the batch
        n_way (int): how many classes to sample for a single few-shot sample
        k_shot (int): the dimension of the support set
        n_query (int): the dimension of the query set
        epoch_size (int): number of batches per epoch
        shuffle (bool): True, if we want to shuffle the dataset before sampling
        oh_labels (bool): True, if labels are one-hot encoded
    """
    def __init__(
        self, dataset_targets: Dict[int | str, List[int]], batch_size: int, n_way: int,
        k_shot: int, n_query: int, epoch_size: int, shuffle: bool = True, oh_labels: bool=False
    ) -> None:
        super(TaskBatchSampler, self).__init__(None)
        self.task_sampler = FewShotSampler(
            dataset_targets,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            epoch_size=epoch_size,
            shuffle=shuffle,
            oh_labels=oh_labels
        )

        self.task_batch_size = batch_size
        self.oh_labels = oh_labels

    def __iter__(self):
        mini_batches = []
        for task_idx, task in enumerate(self.task_sampler):
            mini_batches.extend(task.tolist())
            if (task_idx + 1) % self.task_batch_size == 0:
                yield torch.tensor(mini_batches).flatten().tolist()
                mini_batches = []

    def __len__(self):
        return len(self.task_sampler) // self.task_batch_size
