"""From the paper https://arxiv.org/pdf/2010.09891.pdf"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch_geometric.data import Data
from utils.utils import compute_accuracy, data_batch_collate_edge_renamed
from typing import Union, Tuple, Callable
from functools import wraps


class FlagGDA:
    """
    Free-Large Scale Adversarial Augmentation on Graphs (FLAG) is a 
    data augmentationa technique developed based on the adversarial
    perturbation. While most of the perturbation are directed towards
    edges or nodes, like edge/node dropping/adding, FLAG perturbation
    are directed to node features. This perturbations are generated
    by gradient-based robust optimization techniques. The "Free" stands
    for the fact that this can be used independently from the task goal:
    graph classification, node classification or link prediction. 

    Moreover, it is the first general-purpose feature-based data 
    augmentation method on graph data, which is complementary to other
    regularizers and topological augmentation. 
    """
    @staticmethod
    def flag(model      : nn.Module, 
             input_data : Tuple[Data, Data],
             iterations : int,
             step_size  : float,
             use        : bool=True,
             oh_labels  : bool=False,
             device     : str="cpu") -> Callable:
        """
        Implement the FLAG algorithm as described in the paper and more
        precisely in the source code, which can be found at the following
        link on GitHub: https://github.com/devnkong/FLAG/blob/main/ogb/attacks.py

        :param gnn: a GNN model (pretrained or not)
        :param input_data: the input data (support, query set)
        :param criterion: the loss function
        :param iterations: number of iterations
        :param step_size: the step size for improving the perturbation
        :param use: True if FLAG should be executed, False otherwise
        :param oh_labels: True if the dataset targets are one-hot encoded

        :return: the average loss, the average accuracy and the last loss
        """
        def _flag(func):

            @wraps(func)
            def wrapper(*args, **kwargs) -> Tuple[float, float, nn.Module, torch.Tensor]:
                accs, step, final_loss, total_loss = func(*args, **kwargs)

                if use:
                    avg_acc, avg_loss = [], []
                    avg_acc.append(accs[step])
                    avg_loss.append(final_loss)
                    support_, query_ = input_data

                    for _ in range(iterations):
                        support_perturbation = torch.FloatTensor(*support_.x.shape).uniform_(-step_size, step_size).to(device)
                        support_perturbation.requires_grad_()
                        support_data = Data(x=support_.x + support_perturbation, 
                                            edge_index=support_.edge_index, 
                                            y=support_.y, 
                                            batch=support_.batch
                        )
                        support_data.to(device)

                        query_perturbation = torch.FloatTensor(*query_.x.shape).uniform_(-step_size, step_size).to(device)
                        query_perturbation.requires_grad_()
                        query_data = Data(x=query_.x + query_perturbation, 
                                          edge_index=query_.edge_index, 
                                          y=query_.y, 
                                          batch=query_.batch
                        )
                        query_data.to(device)

                        accs, step, final_loss, new_total_loss, *_ = model(support_data, query_data)
                        avg_acc.append(accs[step])
                        avg_loss.append(final_loss)

                        total_loss += new_total_loss


                    avg_acc = sum(avg_acc) / len(avg_acc)
                    avg_loss = sum(avg_loss) / len(avg_loss)

                    return torch.tensor(avg_acc), step, torch.tensor(avg_loss), total_loss / iterations
                
                return accs, step, final_loss, total_loss

            return wrapper

        return _flag