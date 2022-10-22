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
    def flag(gnn        : nn.Module, 
             criterion  : Union[_Loss, _WeightedLoss], 
             input_data : Tuple[Data, Data],
             optimizer  : optim.Optimizer,
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
                avg_acc, avg_loss = [], []
                avg_acc_train, avg_loss_train, loss = func(*args, **kwargs)
                avg_acc.append(avg_acc_train.item())
                avg_loss.append(avg_loss_train.item())

                if use:
                    # Setup model for training and zeros optimizer grads
                    gnn.train()
                    optimizer.zero_grad()

                    # Since we have the support and query set as input, 
                    # I can merge this two data into a single batch
                    # and then perturb this entire batch. 
                    support_, query_ = input_data
                    data = data_batch_collate_edge_renamed([support_, query_], oh_labels=oh_labels)
                    targets = data.y

                    perturbation = torch.FloatTensor(*data.x.shape).uniform_(-step_size, step_size).to(device)
                    perturbation.requires_grad_()
                    logits, _, _ = gnn(data.x + perturbation, data.edge_index, data.batch)
                    loss = criterion(logits, targets)
                    loss = loss / iterations

                    for _ in range(iterations):
                        with torch.no_grad():
                            preds = F.softmax(logits, dim=1).argmax(dim=1)
                            avg_acc.append(compute_accuracy(preds, targets, oh_labels))
                            avg_loss.append(loss.item())
                        
                        loss.backward()

                        perturb_data = perturbation.detach() + step_size * torch.sign(perturbation.grad.detach())
                        perturbation.data = perturb_data.data
                        perturbation.grad[:] = 0

                        logits, _, _ = gnn(data.x + perturbation, data.edge_index, data.batch)
                        loss = criterion(logits, targets)
                        loss = loss / iterations
                    
                    with torch.no_grad():
                        preds = F.softmax(logits, dim=1).argmax(dim=1)
                        avg_acc.append(compute_accuracy(preds, targets, oh_labels))
                        avg_loss.append(loss)
                        
                    loss.backward()
                    optimizer.step()

                avg_acc = sum(avg_acc) / len(avg_acc)
                avg_loss = sum(avg_loss) / len(avg_loss)

                return torch.tensor(avg_acc), torch.tensor(avg_loss), loss

            return wrapper

        return _flag