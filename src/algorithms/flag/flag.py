"""From the paper https://arxiv.org/pdf/2010.09891.pdf"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch_geometric.data import Data
from utils.utils import compute_accuracy
from typing import Union, Tuple, Callable, Any, List, Dict

import config
import wrapt


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
             data       : Data,
             optimizer  : optim.Optimizer,
             targets    : torch.Tensor,
             iterations : int,
             step_size  : float,
             use        : bool=True,
             oh_labels  : bool=False) -> Callable[
                [Callable[[Any], Any], List[Any], Dict[Any, Any]], 
                Tuple[float, float, nn.Module, torch.Tensor]
    ]:
        """
        Implement the FLAG algorithm as described in the paper and more
        precisely in the source code, which can be found at the following
        link on GitHub: https://github.com/devnkong/FLAG/blob/main/ogb/attacks.py

        :param gnn: a GNN model (pretrained or not)
        :param data: the input data
        :param criterion: the loss function
        :param targets: target classes
        :param iterations: number of iterations
        :param step_size: the step size for improving the perturbation
        :param use: True if FLAG should be executed, False otherwise
        :param oh_labels: True if the dataset targets are one-hot encoded

        :return: the average loss, the average accuracy, the final model and the last loss
        """
        @wrapt.decorator
        def _flag(func, *args, **kwargs) -> Tuple[float, float, nn.Module, torch.Tensor]:
            # wrapped function
            avg_acc, avg_loss = func(*args, **kwargs)
            loss = None
            
            if use:
                # Setup model for training and zeros optimizer grads
                gnn.train()
                optimizer.zero_grad()

                perturbation = torch.FloatTensor(*data.x.shape).uniform_(-step_size, step_size).to(config.DEVICE)
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

                    logits = gnn(data.x + perturbation, data.edge_index, data.batch)
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

            return avg_loss, avg_acc, gnn, loss

        return _flag