"""From the paper https://arxiv.org/pdf/2007.05700.pdf"""
import torch
import torch.nn as nn

from torch_geometric.data import Data

from data.dataset import GraphDataset, augment_dataset
from utils.utils import rename_edge_indexes, graph2data
from utils.trainers import BaseTrainer
from typing import Dict, List, Tuple

import config
import networkx as nx
import logging


class MEvolveGDA:
    """
    The Model Evolution framework presented in the paper

    **NOTE**: the train and validation set must have common classes
              however, we are not able to compute data filtering.
              More specifically, the confusion matrix based on the
              results given by the validation set.
    
    Args
        trainer (BaseTrainer): the trainer used to train the model 
        n_iters (int): number of iterations (or evolutions)
        train_ds (GraphDataset): the training dataset
        val_ds (GraphDataset): the validation dataset
        pre_trained_model (nn.Module): the pre-trained classifier
    """
    def __init__(self, 
        trainer: BaseTrainer, n_iters: int, logger: logging.Logger,
        train_ds: GraphDataset, validation_ds: GraphDataset
    ) -> None:
        self.trainer = trainer
        self.n_iters = n_iters
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.logger = logger
        self.pre_trained_model = self.trainer.model
 
    @staticmethod
    def compute_threshold(graph_probability_vector: Dict[int, torch.Tensor],
                          data_list               : List[Data],
                          label_reliabilities     : Dict[Data, float]) -> float:
        """
        Compute the threshold for the label reliability acceptation.
        This threshold is the result of the following expression

            t = arg min_t sum(T[(t - ri) * g(G_i, y_i)], (G_i, y_i) in D_val)
        
        Thus, we have to compute the derivative equal to 0

        :param graph_probability_vector: the probability vector for each graph
        :param data_list: a list of graph data
        :param label_reliabilities: label reliability value for each graph
        :return: the optimal threshold
        """
        def g_function() -> torch.Tensor:
            """Compute the function g(G, y) = 1 if C(G) = y else -1 where C is the classifier"""
            graph_pred, y = [], []
            for graph, pred_vector in graph_probability_vector.items():
                y.append(data_list[graph].y.item())
                graph_pred.append(torch.argmax(pred_vector).item())
            
            graph_pred = torch.tensor(graph_pred)
            y = torch.tensor(y)

            diff = (graph_pred - y == 0)
            return torch.tensor(list(map(lambda x: 1 if x else -1, diff.tolist())))

        def phi(theta: torch.Tensor, ri: torch.Tensor, g_values: torch.Tensor) -> torch.Tensor:
            """Compute the function Phi[x] = max(0, sign(x))"""
            mul = (theta - ri) * g_values

            # In this case I decided to use a differentiable 
            # approximation of the sign function with respect
            # to the variable Theta (which we have to differentiate)
            # The chosen approximation is the tanh function using a 
            # value for beta >> 1. 
            sign_approximation = torch.tanh(config.LABEL_REL_THRESHOLD_BETA * mul)
            zero = torch.zeros((sign_approximation.shape[0],))
            return torch.maximum(zero, sign_approximation)

        theta = torch.rand((1,), requires_grad=True)
        total_ri = torch.tensor(list(label_reliabilities.values()), dtype=torch.float)
        g_values = g_function()
        
        current_step = 0
        current_minimum = float('inf')
        while current_step < config.LABEL_REL_THRESHOLD_STEPS and theta.item() != 0.0:
            f = phi(theta, total_ri, g_values).sum()
            f.backward()

            with torch.no_grad():
                theta = theta - config.LABEL_REL_THRESHOLD_STEP_SIZE * theta.grad

            theta.requires_grad = True
            current_step += 1

            if f.item() < current_minimum:
                current_minimum = f.item()
        
        return theta.item()

    @staticmethod
    def data_filtering(validation_ds           : GraphDataset,
                       graph_probability_vector: Dict[int, torch.Tensor],
                       data_list               : List[Data],
                       classes                 : List[int],
                       classifier_model        : torch.nn.Module,
                       logger                  : logging.Logger,
                       augmented_data          : List[Tuple[nx.Graph, str]]) -> List[Tuple[nx.Graph, str]]:
        """
        After applying the heuristic for data augmentation, we have a
        bunch of new graphs and respective labels that needs to be
        added to the training set. Before this, we have to filter this data
        by label reliability. For further information look at the paper
        https://arxiv.org/pdf/2007.05700.pdf by Zhou et al.
        
        :param validation_ds: the validation dataset
        :param classes: the list of targets label
        :param classifier_model: the classifier
        :param logger: a simple logger
        :param augmented_data: the new data generated from the train set
        :param data_list: a list of graphs
        :param graph_probability_vector: a dictionary mapping for each graph the
                                        probability vector obtained after running
                                        the pre-trained classifier.

        :return: the list of graphs and labels that are reliable to be added
        """
        count_per_labels = validation_ds.count_per_class
        

        # Compute the confusion matrix Q
        n_classes = len(classes)
        classes_mapping = dict(zip(classes, range(n_classes)))
        confusion_matrix = torch.zeros((n_classes, n_classes))
        for idx, target in enumerate(classes):
        
            # Get graphs with a specific target
            prob_vector_target = [v.tolist() for g, v in graph_probability_vector.items() if data_list[g].y.item() == target]
            prob_vector_tensor = torch.tensor(prob_vector_target)
            confusion_matrix[idx] = 1 / count_per_labels[target] *  prob_vector_tensor.sum(dim=0)
        
        # Now, compute the label reliability for all graphs in the validation set
        label_reliabilities = dict()
        for graph, prob_vect in graph_probability_vector.items():
            label = data_list[graph].y.item()
            label_idx = classes.index(label)
            label_reliabilities[graph] = prob_vect @ confusion_matrix[label_idx]
        
        # Compute the label reliability threshold theta
        label_rel_threshold = MEvolveGDA.compute_threshold(graph_probability_vector, data_list, label_reliabilities)
        logger.debug(f"Computed new label reliability threshold to {label_rel_threshold}")

        # Filter data
        filtered_data = []
        for graph, target in augmented_data:
            geotorch_data = rename_edge_indexes([graph2data(graph, target)])[0]
            prob_vector, _, _ = classifier_model(geotorch_data.x, geotorch_data.edge_index, geotorch_data.batch)
            r = prob_vector @ confusion_matrix[classes_mapping[target]]
            if r > label_rel_threshold:
                filtered_data.append((graph, target))
        
        return filtered_data

    def evolve(self) -> nn.Module:
        """Run the evolution algorithm and return the final classifier"""
        current_iteration = 0
        while current_iteration < self.n_iters:
            # 1. Get augmented Data
            d_pool = augment_dataset(self.train_ds, heuristic=config.HEURISTIC)

            # 2. Compute the graph probability vector
            validation_data, validation_data_list = self.validation_ds.to_data()
            pred, _, _ = self.pre_trained_model(validation_data.x, 
                                                validation_data.edge_index,
                                                validation_data.batch
            )

            prob_vect = dict(zip(range(pred.shape[0]), pred))

            # 3. Compute data filtering
            filtered_data = self.data_filtering(
                self.validation_ds, prob_vect, validation_data_list, 
                self.train_ds.targets().tolist(), self.pre_trained_model, 
                self.logger, d_pool
            )

            # 4. Change the current train dataset of the trainer
            self.train_ds = self.train_ds + filtered_data
            self.trainer.train_ds = self.train_ds
            self.trainer.val_ds   = self.validation_ds
            self.trainer.train_dl, self.trainer.val_dl = self.trainer.get_dataloaders()

            # 5. Re run the trainer
            self.trainer.train()

            # 6. Increment the iteration counter
            current_iteration += 1
        
        return self.trainer.model