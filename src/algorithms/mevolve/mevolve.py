"""From the paper https://arxiv.org/pdf/2007.05700.pdf"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from data.dataset import GraphDataset, split_dataset
from data.dataloader import GraphDataLoader
from utils.utils import rename_edge_indexes,    \
                        to_pygdata,             \
                        cartesian_product,      \
                        build_adjacency_matrix, \
                        to_nxgraph
from typing import Dict, List, Tuple, Any
from numpy.linalg import matrix_power
from networkx.algorithms.link_prediction import resource_allocation_index
from tqdm import tqdm

import logging
import random
import config
import math
import random
import numpy as np


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
        trainer: 'utils.train.Trainer', n_iters: int, logger: logging.Logger,
        train_ds: GraphDataset, validation_ds: GraphDataset,
        threshold_beta: float, threshold_steps: int, heuristic: str="random_mapping"
    ) -> None:
        self.trainer           = trainer
        self.n_iters           = n_iters
        self.train_ds          = train_ds
        self.validation_ds     = validation_ds
        self.logger            = logger
        self.pre_trained_model = self.trainer.model
        self.threshold_beta    = threshold_beta
        self.threshold_steps   = threshold_steps
        self.heuristic         = heuristic
 
    @staticmethod
    def compute_threshold(graph_probability_vector: torch.Tensor,
                          data_list               : List[Data],
                          total_ri                : torch.Tensor,
                          lr                      : float,
                          theta                   : torch.Tensor,
                          decay                   : float,
                          threshold_beta          : float,
                          threshold_steps         : int) -> float:
        """
        Compute the threshold for the label reliability acceptation.
        This threshold is the result of the following expression

            t = arg min_t sum(T[(t - ri) * g(G_i, y_i)], (G_i, y_i) in D_val)
        
        Thus, we have to compute the derivative equal to 0

        :param graph_probability_vector: the probability vector for each graph
        :param data_list: a list of graph data
        :param label_reliabilities: label reliability value for each graph
        :param lr: learning rate for optimization problem
        :param decay: for lr adaptation
        :param theta: the parameter to optimize
        :param threshold_beta: the beta parameter for tanh approximation
        :param threshold_steps: how many step of GD to run before stopping
        :return: the optimal threshold
        """
        def g_function() -> torch.Tensor:
            """Compute the function g(G, y) = 1 if C(G) = y else -1 where C is the classifier"""
            graph_pred, y = [], []
            for graph, pred_vector in enumerate(graph_probability_vector):
                y.append(data_list[graph].y.item())
                graph_pred.append(F.softmax(pred_vector, dim=0).argmax().item())
            
            graph_pred = torch.tensor(graph_pred)
            y = torch.tensor(y)

            not_diff = (graph_pred - y == 0)
            diff     = (graph_pred - y != 0)

            result = graph_pred - y
            result[not_diff] = 1
            result[diff] = -1

            return result

        def phi(th: torch.Tensor, ri: torch.Tensor, g_values: torch.Tensor) -> torch.Tensor:
            """Compute the function Phi[x] = max(0, sign(x))"""
            mul = (th - ri) * g_values

            # In this case I decided to use a differentiable 
            # approximation of the sign function with respect
            # to the variable Theta (which we have to differentiate)
            # The chosen approximation is the tanh function using a 
            # value for beta >> 1. 
            sign_approximation = torch.tanh(threshold_beta * mul)
            zero = torch.zeros((sign_approximation.shape[0],))
            return torch.maximum(zero, sign_approximation)

        g_values = g_function()
        current_step = 0
        current_minimum = float('inf')
        while current_step < threshold_steps and theta.data.item() != 0.0:
            f = phi(theta, total_ri, g_values).sum()
            f.backward()

            with torch.no_grad():
                theta = theta - lr * theta.grad

            if f.item() < current_minimum:
                current_minimum = f.item()
            
            theta.requires_grad = True
            current_step += 1
        
        lr = lr * (1. / (1. + decay * current_step))
        
        return theta.data.item()

    @staticmethod
    def data_filtering(validation_ds           : GraphDataset,
                       graph_probability_vector: torch.Tensor,
                       data_list               : List[Data],
                       classes                 : List[int],
                       classifier_model        : torch.nn.Module,
                       augmented_data          : List[Tuple[Dict[str, Any], str]],
                       lr                      : float,
                       theta                   : torch.Tensor,
                       decay                   : float,
                       threshold_beta          : float,
                       threshold_steps         : int) -> Dict[int, Tuple[Dict[str, Any], str]]:
        """
        After applying the heuristic for data augmentation, we have a
        bunch of new graphs and respective labels that needs to be
        added to the training set. Before this, we have to filter this data
        by label reliability. For further information look at the paper
        https://arxiv.org/pdf/2007.05700.pdf by Zhou et al.
        
        :param validation_ds: the validation dataset
        :param classes: the list of targets label
        :param classifier_model: the classifier
        :param augmented_data: the new data generated from the train set
        :param data_list: a list of graphs
        :param graph_probability_vector: a dictionary mapping for each graph the
                                        probability vector obtained after running
                                        the pre-trained classifier.
        :param lr: learning rate for optimization problem
        :param decay: for lr adaptation
        :param theta: the parameter to optimize
        :param threshold_beta: the beta parameter for tanh approximation
        :param threshold_steps: how many step of GD to run before stopping

        :return: the list of graphs and labels that are reliable to be added
        """
        count_per_labels = validation_ds.count_per_class
        

        # Compute the confusion matrix Q
        n_classes = len(classes)
        classes_mapping = dict(zip(classes, range(n_classes)))
        confusion_matrix = torch.zeros((n_classes, n_classes))
        for idx, target in enumerate(classes):
            graph_indices = [idx for idx, data in enumerate(data_list) if data.y.item() == target]
            confusion_matrix[idx] = 1 / count_per_labels[target] *  graph_probability_vector[graph_indices, :].sum(dim=0)
        
        # Now, compute the label reliability for all graphs in the validation set
        label_reliabilities = []
        for graph, prob_vect in enumerate(graph_probability_vector):
            label = data_list[graph].y.item()
            label_idx = classes.index(label)
            label_reliabilities.append((prob_vect @ confusion_matrix[label_idx]).item())
        
        label_reliabilities = torch.tensor(label_reliabilities, dtype=torch.float)

        # Compute the label reliability threshold theta
        label_rel_threshold = MEvolveGDA.compute_threshold(
            graph_probability_vector, data_list, 
            label_reliabilities, lr, theta, decay,
            threshold_beta, threshold_steps
        )
        print(f"Computed new label reliability threshold to {label_rel_threshold}")

        # Filter data
        filtered_data = dict()
        counter = 0
        for graph, target in augmented_data:
            geotorch_data = rename_edge_indexes([to_pygdata(graph, target)])[0]
            geotorch_data.batch = torch.tensor([0] * geotorch_data.x.shape[0])
            prob_vector, _, _ = classifier_model(geotorch_data.x, geotorch_data.edge_index, geotorch_data.batch)
            sample_classes = random.sample([classes_mapping[x] for x in classes if x != target], k=prob_vector.shape[1] - 1)
            choices = sorted([classes_mapping[target]] + sample_classes)
            r = prob_vector[0] @ confusion_matrix[classes_mapping[target], choices]
            if r > label_rel_threshold:
                filtered_data[counter] = (graph, target)
                counter += 1
        
        return filtered_data
    
    @staticmethod
    def compute_prob_vector(val_dl: DataLoader, n_classes: int, net: nn.Module) -> Dict[int, torch.Tensor]:
        """Compute the probability vector for each graph in the validation set"""
        dataset_size = len(val_dl) * val_dl.batch_size
        preds = torch.rand((dataset_size, n_classes))
        idx = 0
        for val_data, _ in val_dl:
            classes, _ = val_data.y.sort()
            pred, _, _ = net(val_data.x, val_data.edge_index, val_data.batch)
            preds[idx : idx + val_dl.batch_size, classes] = pred
            idx += val_dl.batch_size

        return preds

    def evolve(self, train: bool=False) -> nn.Module:
        """Run the evolution algorithm and return the final classifier"""
        theta = torch.rand((1,), requires_grad=True)
        lr = 1e-2
        decay = lr / self.n_iters

        # Needed if the model is not pre-trained
        if train:
            self.trainer.train()

        current_iteration = 0
        print(f"Starting Model-Evolution Graph Data Augmentation Technique")
        print(f"USING HEURISTIC - {self.heuristic}")
        print("==========================================================")

        # For M-Evolve we need the train set and the validation set
        # sharing the same class space. For this reason I have decided
        # first to the merge the two dataset and then split them 
        # another time into train and validation set that satisfy this constraint
        self.train_ds, self.validation_ds = split_dataset(self.train_ds + self.validation_ds)

        while current_iteration < self.n_iters:
            print(f"MODEL-EVOLUTION ITERATION NUMBER: {current_iteration}/{self.n_iters}")

            # 1. Get augmented Data
            d_pool = augment_dataset(self.train_ds, heuristic=self.heuristic)

            # 2. Compute the graph probability vector
            val_dl = GraphDataLoader(self.validation_ds, batch_size=self.pre_trained_model.num_classes, drop_last=True)
            prob_matrix = self.compute_prob_vector(val_dl, self.validation_ds.number_of_classes, self.pre_trained_model)
            
            # 2.5 Create a list with all the validation graph data
            validation_data_list = []
            for _, data_list in val_dl:
                validation_data_list.extend(data_list)

            # 3. Compute data filtering
            filtered_data = self.data_filtering(
                self.validation_ds, prob_matrix, validation_data_list, 
                self.validation_ds.classes, self.pre_trained_model,
                d_pool, lr, theta, decay, self.threshold_beta, self.threshold_steps
            )

            print(f"Number of new generated data: {len(filtered_data)}")

            # 4. Change the current train dataset of the trainer
            self.train_ds = self.train_ds + filtered_data
            self.trainer.train_dl, self.trainer.val_dl = self.trainer._get_dataloaders()

            print(f"The new training set has dimension: {len(self.train_ds)}")

            # 5. Re run the trainer
            self.trainer.use_mevolve = False
            self.trainer.run()

            # 6. Increment the iteration counter
            current_iteration += 1
            print("==========================================================")
        
        return self.trainer.model


#####################################################################################
############################### ML-EVOLVE HEURISTICS ################################
#####################################################################################

def random_mapping_heuristic(graphs: GraphDataset) -> List[Tuple[Dict[str, Any], str]]:
    """
    Random mapping is the first baseline heuristics used in the
    ML-EVOLVE graph data augmentation technique, shown in the 
    https://dl.acm.org/doi/pdf/10.1145/3340531.3412086 paper by Zhou et al.

    The idea is the followind (for a single graph): we have to create E_cdel 
    and E_cadd. First of all they set E_cdel = E (i.e., the entire set of 
    existing edges) and E_cadd = all non existing edges. Then to construct
    E_add and E_del they sample from the respective set.

            E_add = random.sample(E_cadd, size=ceil(m * beta)) and
            E_del = random.sample(E_cdel, size=ceil(m * beta))

    where m = |E| and beta is a number setted to 0.15 in the paper.

    :param graphs: the entire dataset of graphs
    :return: the new graph G' = (V, (E + E_add) - E_del)
    """
    new_graphs = []
    
    # Iterate over all graphs
    for _, ds_element in graphs.graph_ds.items():
        current_graph_data, label = ds_element

        # Takes all edges
        e_cdel = current_graph_data["edges"]

        # Takes every pair of nodes that is not an edge
        e_cadd = []
        for node_x, node_y in cartesian_product(current_graph_data["nodes"]):
            if node_x != node_y and (node_x, node_y) not in e_cdel:
                e_cadd.append([node_x, node_y])
                e_cadd.append([node_y, node_x])
        
        if not e_cadd:
            continue
        
        # Then we have to sample
        number_of_edges = len(current_graph_data["edges"])
        e_add = random.sample(e_cadd, k=math.ceil(number_of_edges * config.BETA))
        e_del = random.sample(e_cdel, k=math.ceil(number_of_edges * config.BETA))

        # Remove and add edges
        for e in e_del:
            current_graph_data["edges"].remove(e)
        
        current_graph_data["edges"].extend(e_add)
        
        # Let's do a deepcopy to not modify the original graph
        new_graphs.append((current_graph_data, label))
    
    return new_graphs


def motif_similarity_mapping_heuristic(graphs: GraphDataset) -> List[Tuple[Dict[str, Any], str]]:
    """
    Motif-similarity mapping is the second heuristics for new 
    data generation, presented in https://dl.acm.org/doi/pdf/10.1145/3340531.3412086 
    paper by Zhou et al. The idea is based on the concept of graph motifs: 
    sub-graphs that repeat themselves in a specific graph or even among various
    graphs. Each of these sub-graphs, defined by a particular pattern of
    interactions between vertices, may describe a framework in which particular
    functions are achieved efficiently.

    In this case they consider the so-called open-triad, equivalent to a lenght-2
    paths emanating from the head vertex v that induce a triangle. That is, an
    open-triad is for instance a sub-graph composed of three vertices v1, v2, v3
    and this edges (v1, v2) and (v1, v3). This induce a triangle since we can swap
    edges like (v1, v2) becomes (v2, v3) or (v1, v3) becomes (v3, v2). 

    This is the base idea: for all open-triad with vertex head v and tail u we
    construct E_cadd = {(v, u) | A(u,v)=0 and A^2(u,v)=0 and v != u} where
    A(u,v) is the value of the adjacency matrix for the edge (v,u). Then to construct
    E_add we do a weighted random sampling from E_cadd, where the weight depends on
    an index called the Resource Allocation Index (The formula can be found in the
    'networkx' python module under networkx.algorithms.link_prediction.resource_allocation_index).
    Similarly, we compute the deletation probability as w_del = 1 - w_add, and finally
    for each open-triad involving (v, u) we weighted random sample edges to remove.
    This removed edges will construct the E_del set.

    You can have a better look of the algorithm (Algorithm 1) in this paper
    https://arxiv.org/pdf/2007.05700.pdf (by Zhou et al)


    :param graphs: the dataset with all graphs
    :return: the new graph G' = (V, (E + E_add) - E_del)
    """
    new_graphs = []

    # Iterate over all graphs
    for _, ds_element in tqdm(graphs.graph_ds.items()):
        current_graph_data, label = ds_element
        number_of_nodes = len(current_graph_data["nodes"])
        number_of_edges = len(current_graph_data["edges"])

        # First of all let's define a mapping from graph's nodes and
        # their indexes in the adjacency matrix, so also for a the reverse mapping
        node_mapping = dict(zip(current_graph_data["nodes"], range(number_of_nodes)))
        reverse_node_mapping = {v : k for k, v in node_mapping.items()}

        # Then we need the adjancency matrix and its squared power
        # Recall that the square power of A, A^2, contains for all
        # (i, j): A[i,j] = deg(i) if i = j, otherwise it shows if
        # there exists a path of lenght 2 that connect i with j. In
        # this case the value of the cell is A[i,j] = 1, 0 otherwise.
        adj_matrix = build_adjacency_matrix(current_graph_data).numpy()
        power_2_adjancency = matrix_power(adj_matrix, 2)

        # The first step of the algorithm is to compute E_cadd
        e_cadd = []
        for node_x, node_y in cartesian_product(current_graph_data["nodes"]):

            # mapping is needed to index the adjancency matrix
            node_x, node_y = node_mapping[node_x], node_mapping[node_y]

            # In this case, what we wanna find are all that edges that
            # does not exists in the graph, but if they would exists than
            # no open-triad could be present inside the graph.
            if adj_matrix[node_x,node_y] == 0 and power_2_adjancency[node_x,node_y] != 0 \
                and node_x != node_y and (node_y, node_x) not in e_cadd:
                e_cadd.append((reverse_node_mapping[node_x], reverse_node_mapping[node_y]))

        # If there's not new graph to be added then we can stop the process
        if not e_cadd:
            continue

        possible_triads = dict()
        rai_dict = dict()
        total_rai_no_edges = 0.0

        # In this step, we search for all edges inside E_cadd
        # the other two edges that constitute the triad. The search
        # look only for the first pair, no furthermore. 
        # Here we can also compute the Resource Allocation Index.
        for (node_x, node_y) in e_cadd:
            node_x, node_y = node_mapping[node_x], node_mapping[node_y]
            node   = reverse_node_mapping[(adj_matrix[node_x, :] + adj_matrix[:, node_y]).argmax()]
            node_x = reverse_node_mapping[node_x]
            node_y = reverse_node_mapping[node_y]
            possible_triads[(node_x, node_y)] = [(node_x, node), (node, node_y)]

            nx_graph = to_nxgraph(current_graph_data)
            nx_graph = nx_graph.to_undirected()

            rai = resource_allocation_index(nx_graph, [(node_x, node_y)])
            _, _, rai = next(iter(rai))
            rai_dict[(node_x, node_y)] = rai
            total_rai_no_edges += rai

            edges = possible_triads[(node_x, node_y)]
            for u, v, p in resource_allocation_index(nx_graph, edges):      # Cost O(2)
                if (u, v) not in rai_dict:
                    rai_dict[(u, v)] = p

        # In this step of the algorithm, we have to construct the W_add set.
        # Then, we can sample some edges from E_cadd and construct E_add.
        w_add = dict()
        for (node_x, node_y) in rai_dict:
            if (node_x, node_y) in e_cadd:
                w_add[(node_x, node_y)] = rai_dict[(node_x, node_y)] / total_rai_no_edges

        e_add_sample_number = math.ceil(number_of_edges * config.BETA)

        idxs = list(range(0, len(e_cadd)))
        p_distribution = list(w_add.values())

        if len(idxs) < e_add_sample_number:
            continue

        choices = np.random.choice(idxs, size=e_add_sample_number, p=p_distribution, replace=False)
        e_add = np.array(e_cadd)[choices]

        # Finally, the second to the last step is to fill the E_del set.
        # In this step we compute the deletation weights, only for those
        # edges that belongs to the same triad of the previously chosen
        # edges. Essentially, those edges that belongs to E_add
        e_del = []
        for edge in e_add:
            left, right = possible_triads[tuple(edge)]
            left, right = left, right
            w_del_left = 1 - rai_dict[tuple(left)] / total_rai_no_edges
            w_del_right = 1 - rai_dict[tuple(right)] / total_rai_no_edges
            p_distribution = [w_del_left, w_del_right]
            ch = random.choices([left, right], k=1, weights=p_distribution)[0]

            if list(ch) in e_del:
                continue

            e_del.append(list(ch))
            e_del.append(list(ch[::-1]))

        # Remove and add edges
        for e in e_del:
            current_graph_data["edges"].remove(e)
        
        e_add = np.vstack((e_add.T, e_add[:, ::-1].T)).T.reshape(-1, 2).tolist()
        current_graph_data["edges"].extend(e_add)

        # The last step is to construct the new graph
        new_graphs.append((current_graph_data, label))
    
    return new_graphs


def augment_dataset(dataset: GraphDataset, heuristic: str="random mapping") -> List[Tuple[Dict[str,Any], str]]:
    """Apply the augmentation to the dataset"""
    heuristics = {
        "random_mapping"           : random_mapping_heuristic,
        "motif_similarity_mapping" : motif_similarity_mapping_heuristic
    }

    chosen_heuristic = heuristics[heuristic]
    augmented_data = chosen_heuristic(dataset)

    return augmented_data