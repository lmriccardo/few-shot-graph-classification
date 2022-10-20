import torch
import random
import os
import argparse

from typing import List, Dict, Tuple, Any
from copy import deepcopy
from utils.utils import save_with_pickle,   \
                        elapsed_time,       \
                        compute_percentage, \
                        flat,               \
                        FolderContext


class Txt2Graph:
    def __init__(self, graph_indicator_file: str,
                       node_attributes_file: str,
                       adjacency_matr_file : str,
                       graph_labels_file   : str
    ) -> None:
        # .
        self.graph_indicator_file = graph_indicator_file
        self.node_attributes_file = node_attributes_file
        self.adjacency_matr_file  = adjacency_matr_file
        self.graph_labels_file    = graph_labels_file

    def _get_nodes2graph(self) -> Dict[int, int]:
        """ Return a dictionary containing (node_id, graph_id) """
        nodes2graph = dict()

        # Open the file
        with open(self.graph_indicator_file, mode="r") as iostream:
            node_count = 0
            while line := iostream.readline():
                line = line[:-1]
                graph_id = int( line )
                nodes2graph[node_count] = graph_id
                node_count += 1

        return nodes2graph

    def _reverse_nodes2graph(self, n2g: Dict[int, int]) -> Dict[int, List[int]]:
        """ Return a dictionary containing (graph_id, list of nodes_id) """
        graph2nodes = dict()

        # Reverse the nodes2graph dictionary
        for node_id, graph_id in n2g.items():
            if graph_id not in graph2nodes:
                graph2nodes[graph_id] = []

            graph2nodes[graph_id].append(node_id)

        return graph2nodes

    def _get_label2graphs(self, graphs: List[int]) -> Dict[int, List[int]]:
        """ Return a dictionary containing (label, list of graphs id) """
        label2graphs = dict()
        i_label2graphs = dict()

        # Open the file
        with open(self.graph_labels_file, mode="r") as iostream:
            graph_count = 0
            while line := iostream.readline():
                line = line[:-1]
                label = int( line )
                
                if label not in label2graphs:
                    label2graphs[label] = []

                label2graphs[label].append(graphs[graph_count])

                i_label2graphs[graphs[graph_count]] = label
                graph_count += 1

        return label2graphs, i_label2graphs

    def _get_graph2edges(self, node2graph: Dict[int, int]) -> Dict[int, List[Tuple[int, int]]]:
        """ Return a dictionary containing (graph_id, list of pair of node) """
        graph2edges = dict()

        # Open the file
        with open(self.adjacency_matr_file, mode="r") as iostream:
            while line := iostream.readline():
                line = line[:-1]

                # Get the pair of nodes and convert
                node_x, node_y = line.split(", ")
                node_x, node_y = int( node_x ), int( node_y )
                node_x, node_y = node_x - 1, node_y - 1


                graph_x, graph_y = node2graph[node_x], node2graph[node_y]

                # Check that the two nodes belong to the same graph
                assert (
                    graph_x == graph_y, 
                    f"{node_x} and {node_y} don't belong to the same graph"
                )

                if graph_x not in graph2edges:
                    graph2edges[graph_x] = []

                graph2edges[graph_x].append([node_x, node_y])

        return graph2edges

    def _get_nodes_attributes(self) -> List[torch.Tensor]:
        """ Return a list of tensor """
        attributes = []

        # Open the file
        with open(self.node_attributes_file) as iostream:
            while line := iostream.readline():
                line = line[:-1]
                attribute = line.split(", ")
                attribute = list(map(float, attribute))
                attribute = torch.tensor(attribute)
                attributes.append(attribute)

        return attributes

    def _clean(self, graph2nodes    : Dict[int, List[int]],
                     graph2edges    : Dict[int, List[Tuple[int, int]]],
                     label2graphs   : Dict[int, List[int]],
                     i_label2graphs : Dict[int, int]
    ) -> None:
        """ Remove empty graphs """
        c_graph2nodes = deepcopy(graph2nodes)
        for graph_id in c_graph2nodes.keys():
            if graph_id not in graph2edges:
                graph2nodes.pop(graph_id)
                label2graphs[i_label2graphs[graph_id]].remove(graph_id)

    def convert(self) -> Any:
        """ Execute conversion """
        # Takes nodes and graphs
        node2graph  = self._get_nodes2graph()
        graph2nodes = self._reverse_nodes2graph(node2graph)
        graphs      = sorted(list(graph2nodes.keys()))

        # Takes labels
        label2graphs, i_label2graphs = self._get_label2graphs(graphs)

        # Take edges
        graph2edges = self._get_graph2edges(node2graph)

        # Takes attributes
        attributes  = self._get_nodes_attributes()

        # Clean data
        self._clean(graph2nodes, graph2edges, label2graphs, i_label2graphs)

        return attributes, label2graphs, graph2nodes, graph2edges


def get_single_data(labels       : List[int],
                    label2graphs : Dict[int, List[int]],
                    graph2nodes  : Dict[int, List[int]],
                    graph2edges  : Dict[int, List[Tuple[int, int]]]
) -> Any:
    """ Filter data by labels """
    r_labels2graph = {k : label2graphs[k] for k in labels}
    r_graphs       = flat([v for v in r_labels2graph.values()])
    r_graph2nodes  = {g : graph2nodes[g] for g in r_graphs} 
    r_graph2edges  = {g : graph2edges[g] for g in r_graphs}

    return {
        "label2graphs" : r_labels2graph,
        "graph2nodes"  : r_graph2nodes,
        "graph2edges"  : r_graph2edges
    }


def split_dataset(label2graphs : Dict[int, List[int]],
                  graph2nodes  : Dict[int, List[int]],
                  graph2edges  : Dict[int, List[Tuple[int, int]]],
                  train_n      : float, 
                  val_n        : float, 
                  test_n       : float) -> Tuple[Any, Any, Any]:
    """ Split the dataset into train, val and test """
    n_labels = len(label2graphs)
    n_train  = compute_percentage(n_labels, train_n)
    n_test   = compute_percentage(n_labels, test_n)
    n_val    = compute_percentage(n_labels, val_n)
    n_test  += n_labels - (n_train + n_test + n_val)

    # Get labels and shuffle them
    labels = list(label2graphs.keys())
    random.shuffle(labels)

    # Take and sort labels for train, val and test
    train_labels = labels[:n_train]
    val_labels   = labels[n_train:n_train + n_val]
    test_labels  = labels[n_train + n_val:n_train + n_val + n_test]

    train_labels = sorted(train_labels)
    val_labels   = sorted(val_labels)
    test_labels  = sorted(test_labels)

    print("Labels for train set ", train_labels)
    print("Labels for validation set ", val_labels)
    print("Labels for test set ", test_labels)

    train_data = get_single_data(train_labels, label2graphs, graph2nodes, graph2edges)
    test_data  = get_single_data(test_labels, label2graphs, graph2nodes, graph2edges)
    val_data   = get_single_data(val_labels, label2graphs, graph2nodes, graph2edges)

    return train_data, val_data, test_data

@elapsed_time
def converter(data_dir: str, data_name: str, train_n: float, val_n: float, test_n: float) -> None:
    """
    Takes as input the root directory containing the entire dataset. The
    folder should contains mainly four files with the following names:
    <folder_name>_A.txt, <folder_name>_graph_indicator.txt, 
    <folder_name>_graph_labels.txt and <folder_name>_node_attributes.txt.

    Once extracting the data from the files, i.e. a bunch of graphs, they
    are converted into the transormed dataset (see convert function for
    more details). Finally, the entire dataset is formerly splitted into
    train and test set, and then the train set is splitted in train and
    validation set. 

    When completed, four new files are created in the same directory
    <folder_name>: <folder_name>_node_attributes.pickle,
    <folder_name>_train_set.pickle, <folder_name>_val_set.pickle,
    <folder_name>_test_set.pickle.

    :param data_dir: the directory containing the four files
    :param data_name: the name of the dataset
    :param train_n: the percentage of the train set
    :param val_n: the percentage for the validation set
    :param test_n: the percentage for the test set
    """
    random.seed(42)

    graph_indicator_file = os.path.join(data_dir, f"{data_name}_graph_indicator.txt")
    node_attributes_file = os.path.join(data_dir, f"{data_name}_node_attributes.txt")
    adj_matrix_file = os.path.join(data_dir, f"{data_name}_A.txt")
    graph_labels_file = os.path.join(data_dir, f"{data_name}_graph_labels.txt")

    assert os.path.exists(graph_indicator_file) and \
           os.path.exists(node_attributes_file) and \
           os.path.exists(adj_matrix_file)      and \
           os.path.exists(graph_labels_file), \
           f"ERROR: One of the required files in {data_dir} does not exists. Please check the presence of: \n" + \
           f"\t- {graph_indicator_file}\n\t- {node_attributes_file}\n" + \
           f"\t- {adj_matrix_file}\n\t- {graph_labels_file}"

    conv = Txt2Graph(
        graph_indicator_file, node_attributes_file,
        adj_matrix_file, graph_labels_file
    )

    print("[*] All checks OK")
    print("[*] Running conversion with this parameter")
    print(f"    DATA DIR: {data_dir}")
    print(f"    DATA NAME: {data_name}")
    print(f"    TRAIN PERCENTAGE: {train_n}")
    print(f"    VALIDATION PERCENTAGE: {val_n}")
    print(f"    TEST PERCENTAGE: {test_n}")

    # Get back attributes, label2graph, graph2nodes and graph2edges
    attributes, label2graph, graph2nodes, graph2edges = conv.convert()

    train_data, test_data, val_data = split_dataset(
        label2graph, graph2nodes, graph2edges,
        train_n, val_n, test_n
    )

    print("[*] Conversion ended succesfully")

    # Now save the pickle object
    save_with_pickle(os.path.join(data_dir, f"{data_name}_train_set.pickle"),       train_data)
    save_with_pickle(os.path.join(data_dir, f"{data_name}_test_set.pickle"),        test_data)
    save_with_pickle(os.path.join(data_dir, f"{data_name}_val_set.pickle"),         val_data)
    save_with_pickle(os.path.join(data_dir, f"{data_name}_node_attributes.pickle"), attributes)

    print("[*] Removing previous TXT files from the root directory")

    with FolderContext(data_dir):
        os.remove(f"{data_name}_graph_indicator.txt")
        os.remove(f"{data_name}_node_attributes.txt")
        os.remove(f"{data_name}_A.txt")
        os.remove(f"{data_name}_graph_labels.txt")

    print("[*] EXITING")


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--folder', help="Absolute path to the directory containing data information",    default=os.getcwd(), type=str)
    argparser.add_argument('-n', '--name',   help="The name of the dataset",                                       default="MyData",    type=str)
    argparser.add_argument('--train',        help="The percentage of the dataset corresponding to train set",      default=60.0,        type=float)
    argparser.add_argument('--validation',   help="The percentage of the dataset corresponding to validation set", default=30.0,        type=float)
    argparser.add_argument('--test',         help="The percentage of the dataset corresponding to test set",       default=10.0,        type=float)

    args = argparser.parse_args()
    data_dir  = os.path.abspath(args.folder)
    data_name = args.name
    train_n   = args.train
    val_n     = args.validation
    test_n    = args.test

    assert os.path.exists(data_dir), f"ERROR: {data_dir} does not exists ... Please provide a correct path"
    assert int( train_n + val_n + test_n ) == 100, \
           f"ERROR: Percentages sum up to {int( train_n + val_n + test_n )} instead of 100"

    converter(data_dir, data_name, train_n, val_n, test_n)