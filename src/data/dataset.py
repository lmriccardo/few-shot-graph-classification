import torch
import torch_geometric.data as gdata

from utils.utils import download_zipped_data, load_with_pickle
import config

import networkx as nx
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import os


class GraphDataset(gdata.Dataset):
    def __init__(self, graphs_ds: Dict[str, Tuple[nx.Graph, str]]) -> None:
        super(GraphDataset, self).__init__()
        self.graphs_ds = graphs_ds

    @classmethod
    def get_dataset(cls, attributes: List[Any], data: Dict[str, Any]) -> 'GraphDataset':
        """
        Returns a new instance of GraphDataset filled with graphs inside data. 'attributes'
        is the list with all the attributes (not only those beloging to nodes in 'data').

        :param data: a dictionary with label2graphs, graph2nodes and graph2edges
        :param attributes: a list with node attributes
        :return: a new instance of GraphDataset
        """
        graphs = dict()

        label2graphs = data["label2graphs"]
        graph2nodes  = data["graph2nodes"]
        graph2edges  = data["graph2edges"]

        for label, graph_list in label2graphs.items():
            for graph_id in graph_list:
                graph_nodes = graph2nodes[graph_id]
                graph_edges = graph2edges[graph_id]
                nodes_attributes = [[attributes[node_id]] for node_id in graph_nodes]
                nodes = []
                for node, attribute in zip(graph_nodes, nodes_attributes):
                    nodes.append((node, {f"attr{i}" : a for i, a in enumerate(attribute)}))

                g = nx.Graph()

                g.add_edges_from(graph_edges)
                g.add_nodes_from(nodes)
            
                graphs[graph_id] = (g, label)

        graphs = dict(sorted(graphs.items(), key=lambda x: x[0]))
        graph_dataset = super(GraphDataset, cls).__new__(cls)
        graph_dataset.__init__(graphs)

        return graph_dataset

    def __repr__(self) -> str:
        return f"GraphDataset(classes={set(self.targets().tolist())},n_graphs={self.len()})"

    def indices(self) -> List[str]:
        """ Return all the graph IDs """
        return list(self.graphs_ds.keys())

    def len(self) -> int:
        return len(self.graphs_ds.keys())

    def targets(self) -> torch.Tensor:
        """ Return all the labels """
        targets = []
        for _, graph in self.graphs_ds.items():
            targets.append(int(graph[1]))

        return torch.tensor(targets)

    def get(self, idx: Union[int, str]) -> gdata.Data:
        """ Return (Graph object, Adjacency matrix and label) of a graph """
        if isinstance(idx, str):
            idx = int(idx)

        graph = self.graphs_ds[idx]
        g, label = graph[0].to_directed(), graph[1]

        # Retrieve nodes attributes
        attrs = list(g.nodes(data=True))
        x = torch.tensor([list(map(int, a.values())) for _, a in attrs], dtype=torch.float)

        # Retrieve edges
        edge_index = torch.tensor([list(e) for e in g.edges], dtype=torch.long) \
                          .t()                                                  \
                          .contiguous()                                         \
                          .long()

        # Retrieve ground trouth labels
        y = torch.tensor([int(label)], dtype=torch.int)

        return gdata.Data(x=x, edge_index=edge_index, y=y)


def get_all_labels(graphs: Dict[str, Tuple[nx.Graph, str]]) -> torch.Tensor:
    """ Return a list containings all labels of the dataset """
    return torch.tensor(list(set([int(v[1]) for _, v in graphs.items()])))


def generate_train_val_test(dataset_name: str,
                            logger: logging.Logger,
                            data_dir: Optional[str]=None, 
                            download: bool=True,
                            download_folder: str="../data"
) -> Tuple[GraphDataset, GraphDataset, GraphDataset]:
    """ Return dataset for training, validation and testing """
    logger.debug("--- Generating Train, Test and Validation datasets --- ")
    
    assert download or data_dir is not None, "At least one between: data_dir and download must be given"

    node_attribute = None
    test_file = None
    train_file = None
    val_file = None

    if data_dir is not None:
        node_attribute = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_node_attributes.pickle")
        test_file = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_test_set.pickle")
        train_file = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_train_set.pickle")
        val_file = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_val_set.pickle")

    if download:
        node_attribute, test_file, train_file, val_file = download_zipped_data(
            config.DATASETS[dataset_name], 
            download_folder, 
            dataset_name
        )

        data_dir = "\\".join(node_attribute.replace("\\", "/").split("/")[:-2])

    node_attribute_data = load_with_pickle(node_attribute)
    test_data = load_with_pickle(test_file)
    train_data = load_with_pickle(train_file)
    val_data = load_with_pickle(val_file)

    train_ds = GraphDataset.get_dataset(node_attribute_data, train_data)
    test_ds  = GraphDataset.get_dataset(node_attribute_data,  test_data)
    val_ds   = GraphDataset.get_dataset(node_attribute_data,   val_data)

    return train_ds, test_ds, val_ds, data_dir



def get_dataset(logger: logging.Logger, 
                download: bool=False, 
                dataset_name: str="TRIANGLES", 
                data_dir: str="../data") -> Tuple[GraphDataset, GraphDataset, GraphDataset, str]:
    """Generate the train, test and validation dataset"""
    data_dir = data_dir if not download else None
    train_ds, test_ds, val_ds, data_dir = generate_train_val_test(
        data_dir=data_dir,
        download=download,
        dataset_name=dataset_name,
        logger=logger
    )
    return train_ds, test_ds, val_ds, data_dir