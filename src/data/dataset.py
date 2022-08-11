import torch
import torch_geometric.data as gdata

from typing import Optional
import requests
import zipfile
import networkx as nx
from utils.utils import GeneratorTxt2Graph
from typing import Dict, List, Tuple, Union, Optional
import config


class GraphDataset(gdata.Dataset):
    def __init__(self, graphs_ds: Dict[str, Tuple[nx.Graph, str]]) -> None:
        super(GraphDataset, self).__init__()
        self.graphs_ds = graphs_ds

    @staticmethod
    def download_data_from_url() -> None:
        """ Download the dataset from the Internet """
        print("--- Downloading The dataset from the Internet as a ZIP archive ---")
        response = requests.get(config.TRIANGLES_DATA_URL)

        # Save the content as a zip file
        with open(f"{config.ROOT_PATH}/TRIANGLES.zip", mode="wb") as iofile:
            iofile.write(response.content)

        # Extract the file
        print("--- Extracting files from the archive ---")
        with zipfile.ZipFile(f"{config.ROOT_PATH}/TRIANGLES.zip", mode="r") as zip_ref:
            zip_ref.extractall(config.ROOT_PATH)

        graph_attribute = open(config.GRAPH_ATTRIBUTE).readlines()
        graph_labels = open(config.GRAPH_LABELS).readlines()
        graph_node_attribute = open(config.NODE_NATTRIBUTE).readlines()
        graph_indicator = open(config.GRAPH_INDICATOR).readlines()
        graph_a = open(config.GRAPH_A).readlines()

        graphs_gen = GeneratorTxt2Graph(
            graph_attribute=graph_attribute,
            graph_labels=graph_labels,
            node_attribute=graph_node_attribute,
            graph_indicator=graph_indicator,
            graph_adjacency=graph_a,
            node_labels=None,
            edge_labels=None,
            edge_attributes=None
        )

        graphs = graphs_gen.generate()

        return graphs

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

        graph = self.graphs_ds[str(idx)]
        g, label = graph[0].to_directed(), graph[1]

        # Retrieve nodes attributes
        attrs = list(g.nodes(data=True))
        x = torch.tensor([list(map(int, a.values()))
                         for _, a in attrs], dtype=torch.float)

        # Retrieve edges
        edge_index = torch.tensor([list(e) for e in g.edges], dtype=torch.long) \
                          .t()                                                  \
                          .contiguous()

        # Retrieve ground trouth labels
        y = torch.tensor([int(label)], dtype=torch.int)

        return gdata.Data(x=x, edge_index=edge_index, y=y)

    @classmethod
    def dataset_from_labels(cls, mask: torch.Tensor,
                            classes: torch.Tensor,
                            graphs: Dict[str, Tuple[nx.Graph, str]]
                            ) -> 'GraphDataset':
        """ Return a new Dataset containing only graphs with specific labels """
        print("--- Creating the Dataset ---")
        filter = classes[(mask[:, None] == classes[None, :]).any(dim=0)].numpy()\
            .astype(str)\
            .tolist()

        filtered_graphs = {k: v for k, v in graphs.items() if v[1] in filter}
        graph_dataset = super(GraphDataset, cls).__new__(cls)

        graph_dataset.__init__(filtered_graphs)

        return graph_dataset


def get_all_labels(graphs: Dict[str, Tuple[nx.Graph, str]]) -> torch.Tensor:
    """ Return a list containings all labels of the dataset """
    return torch.tensor(list(set([int(v[1]) for _, v in graphs.items()])))


def generate_train_val_test(perc_test: float,
                            perc_train: float,
                            graphs: Optional[Dict[str, Tuple[nx.Graph, str]]]=None,
                            download_data: bool=True,
) -> Tuple[GraphDataset, GraphDataset, GraphDataset]:
    """ Return dataset for training, validation and testing """
    print("--- Generating Train, Test and validation datasets --- ")
    if download_data:
        graphs = GraphDataset.download_data_from_url()

    classes = get_all_labels(graphs)
    n_class = len(classes)
    perm = torch.randperm(n_class) + 1

    q_train = n_class * perc_train // 100
    q_test = n_class * perc_test // 100

    train_perm = perm[:q_train]
    test_perm = perm[q_train: q_train + q_test]
    val_perm = perm[q_train + q_test:]

    train_ds = GraphDataset.dataset_from_labels(train_perm, classes, graphs)
    test_ds = GraphDataset.dataset_from_labels(test_perm,  classes, graphs)
    val_ds = GraphDataset.dataset_from_labels(val_perm,   classes, graphs)

    return train_ds, test_ds, val_ds
