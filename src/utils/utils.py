import pickle
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from typing import Any, Dict, List, Tuple, Union
import config
import os
import shutil
import logging
import torch
import numpy as np
import torch_geometric.data as gdata
import random
from functools import wraps
import time


def plot_graph(G : Union[nx.Graph, nx.DiGraph], name: str) -> None:
    """
    Plot a graph
    
    Parameters
    ----------
    graph : Union[nx.Graph, nx.DiGraph]
        Just a nx.Graph object
    name  : str
        The name of the graph
        
    Returns
    -------
    None
    """
    # Getting the 3D Spring layout
    layout = nx.spring_layout(G, dim=3, seed=18)
    
    # Getting nodes coordinate
    x_nodes = [layout[i][0] for i in layout]  # x-coordinates of nodes
    y_nodes = [layout[i][1] for i in layout]  # y-coordinates of nodes
    z_nodes = [layout[i][2] for i in layout]  # z-coordinates of nodes
    
    # Getting a list of edges and create a list with coordinates
    elist = G.edges()
    x_edges, y_edges, z_edges = [], [], []
    for edge in elist:
        x_edges += [layout[edge[0]][0], layout[edge[1]][0], None]
        y_edges += [layout[edge[0]][1], layout[edge[1]][1], None]
        z_edges += [layout[edge[0]][2], layout[edge[1]][2], None]

    colors = np.linspace(0, len(x_nodes))
        
    # Create a trace for the edges
    etrace = go.Scatter3d(x=x_edges,
                          y=y_edges,
                          z=z_edges,
                          mode='lines',
                          line=dict(color='rgb(125,125,125)', width=1),
                          hoverinfo='none'
                         )
    
    # Create a trace for the nodes
    ntrace = go.Scatter3d(x=x_nodes,
                          y=y_nodes,
                          z=z_nodes,
                          mode='markers',
                          marker=dict(
                              symbol='circle',
                              size=6,
                              color=colors,
                              colorscale='Viridis',
                              line=dict(color='rgb(50,50,50)', width=.5)),
                          text=list(layout.keys()),
                          hoverinfo='text'
                         )
    
    # Set the axis
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    
    # Create a layout for the plot
    go_layout = go.Layout(title=f"{name} Network Graph",
                          width=600,
                          height=600,
                          showlegend=False,
                          scene=dict(xaxis=dict(axis),
                                     yaxis=dict(axis),
                                     zaxis=dict(axis)),
                          margin=dict(t=100),
                          hovermode='closest'
                         )
    
    # Plot
    data = [etrace, ntrace]
    fig = go.Figure(data=data, layout=go_layout)
    fig.show()


# FIXME: Delete the folder that corrisponds to the right dataset
def delete_data_folder() -> None:
    """Delete the folder containing data"""
    logging.debug("--- Removing Content Data ---")

    data_folder = os.path.join(config.ROOT_PATH, "TRIANGLES")
    shutil.rmtree(data_folder)
    os.remove(data_folder + ".zip")
    
    logging.debug("--- Removed Finished Succesfully ---")


def setup_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_batch_number(databatch, i_batch, n_way, k_shot):
    """From a N batch takes the i-th batch"""
    dim_databatch = n_way * k_shot
    indices = torch.arange(0, config.BATCH_PER_EPISODES)
    return gdata.Batch.from_data_list(databatch[indices * dim_databatch + i_batch])


# FIXME: Rewrite the extractor starting from .pickle files
class GeneratorTxt2Graph:
    """
    Takes as input a number of graph attributes, labels,
    node attributes, graph indicator, graph adjacency matrix,
    node labels, edge labels and edge attributes and generate
    a number of graphs described by these factors. 
    """
    def __init__(self, **kwargs) -> None:
        self.__graph_attribute  = kwargs['graph_attribute']
        self.__graph_labels     = kwargs['graph_labels']
        self.__node_attribute   = kwargs['node_attribute']
        self.__graph_indicator  = kwargs['graph_indicator']
        self.__graph_adjacency  = kwargs['graph_adjacency']
        self.__node_labels      = kwargs['node_labels']
        self.__edge_labels      = kwargs['edge_labels']
        self.__edge_attributes  = kwargs['edge_attributes']

    def _collect_nodes(self) -> Tuple[Dict[str, List[int]], Dict[str, Tuple[str, int]]]:
        """ Look at the graph_indicator.txt file and return
        a dictionary containing as keys the ID of the graph
        and as values a list of nodes belonging to that graph """

        logging.debug("--- Collecting Nodes ---")

        nodes, i_nodes = dict(), dict()
        for i, graph_id in enumerate(self.__graph_indicator):
            graph_id = graph_id[:-1]
            if graph_id not in nodes:
                nodes[graph_id] = []

            nodes[graph_id].append(i + 1)
            i_nodes[i + 1] = [graph_id, i + 1]

        return nodes, i_nodes

    def _collect_edges(self, i_nodes: Dict[str, Tuple[str, int]], 
                             direct: bool=False) -> Dict[str, List[Tuple[int, int]]]:
        """ Look at the graph_A.txt file and return a dictionary
        containing as keys the ID of the graph and as values
        a list of edges of that graph """
        logging.debug("--- Collecting Edges ...")

        edges = dict()
        for line in self.__graph_adjacency:
            if line == "\n":
                continue
            
            a, b = line.split(", ")
            a, b = a.strip(), b.strip()

            graph_a, node_a = i_nodes[int(a)]
            graph_b, node_b = i_nodes[int(b)]

            if not graph_a == graph_b:
                logging.error(f"Two graphs are not equal: {graph_a} != {graph_b}")
                import sys
                sys.exit(1)

            if graph_a not in edges:
                edges[graph_a] = []
            
            edges[graph_a].append((node_a, node_b))

        return edges
    
    def _collect_node_attributes(self, i_nodes: Dict[str, Tuple[str, int]]) -> None:
        """ Set attributes for each nodes """
        logging.debug("--- Collecting Node Attributes ...")
        for i, attr in enumerate(self.__node_attribute):
            node_i = i_nodes[i + 1]
            attrs = attr.split(", ")
            attrs[-1] = attrs[-1][:-1]
            node_i.append({f"attr{i}" : attr for i, attr in enumerate(attrs)})

    def _collect_graph_labels(self, graphs: Dict[str, nx.Graph]) -> None:
        """ Set the attribute label for each graph """
        logging.debug("--- Collecting Graph Labels ...")
        for i, label in enumerate(self.__graph_labels):
            graph_i = graphs[str(i + 1)]
            graphs[str(i + 1)] = (graph_i, label[:-1])

    def generate(self) -> Dict[str, nx.Graph]:
        """ Return a dictionary of {i : Graph_i} """
        # Get Nodes and Edges
        nodes, i_nodes = self._collect_nodes()
        edges          = self._collect_edges(i_nodes, False)

        # Set attributes for nodes
        self._collect_node_attributes(i_nodes)
        
        # Create the graphs
        graphs = dict()
        for graph_id in edges:
            g = nx.Graph()
            g_nodes = [(i_nodes[n][1], i_nodes[n][-1]) for n in nodes[graph_id]]
            g_edges = edges[graph_id]

            g.add_nodes_from(g_nodes)
            g.add_edges_from(g_edges)

            graphs[graph_id] = g
        
        # Collect graphs without any edges
        for graph_id, node in nodes.items():
            if graph_id not in graphs:
                g = nx.Graph()
                g.add_nodes_from(node)

                graphs[graph_id] = g

        # Set labels for graph
        self._collect_graph_labels(graphs)

        return graphs


def save_with_pickle(path2save: str, content: Any) -> None:
    """Save content inside a .pickle file denoted by path2save"""
    path2save = path2save + ".pickle" if ".pickle" not in path2save else path2save
    with open(path2save, mode="wb") as iostream:
        pickle.dump(content, iostream)


def load_with_pickle(path2load: str) -> Any:
    """Load a content from a .pickle file"""
    with open(path2load, mode="rb") as iostream:
        return pickle.load(iostream)


def compute_num_nodes(graph_list: Dict[str, Tuple[nx.Graph, str]]) -> int:
    """Given a dictionary of graphs, it returns the total number of nodes"""
    num_nodes = 0
    for _, (graph, _) in graph_list.items():
        num_nodes += graph.number_of_nodes()
    
    return num_nodes

def convert(graph_list: Dict[str, Tuple[nx.Graph, str]], 
            num_nodes : int) -> Tuple[Dict[str, dict], torch.Tensor]:
    """
    It takes as input a dictionary of (graph_id, (:obj:`networkx.Graph`, str))
    and the number of nodes and return a transformed dataset and nodes attributes. 
    The transformed dataset is as following, it is a dict with three keys: 

        - 'label2graphs': a dictionary with keys labels and values
                          values a list of graphs with that label
        - 'graph2nodes' : a dictionary with keys graphs id and values
                          a list containing all nodes of that graph
        - 'graph2edges' : a dictionary with keys graphs id and values
                          a list of edges (x, y) for that graph
    
    Finally, attributes is just a list of attributes such that, in position 'i'
    there is the attribute vector for the node with id 'i'.

    :param graph_list: a dictionary of (graph_id, (:obj:`networkx.Graph`, str))
    :param num_nodes: the total number of nodes
    :return: the transformed dataset and the attributes
    """
    label2graphs = dict()
    graph2nodes = dict()
    graph2edges = dict()
    attributes = [0] * num_nodes

    for i_graph, (graph, label) in graph_list.items():
        label = int(label)
        i_graph = int(i_graph)

        # Populate label2graph
        if label not in label2graphs:
            label2graphs[label] = []

        label2graphs[label].append(i_graph)

        # Populate graph2nodes
        if i_graph not in graph2nodes:
            graph2nodes[i_graph] = []
        
        graph2nodes[i_graph] = list(graph.nodes())

        # Populate attributes
        nodes_attrs = graph.nodes(data=True)
        for node_i, attrs in nodes_attrs:
            attrs_list = list(map(lambda x: float(x), attrs.values()))
            attributes[node_i - 1] = attrs_list if attrs_list else [.0, .0]
        
        # Populate graph2edges
        if i_graph not in graph2edges:
            graph2edges[i_graph] = []
        
        graph2edges[i_graph] = list(map(list, graph.edges()))

    total_data = {
        "label2graphs" : label2graphs,
        "graph2nodes"  : graph2nodes,
        "graph2edges"  : graph2edges
    }

    attributes = torch.tensor(attributes)

    return total_data, attributes


def split(data: Dict[str, dict], train_percentage: float=80.0) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """split the data into train and test set"""
    all_labels = torch.tensor(list(data["label2graphs"].keys())).unique()
    num_labels = all_labels.shape[0]
    num_train  = num_labels * train_percentage // 100
    sampled_labels = random.sample(all_labels.tolist(), int(num_train))

    train_label2graphs = {k : v for k, v in data["label2graphs"].items() if k in sampled_labels}
    remaining_graphs   = torch.tensor(list(train_label2graphs.values())).view(1, -1)[0].tolist()
    train_graph2nodes  = {k : v for k, v in data["graph2nodes"].items() if k in remaining_graphs}
    train_graph2edges  = {k : v for k, v in data["graph2edges"].items() if k in remaining_graphs}

    train_data = {
        "label2graphs" : train_label2graphs,
        "graph2nodes"  : train_graph2nodes,
        "graph2edges"  : train_graph2edges
    }

    test_label2graphs = {k : v for k, v in data["label2graphs"].items() if k not in sampled_labels}
    test_graph2nodes  = {k : v for k, v in data["graph2nodes"].items() if k not in remaining_graphs}
    test_graph2edges  = {k : v for k, v in data["graph2edges"].items() if k not in remaining_graphs}

    test_data = {
        "label2graphs" : test_label2graphs,
        "graph2nodes"  : test_graph2nodes,
        "graph2edges"  : test_graph2edges
    }

    return train_data, test_data


def elapsed_time(func):
    """Just a simple wrapper for counting elapsed time from start to end"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        logging.debug("Elapsed Time: {:.6f}".format(end - start))
    
    return wrapper


def get_max_acc(accs, step, scores, min_step, test_step):
    step = np.argmax(scores[min_step - 1 : test_step]) + min_step - 1
    return accs[step]


def rename_edge_indexes(data_list: List[gdata.Data]) -> List[gdata.Data]:
    """
    Takes as input a bunch of :obj:`torch_geometric.data.Data` and renames
    each edge node (x, y) from 1 to total number of nodes. For instance, if we have
    this edge_index = [[1234, 1235, 1236, 1237], [1238, 1239, 1230,1241]] this became
    egde_index = [[0, 1, 2, 3],[4, 5, 6, 7]] and so on. 

    :param data_list: the list of :obj:`torch_geometric.data.Data`
    :return: a new list of data
    """
    # First of all let's compute the total number of nodes overall
    total_number_nodes = 0
    for data in data_list:
        total_number_nodes += data.x.shape[0]
    
    # Generate the new nodes
    nodes = torch.arange(0, total_number_nodes)
    
    # Takes the old nodes from the edge_index attribute
    old_nodes = None
    for data in data_list:
        x, y = data.edge_index
        x = torch.hstack((x, y)).unique(sorted=False)
        
        if old_nodes is None:
            old_nodes = x
            continue
    
        old_nodes = torch.hstack((old_nodes, x))
    
    # Create mapping from old to new nodes
    mapping = dict(zip(old_nodes.tolist(), nodes.tolist()[:old_nodes.shape[0]]))
    
    # Finally, map the new nodes
    for data in data_list:
        x, y = data.edge_index
        new_x = torch.tensor(list(map(lambda x: mapping[x], x.tolist())), dtype=x.dtype, device=x.device)
        new_y = torch.tensor(list(map(lambda y: mapping[y], y.tolist())), dtype=y.dtype, device=y.device)
        new_edge_index = torch.vstack((new_x, new_y))
        data.edge_index = new_edge_index
    
    return data_list


def data_batch_collate(data_list: List[gdata.Data]) -> gdata.Data:
    """
    Takes as input a list of data and create a new :obj:`torch_geometric.data.Data`
    collating all together. This is a replacement for torch_geometric.data.Batch.from_data_list

    :param data_list: a list of torch_geometric.data.Data objects
    :return: a new torch_geometric.data.Data object
    """
    x = None
    edge_index = None
    batch = []
    num_graphs = 0
    y = None

    # Do a shuffle of the data
    random.shuffle(data_list)
    
    for i_data, data in enumerate(data_list):
        x = data.x if x is None else torch.vstack((x, data.x))
        edge_index = data.edge_index if edge_index is None else torch.hstack((edge_index, data.edge_index))
        batch += [i_data] * data.x.shape[0]
        num_graphs += 1
        y = data.y if y is None else torch.hstack((y, data.y))

    # Create a mapping between y and a range(0, num_classes_of_y)
    # First we need to compute how many classes do we have
    num_classes = y.unique().shape[0]
    classes = list(range(0, num_classes))
    mapping = dict(zip(y.unique(sorted=False).tolist(), classes))
    
    # This mapping is necessary when computing the cross-entropy-loss
    new_y = torch.tensor(list(map(lambda x: mapping[x], y.tolist())), dtype=y.dtype, device=y.device)
    
    data_batch = gdata.Data(
        x=x, edge_index=edge_index, batch=torch.tensor(batch),
        y=new_y, num_graphs=num_graphs, old_classes_mapping=mapping
    )

    return data_batch


def task_sampler_uncollate(task_sampler: 'data.sampler.TaskBatchSampler', data_batch: gdata.Batch):
    """
    Takes as input the task sampler and a batch containing both the 
    support and the query set. It returns two different DataBatch
    respectively for support and query_set.

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

    :param task_sampler: The task sampler
    :param data_batch: a batch with support and query set
    :return: support batch, query batch
    """
    n_way = task_sampler.task_sampler.n_way
    k_shot = task_sampler.task_sampler.k_shot
    n_query = task_sampler.task_sampler.n_query
    task_batch_size = task_sampler.task_batch_size

    total_support_query_number = n_way * (k_shot + n_query)
    support_plus_query = k_shot + n_query

    # Initialize batch list for support and query set
    support_data_batch = []
    query_data_batch = []

    # I know how many batch do I have, so
    for batch_number in range(task_batch_size):

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
    
    # Rename the edges
    support_data = data_batch_collate(rename_edge_indexes(support_data_batch))
    query_data   = data_batch_collate(rename_edge_indexes(query_data_batch))

    # Create new DataBatchs and return
    return support_data, query_data
        