from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Generator, Callable
from functools import wraps
from datetime import datetime
import torch_geometric.data as gdata
import pickle
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import config
import os
import shutil
import logging
import torch
import random
import time
import requests
import zipfile
import sys


########################################################
################## PLOT GRAPH UTILITY ##################
########################################################

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
                          width=1000,
                          height=1000,
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


#################################################################
################## FOLDER MANAGEMENT UTILITIES ##################
#################################################################

class FolderContext:
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.old_dir = os.getcwd()

    def __enter__(self) -> None:
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_value, tb) -> None:
        os.chdir(self.old_dir)

def delete_data_folder(path2delete: str) -> None:
    """Delete the folder containing data"""
    logging.debug("--- Removing Content Data ---")
    shutil.rmtree(path2delete)
    logging.debug("--- Removed Finished Succesfully ---")


def scandir(root_path: str) -> List[str]:
    """Recursively scan a directory looking for files"""
    root_path = os.path.abspath(root_path)
    content = []
    for file in os.listdir(root_path):
        new_path = os.path.join(root_path, file)
        if os.path.isfile(new_path):
            content.append(new_path)
            continue
        
        content += scandir(new_path)
    
    return content


def download_zipped_data(url: str, path2extract: str, dataset_name: str, logger: logging.Logger) -> List[str]:
    """Download and extract a ZIP file from URL. Return the content filename"""
    logger.debug(f"--- Downloading from {url} ---")
    response = requests.get(url)

    abs_path2extract = os.path.abspath(path2extract)
    zip_path = os.path.join(abs_path2extract, f"{dataset_name}.zip")
    with open(zip_path, mode="wb") as iofile:
        iofile.write(response.content)

    # Extract the file
    logger.debug("--- Extracting files from the archive ---")
    with zipfile.ZipFile(zip_path, mode="r") as zip_ref:
        zip_ref.extractall(abs_path2extract)

    logger.debug(f"--- Removing {zip_path} ---")
    os.remove(zip_path)

    return scandir(os.path.join(path2extract, dataset_name))


def setup_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_with_pickle(path2save: str, content: Any) -> None:
    """Save content inside a .pickle file denoted by path2save"""
    print("[*] Saving object into ", path2save)
    path2save = path2save + ".pickle" if ".pickle" not in path2save else path2save
    mode = "wb" if os.path.exists(os.path.abspath(path2save)) else "xb"
    with open(os.path.abspath(path2save), mode=mode) as iostream:
        pickle.dump(content, iostream)


def load_with_pickle(path2load: str) -> Any:
    """Load a content from a .pickle file"""
    with open(path2load, mode="rb") as iostream:
        return pickle.load(iostream)


def elapsed_time(func):
    """Just a simple wrapper for counting elapsed time from start to end"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        logging.debug("Elapsed Time: {:.6f}".format(end - start))
    
    return wrapper


######################################################################
################## DATASET AND DATALOADER UTILITIES ##################
######################################################################

def rename_edge_indexes(data_list: List[gdata.Data]) -> List[gdata.Data]:
    """
    Takes as input a bunch of :obj:`torch_geometric.data.Data` and renames
    each edge node (x, y) from 1 to total number of nodes. For instance, if we have
    this edge_index = [[1234, 1235, 1236, 1237], [1238, 1239, 1230,1241]] this became
    egde_index = [[0, 1, 2, 3],[4, 5, 6, 7]] and so on. 

    :param data_list: the list of :obj:`torch_geometric.data.Data`
    :param ohe_labels: if we have to manage labels with One-hot encoding.
    :return: a new list of data
    """
    data_ = deepcopy(data_list)
    # First of all let's compute the total number of nodes overall
    total_number_nodes = 0
    for data in data_:
        total_number_nodes += data.x.shape[0]

    # Generate the mapping from old_nodes identifiers to new_node identifiers
    mapping = dict()
    node_number = 0
    for data in data_:
        x, y = data.edge_index
        x = torch.hstack((x, y)).unique(sorted=True)
        mapping.update(dict(zip(x.tolist(), range(node_number, node_number + x.shape[0]))))
        node_number = node_number + x.shape[0]
    
    # Finally, map the new nodes
    for data in data_:
        x, y = data.edge_index
        new_x = torch.tensor(list(map(lambda x: mapping[x], x.tolist())), dtype=x.dtype, device=x.device)
        new_y = torch.tensor(list(map(lambda y: mapping[y], y.tolist())), dtype=y.dtype, device=y.device)
        new_edge_index = torch.vstack((new_x, new_y))
        data.edge_index = new_edge_index
    
    return data_


def data_batch_collate(data_list: List[gdata.Data], oh_labels: bool=False) -> Tuple[gdata.Data, List[gdata.Data]]:
    """
    Takes as input a list of data and create a new :obj:`torch_geometric.data.Data`
    collating all together. This is a replacement for torch_geometric.data.Batch.from_data_list

    :param data_list: a list of torch_geometric.data.Data objects
    :param oh_labels: True, if data's labels are one-hot encoded
    :return: a new torch_geometric.data.Data object as long as the original list
    """
    x = None
    edge_index = None
    batch = []
    num_graphs = 0
    y = None
    
    for i_data, data in enumerate(data_list):
        x = data.x if x is None else torch.vstack((x, data.x))
        edge_index = data.edge_index if edge_index is None else torch.hstack((edge_index, data.edge_index))
        batch += [i_data] * data.x.shape[0]
        num_graphs += 1

        if y is None:
            y = data.y
            continue
        
        y = torch.hstack((y, data.y)) if not oh_labels else torch.vstack((y, data.y))
    
    if not oh_labels:
        # Create a mapping between y and a range(0, num_classes_of_y)
        # First we need to compute how many classes do we have
        num_classes = y.shape[0]
        classes = list(range(0, num_classes))
        mapping = dict(zip(y.unique(sorted=False).tolist(), classes))

        # This mapping is necessary when computing the cross-entropy-loss
        new_y = torch.tensor(list(map(lambda x: mapping[x], y.tolist())), dtype=y.dtype, device=y.device)
    else:
        # We need just to drop those columns with all zeros
        # Now "y" is something like this tensor
        #
        #             [0,  0, 0, 0,  1]
        #             [1,  0, 0, 0,  0]
        #             [0,  1, 0, 0,  0]
        #             [0, .7, 0, 0, .3]
        #                    ...
        # 
        # In this case we have that only the label 0, 1, and 4
        # are used in the sample. So, the idea is to drop the
        # two zero columns in order to have a y tensor that
        # fit the output dimension of the model, i.e., the number
        # of N-way. 
        new_y = y[:, y.sum(dim=0) != 0]
        mapping = None  # Set the mapping to None (it is not necessary)
 
    data_batch = gdata.Data(
        x=x, edge_index=edge_index, batch=torch.tensor(batch),
        y=new_y, num_graphs=num_graphs, old_classes_mapping=mapping
    )

    return data_batch, data_list


def data_batch_collate_edge_renamed(data: List[gdata.Data], oh_labels: bool=False) -> gdata.Data:
    """
    Given a list of data whose egdes have been renamed previously,
    i.e., nodes goes from 0 to #Nodes, create a new single data
    merging those ones from the list.

    :param data: a list of torch_geometric.data.Data
    :param oh_labels: True if labels are one hot encoded, False otherwise
    :return: a single data obtained by merging each data in the list
    """
    # Take the first data and find the maximum node
    max_node_number = data[0].edge_index.flatten().max().item() + 1
    current_edge_index = deepcopy(data[0].edge_index)
    current_x = deepcopy(data[0].x)
    current_labels = deepcopy(data[0].y)
    current_batch = data[0].batch
    max_batch_number = current_batch.max().item()

    # Iterate through the entire list starting from the second element
    for i_data, data_el in enumerate(data, 1):
        current_x = torch.vstack((current_x, data_el.x))
        stack_f = torch.hstack if not oh_labels else torch.vstack
        current_labels = stack_f((current_labels, data_el.y))
        current_batch = torch.hstack((current_batch, data_el.batch + max_batch_number + 1))

        # Now we have to rename the edges.
        # First let's take the nodes of the graph
        nodes_idxs = data_el.edge_index.flatten().unique(sorted=True).tolist()
        nodes_mapping = dict(zip(nodes_idxs, range(max_node_number, max_node_number + len(nodes_idxs))))
        rerenamed_edges = torch.tensor(
            list(
                map(
                    lambda x: nodes_mapping[x],
                    data_el.edge_index.flatten().tolist()
                )
            )
        ).view(2, -1)

        current_edge_index = torch.hstack((current_edge_index, rerenamed_edges))

        # Update the counter for the max node number and the batch number
        max_node_number += len(nodes_idxs)
        max_batch_number = data_el.batch.max().item() + max_batch_number + 1

    return gdata.Data(
        x=current_x, edge_index=current_edge_index, 
        y=current_labels, batch=current_batch,
        old_classes_mapping=data[0].old_classes_mapping
    )

def task_sampler_uncollate(task_sampler: 'TaskBatchSampler', 
                           data_batch  : List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] ) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
]:
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
    
    return support_data_batch, query_data_batch


def add_remaining_edges(edges: List[Tuple[int, int]]) -> List[Tuple[int,int]]:
    """Add for each (x,y) edge a new edge (y,x) if it is not already present"""
    for idx, e in enumerate(edges):
        x, y = e
        if (y, x) not in edges and [y, x] not in edges:
            edges.insert(idx + 1, (y, x))
    
    return edges


def graph2data(graph: nx.Graph, target: str | int, edges: List[Tuple[int, int]]) -> gdata.Data:
    """From a networkx.Graph returns a torch_geometric.data.Data"""
    # Retrieve nodes attributes
    attrs = sorted(list(graph.nodes(data=True)), key=lambda x: x[0])
    x = torch.tensor([list(map(int, a.values())) for _, a in attrs], dtype=torch.float)
    
    edges = add_remaining_edges(edges)
    edge_index = torch.tensor([list(e) for e in edges], dtype=torch.long) \
                        .t()                                                  \
                        .contiguous()                                         \
                        .long()

    # Retrieve ground trouth labels
    y = torch.tensor([int(target)], dtype=torch.long)

    return gdata.Data(x=x, edge_index=edge_index, y=y)
    

def to_nxgraph(graph_data: Dict[str, Any] | gdata.Data, directed: bool=True) -> nx.Graph:
    """Return a networkx.Graph representation of the input graph"""
    if isinstance(graph_data, dict):
        nodes = graph_data["nodes"]
        attributes = graph_data["attributes"]
        edges = graph_data["edges"]

    elif isinstance(graph_data, gdata.Data):
        attributes = graph_data.x.tolist()
        nodes = torch.hstack((graph_data.edge_index[0], graph_data.edge_index[1])).unique().tolist()
        edges = list(map(tuple, graph_data.edge_index.transpose(0,1).tolist()))

    nodes_attributes = []
    for node_i, node in enumerate(nodes):
        nodes_attributes.append((node, {f"attr{i}" : attr for i, attr in enumerate(attributes[node_i])}))

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(nodes_attributes)
    graph.add_edges_from(edges)

    return graph


def to_pygdata(graph_data: Dict[str, Any], label: str | int) -> gdata.Data:
    """Return a torch_geometric.data.Data representation of the graph"""
    x = torch.tensor(graph_data["attributes"], dtype=torch.float)
    edge_index = torch.tensor(graph_data["edges"], dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)
    return gdata.Data(x=x, edge_index=edge_index, y=y)


def to_datadict(data: Union[gdata.Data, List[gdata.Data]]) -> Dict[str,Any]:
    """Return the dictionary representation of one or more graphs data"""
    if not isinstance(data, list):
        data = [data]

    graphs = dict()
    for graph_i, graph_data in enumerate(data):
        edges = graph_data.edge_index.transpose(0,1).tolist()
        nodes = graph_data.edge_index[0].unique(sorted=True)
        attributes = graph_data.x.tolist()

        # Handle one-hot encoded labels
        label = graph_data.y
        if label.shape[0] == 1:
            label = label.item()

        graphs[graph_i] = ({
            "nodes"      : nodes,
            "edges"      : edges,
            "attributes" : attributes
        }, label)

    return graphs


#######################################################
################## GENERAL UTILITIES ##################
#######################################################


def configure_logger(file_logging: bool=False, 
                     logging_path: str="../log", 
                     dataset_name: str="TRIANGLES") -> logging.Logger:
    """Configure the logger and create the file"""
    logger = logging.getLogger(name="fsgc-logger")
    stream_handler = logging.StreamHandler(sys.stdout)

    # Add the formatter
    logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(logger_formatter)

    # Add the stream handler
    logger.addHandler(stream_handler)

    # Set up the file handler if specified
    if file_logging:
        # check if <logging_path>/<dataset_name> dir exists or not
        # if it does not exists then we must crete it
        if not os.path.exists(os.path.join(logging_path, dataset_name)):
            os.makedirs(os.path.join(logging_path, dataset_name))
        
        # Then we have to specify where to log
        now_time = datetime.now()
        logging_file_name = "{name}_{year}-{month}-{day}_{hour}-{minute}.log".format(
            name=dataset_name, year=now_time.year, month=now_time.month, 
            day=now_time.day, hour=now_time.hour, minute=now_time.minute
        )
        logging_file = os.path.join(os.path.join(logging_path, dataset_name), logging_file_name)

        file_handler = logging.FileHandler(filename=logging_file)
        file_handler.setFormatter(logger_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    return logger


def cartesian_product(x: Sequence, y: Optional[Sequence]=None, 
                      filter: Optional[Callable[[Any, Any], bool]]=None) -> Generator:
    """Return the cartesian product between two sequences"""
    if y is None:
        y = x

    for el_x in x:
        for el_y in y:
            if filter is None or filter(el_x, el_y):
                yield (el_x, el_y)


def get_all_labels(graphs: Dict[str, Tuple[nx.Graph, str]]) -> torch.Tensor:
    """ Return a list containings all labels of the dataset """
    return torch.tensor(list(set([int(v[1]) for _, v in graphs.items()])))


def compute_accuracy(vector_a: torch.Tensor, vector_b: torch.Tensor, oh_labels: bool=False) -> float:
    """Compute the accuracy, i.e., the percentage of equal elements"""
    if oh_labels:
        vector_b = vector_b.argmax(dim=1)
        
    equals = torch.eq(vector_a, vector_b)
    return equals.sum() * 100 / vector_a.shape[0]


def get_batch_number(databatch, i_batch, n_way, k_shot):
    """From a N batch takes the i-th batch"""
    dim_databatch = n_way * k_shot
    indices = torch.arange(0, config.BATCH_PER_EPISODES)
    return gdata.Batch.from_data_list(databatch[indices * dim_databatch + i_batch])


def get_max_acc(accs, step, scores, min_step, test_step):
    step = np.argmax(scores[min_step - 1 : test_step]) + min_step - 1
    return accs[step]


def build_adjacency_matrix(graph_data: Dict[str, Any] | gdata.Data) -> torch.Tensor:
    """Construct the adjacency matrix of a graph"""
    if isinstance(graph_data, gdata.Data):
        graph_data = {
            "nodes": graph_data.edge_index[0].unique(sorted=True).tolist(), 
            "edges": graph_data.edge_index.transpose(0, 1).tolist()
        }

    number_of_nodes = len(graph_data["nodes"])
    node_mapping = dict(zip(graph_data["nodes"], range(number_of_nodes)))
    adj_matrix = torch.zeros((number_of_nodes, number_of_nodes), dtype=torch.int)

    for node_x, node_y in graph_data["edges"]:
        node_x = node_mapping[node_x]
        node_y = node_mapping[node_y]
        adj_matrix[node_x, node_y] = 1
    
    return adj_matrix


def compute_percentage(n: int, p: float) -> int:
    """ Compute the percentage p of n """
    return int( (p * n) / 100 )


def flat(x: List[List[int]]) -> List[int]:
    """ From a list of list returns a single list """
    flattened = []

    for e in x:
        flattened.extend(e)

    return flattened