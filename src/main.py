import sys
import os
sys.path.append(os.getcwd())

from data.dataset import generate_train_val_test
from data.dataloader import FewShotDataLoader
from data.sampler import TaskBatchSampler
from utils.utils import delete_data_folder, setup_seed, get_batch_number

import config
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    setup_seed()

    train_ds, test_ds, val_ds = generate_train_val_test(
        download_data=False, 
        perc_train=50, 
        perc_test=30
    )

    logging.debug("--- Datasets ---")
    print("\n- Train: ", train_ds)
    print("- Test : ", test_ds)
    print("- Validation: ", val_ds)
    print()

    logging.debug("--- Creating the DataLoader for Training ---")
    graph_train_loader = FewShotDataLoader(
        dataset=train_ds,
        batch_sampler=TaskBatchSampler(
            dataset_targets=train_ds.targets(),
            n_way=config.TRAIN_WAY,
            k_shot=config.TRAIN_SHOT,
            n_query=config.TRAIN_QUERY,
            epoch_size=config.TRAIN_EPISODE,
            shuffle=True,
            batch_size= 1# config.BATCH_PER_EPISODES
        )
    )

    logging.debug("--- Getting the First Sample ---")
    support, query = next(iter(graph_train_loader))
    print("\n- Support Sample Batch: ", support)
    print("- Query Sample Batch: ", query)
    print()

    # delete_data_folder()


def try_paper_dataset():
    node_attribute_file = "../TRIANGLES_node_attributes.pickle"
    train_set_file = "../TRIANGLES_train_set.pickle"

    import pickle
    import torch
    import numpy as np
    import random

    with open(node_attribute_file, mode="rb") as iostream:
        node_attribute_data = pickle.load(iostream)
        node_attribute_data = list(map(float, node_attribute_data))

    with open(train_set_file, mode="rb") as iostream:
        train_set_data = pickle.load(iostream)

    graph_indicator = dict()

    for index, node_list in train_set_data["graph2nodes"].items():
        for node in node_list:
            graph_indicator[node] = index
    
    num_graph = len(train_set_data["graph2nodes"])
    label2graphs = train_set_data["label2graphs"]
    graph2nodes = train_set_data["graph2nodes"]
    graph2edges = train_set_data["graph2edges"]

    def sample_classes():
        return random.sample(label2graphs.keys(), config.TRAIN_WAY)

    def sample_graphs_id(classes):
        support_graphs = []
        query_graphs = []
        support_labels = [] 
        query_labels = []

        for index, label in enumerate(classes):
            graphs = label2graphs[label]
            selected_graphs = random.sample(graphs, config.TRAIN_SHOT + config.TRAIN_QUERY)
            support_graphs.extend(selected_graphs[:config.TRAIN_SHOT])
            query_graphs.extend(selected_graphs[config.TRAIN_SHOT:])
            support_labels.extend([index] * config.TRAIN_SHOT)
            query_labels.extend([index] * config.TRAIN_QUERY)

        sindex=list(range(len(support_graphs)))
        random.shuffle(sindex)

        support_graphs=np.array(support_graphs)[sindex]
        support_labels=np.array(support_labels)[sindex]

        qindex=list(range(len(query_graphs)))
        random.shuffle(qindex)
        query_graphs=np.array(query_graphs)[qindex]
        query_labels=np.array(query_labels)[qindex]

        return np.array(support_graphs), np.array(query_graphs), np.array(support_labels), np.array(query_labels)
    
    def sample_graph_data(graph_ids):
        """
        :param graph_ids: a numpy shape n_way*n_shot/query
        :return:
        """
        print(graph_ids)
        edge_index=[]
        graph_indicator=[]
        node_attr=[]

        current = 0

        node_number=0
        for index,gid in enumerate(graph_ids):
            nodes=graph2nodes[gid]
            new_nodes=list(range(node_number,node_number+len(nodes)))
            node_number=node_number+len(nodes)
            node2new_number=dict(zip(nodes,new_nodes))

            current += np.array([node_attribute_data[node] for node in nodes]).reshape(len(nodes),-1).shape[0]

            node_attr.append(np.array([node_attribute_data[node] for node in nodes]).reshape(len(nodes),-1))
            edge_index.extend([[node2new_number[edge[0]],node2new_number[edge[1]]]for edge in graph2edges[gid]])
            graph_indicator.extend([index]*len(nodes))
        
        print(current)

        node_attr = np.vstack(node_attr)

        return [torch.from_numpy(node_attr).float(), \
               torch.from_numpy(np.array(edge_index)).long(), \
               torch.from_numpy(np.array(graph_indicator)).long()]
        
    def sample_episode():
        classes = sample_classes()
        support_graphs, query_graphs, support_labels, query_labels = sample_graphs_id(classes)

        support_data = sample_graph_data(support_graphs)
        support_labels = torch.from_numpy(support_labels).long()
        support_data.append(support_labels)

        query_data = sample_graph_data(query_graphs)
        query_labels = torch.from_numpy(query_labels).long()
        query_data.append(query_labels)

        return support_data, query_data


    support_data, query_data = sample_episode()

    support_data=[item.to("cuda") for item in support_data]
    query_data=[item.to("cuda") for item in query_data]

    (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
    (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

    print("Support Labels --- ", support_label)
    # print("Support Nodes --- ", support_nodes)
    print("Task Num --- ", support_nodes.size()[0])

    print("Query Labels --- ", query_label)
    print("Query Labels Size --- ", query_label.size()[0])

    print("Support Nodes - ith --- ", support_nodes[0])
    print("Support Edge Index - ith --- ", support_edge_index[0])
    print("Transposed Support Edge Index - ith --- ", support_edge_index[0].transpose(0, 1))
    print("Support Graph Indicator - ith --- ", support_graph_indicator[0])


if __name__ == "__main__":
    # main()
    try_paper_dataset()