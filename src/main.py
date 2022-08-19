import sys
import os

from data.dataset import GraphDataset
sys.path.append(os.getcwd())

from data.dataset import generate_train_val_test
from data.dataloader import FewShotDataLoader
from data.sampler import TaskBatchSampler
from utils.utils import (
    delete_data_folder, setup_seed, 
    get_batch_number, elapsed_time, 
    get_max_acc, load_with_pickle,
    GeneratorTxt2Graph, save_with_pickle
)
from models.asmaml.asmaml import AdaptiveStepMAML
from models.asmaml.gcn4maml import GCN4MAML

import config
import logging
from tqdm import tqdm
import numpy as np

import torch
torch.set_printoptions(edgeitems=config.EDGELIMIT_PRINT)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@elapsed_time
def run_train(train_dl: FewShotDataLoader, val_dl: FewShotDataLoader, paper: bool=False):
    """Run the training for optimization"""
    # Model creation
    gcn4maml = GCN4MAML(num_classes=config.TRAIN_WAY, paper=paper)# .to(config.DEVICE)
    meta_model = AdaptiveStepMAML(gcn4maml,
                                  inner_lr=config.INNER_LR,
                                  outer_lr=config.OUTER_LR,
                                  stop_lr=config.STOP_LR,
                                  weight_decay=config.WEIGHT_DECAY,
                                  paper=paper)# .to(config.DEVICE)

    write_count = 0
    val_count = 0
    max_val_acc = 0
    max_score_val_acc = 0

    print("=" * 100)

    for epoch in range(config.EPOCHS):
        logging.debug(f"--- Starting Epoch N. {epoch + 1} ---")
        loss_train = 0.0
        correct = 0
        
        meta_model.train()
        train_accs, train_final_losses, train_total_losses, val_accs, val_losses = [], [], [], [], []
        score_val_acc = []

        logging.debug("--- Starting Training ---")

        for i, data in enumerate(tqdm(train_dl)):
            support_data, query_data = data

            # Set support and query data to the GPU
            # support_data = support_data.to(config.DEVICE)
            # query_data = query_data.to(config.DEVICE)
            
            accs, step, final_loss, total_loss, stop_gates, scores, train_losses, train_accs_support = meta_model(
                support_data, query_data
            )

            train_accs.append(accs[step])
            train_final_losses.append(final_loss)
            train_total_losses.append(total_loss)

            if (i + 1) % 100 == 0:
                if np.sum(stop_gates) > 0:
                    print("\nstep", len(stop_gates), np.array(stop_gates))
                
                print("accs {:.6f}, final_loss {:.6f}, total_loss {:.6f}".format(
                    np.mean(train_accs), np.mean(train_final_losses), np.mean(train_total_losses)
                ))

        logging.debug("--- Ended Training ---")

        # validation step
        logging.debug("--- Starting Validation Phase ---")
        meta_model.eval()
        for i, data in enumerate(tqdm(val_dl)):
            support_data, query_data = data
            
            for support in support_data:
                support.to(config.DEVICE)
            
            for query in query_data:
                query.to(config.DEVICE)
            
            accs, step, stop_gates, scores, query_losses = meta_model.finetunning(support_data, query_data)
            acc = get_max_acc(accs, step, scores, config.MIN_STEP, config.MAX_STEP)

            val_accs.append(accs[step])
            if (i + 1) % 200 == 0:
                print("\n{}th test".format(i))
                if np.sum(stop_gates)>0:
                    print("stop_prob", len(stop_gates), np.array(stop_gates))

                print("scores", len(scores), np.array(scores))
                print("query_losses", len(query_losses), np.array(query_losses))
                print("accs", step, np.array([accs[i] for i in range(0, step + 1)]))
        
        val_acc_avg = np.mean(val_accs)
        train_acc_avg = np.mean(train_accs)
        train_loss_avg = np.mean(train_final_losses)
        val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(config.VAL_EPISODE)

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            logging.debug('\nEpoch(***Best***): {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                   'acc_val:{:.2f} ±{:.2f},meta_lr: {:.6f},best {:.2f}'.format(
                            epoch, train_loss_avg, train_acc_avg,
                            val_acc_avg, val_acc_ci95, meta_model.get_meta_learning_rate(),
                            max_val_acc
                        )
            )

            # torch.save({'epoch': epoch, 'embedding':meta_model.state_dict(),
            #             # 'optimizer': optimizer.state_dict()
            #             }, os.path.join(config["save_path"], 'best_model.pth'))
        else :
            logging.debug('\nEpoch: {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                    'acc_val:{:.2f} ±{:.2f},meta_lr: {:.6f},best {:.2f}'.format(
                            epoch, train_loss_avg, train_acc_avg, val_acc_avg, 
                            val_acc_ci95, meta_model.get_meta_learning_rate(), max_val_acc
                        )
            )

        meta_model.adapt_meta_learning_rate(train_loss_avg)

    print('Optimization Finished!')


def get_dataloader(
    ds: GraphDataset, n_way: int, k_shot: int, n_query: int, 
    epoch_size: int, shuffle: bool, batch_size: int
) -> FewShotDataLoader:
    """Return a dataloader instance"""
    return FewShotDataLoader(
        dataset=ds,
        batch_sampler=TaskBatchSampler(
            dataset_targets=ds.targets(),
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            epoch_size=epoch_size,
            shuffle=shuffle,
            batch_size=batch_size
        )
    )


def main():
    setup_seed()

    train_ds, test_ds, val_ds = generate_train_val_test(
        download_data=False, 
        perc_train=30, 
        perc_test=40
    )

    logging.debug("--- Datasets ---")
    print("\n- Train: ", train_ds)
    print("- Test : ", test_ds)
    print("- Validation: ", val_ds)
    print()

    logging.debug("--- Creating the DataLoader for Training ---")
    train_dataloader = get_dataloader(
        ds=train_ds, n_way=config.TRAIN_WAY, k_shot=config.TRAIN_SHOT,
        n_query=config.TRAIN_QUERY, epoch_size=config.TRAIN_EPISODE,
        shuffle=True, batch_size=1
    )

    logging.debug("--- Creating the DataLoader for Validation ---")
    validation_dataloader = get_dataloader(
        ds=val_ds, n_way=config.TEST_WAY, k_shot=config.VAL_SHOT,
        n_query=config.VAL_QUERY, epoch_size=config.VAL_EPISODE,
        shuffle=True, batch_size=1
    )

    logging.debug("--- Getting the First Sample ---")
    # support, query = next(iter(train_dataloader))
    # print("\n- Support Sample Batch: ", support)
    # print("- Query Sample Batch: ", query)
    # print("- Support Sample Graph Index: ", support.edge_index)
    # print()

    #run_train(train_dataloader, validation_dataloader)

    # delete_data_folder()


def func():
    setup_seed()

    from typing import Dict, Tuple, List
    import networkx as nx
    import random

    def compute_num_nodes(graph_list: Dict[str, Tuple[nx.Graph, str]]):
        num_nodes = 0
        for _, (graph, _) in graph_list.items():
            num_nodes += graph.number_of_nodes()
        
        return num_nodes

    def get(graph_list: Dict[str, Tuple[nx.Graph, str]], 
            num_nodes: int
    ) -> Tuple[Dict[str, dict], torch.Tensor]:
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
    
    def split_train_validation(train_data: Dict[str, dict], train_num_graphs_perc: float=70.0):
        """Split the train into train set and validation set"""
        remaining_graphs = torch.tensor(list(train_data["label2graphs"].values())) \
                                .view(1, -1)[0]
        
        total_graphs = remaining_graphs.shape[0]
        total_train_graphs_number = int(total_graphs * train_num_graphs_perc // 100)
        train_graphs = random.sample(remaining_graphs.tolist(), total_train_graphs_number)
        
        train_label2graphs = dict()
        validation_label2graphs = dict()
        for label, graphs in train_data["label2graphs"].items():
            train_label2graphs[label] = []
            validation_label2graphs[label] = []
            for graph in graphs:
                if graph in train_graphs:
                    train_label2graphs[label].append(graph)
                else:
                    validation_label2graphs[label].append(graph)
        
        train_graph2nodes  = {k : v for k, v in train_data["graph2nodes"].items() if k in train_graphs}
        train_graph2edges  = {k : v for k, v in train_data["graph2edges"].items() if k in train_graphs}

        validation_graph2nodes  = {k : v for k, v in train_data["graph2nodes"].items() if k not in train_graphs}
        validation_graph2edges  = {k : v for k, v in train_data["graph2edges"].items() if k not in train_graphs}

        train_data = {
            "label2graphs" : train_label2graphs,
            "graph2nodes"  : train_graph2nodes,
            "graph2edges"  : train_graph2edges
        }

        validation_data = {
            "label2graphs" : validation_label2graphs,
            "graph2nodes"  : validation_graph2nodes,
            "graph2edges"  : validation_graph2edges
        }

        return train_data, validation_data

        
    graph_attribute = []
    graph_labels = open("../data/Letter-High/Letter-high_graph_labels.txt").readlines()
    graph_node_attribute = open("../data/Letter-High/Letter-high_node_attributes.txt").readlines()
    graph_indicator = open("../data/Letter-High/Letter-high_graph_indicator.txt").readlines()
    graph_a = open("../data/Letter-High/Letter-high_A.txt").readlines()

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

    num_nodes = compute_num_nodes(graphs)
    final_data, attributes = get(graph_list=graphs, num_nodes=num_nodes)
    train_data, test_data = split(final_data)
    train_data, validation_data = split_train_validation(train_data)

    print(test_data, file=open("../test_data.txt", mode="w"))
    print(train_data, file=open("../train_data.txt", mode="w"))
    print(validation_data, file=open("../val_data.txt", mode="w"))

    save_with_pickle("../data/Letter-High/Letter-High_test_set.pickle", test_data)
    save_with_pickle("../data/Letter-High/Letter-High_train_set.pickle", train_data)
    save_with_pickle("../data/Letter-High/Letter-High_val_set.pickle", validation_data)
    save_with_pickle("../data/Letter-High/Letter-High_node_attributes.pickle", attributes)


if __name__ == "__main__":
    # main()
    func()