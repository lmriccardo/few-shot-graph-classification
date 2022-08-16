import torchnet as tnt
import random
import numpy as np
import pickle
import torch.utils.data as data
import torch
import tqdm


def setup_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class GraphDataset(data.Dataset):
    node_attribute_file = "../TRIANGLES_node_attributes.pickle"
    train_set_file = "../TRIANGLES_train_set.pickle"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.node_attribute_data = []
        self.graph_indicator = dict()
        self.train_set_data = dict()

        with open(self.node_attribute_file, mode="rb") as iostream:
            self.node_attribute_data = pickle.load(iostream)
            self.node_attribute_data = list(
                map(float, self.node_attribute_data))

        with open(self.train_set_file, mode="rb") as iostream:
            self.train_set_data = pickle.load(iostream)

        for index, node_list in self.train_set_data["graph2nodes"].items():
            for node in node_list:
                self.graph_indicator[node] = index

        self.num_graph = len(self.train_set_data["graph2nodes"])
        self.label2graphs = self.train_set_data["label2graphs"]
        self.graph2nodes = self.train_set_data["graph2nodes"]
        self.graph2edges = self.train_set_data["graph2edges"]

    def __getitem__(self, index):
        return self.graph_indicator[index]

    def __len__(self) -> int:
        return len(self.label2graphs)


class FewShotDataLoader:
    def __init__(self,
                 dataset: GraphDataset,
                 n_way: int = 5,
                 k_shot: int = 5,
                 n_query: int = 5,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 epoch_size: int = 2000
                 ) -> None:
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size

    def sample_classes(self):
        return random.sample(self.dataset.label2graphs.keys(), self.n_way)

    def sample_graphs_id(self, classes):
        support_graphs = []
        query_graphs = []
        support_labels = []
        query_labels = []

        for index, label in enumerate(classes):
            graphs = self.dataset.label2graphs[label]
            selected_graphs = random.sample(graphs, self.k_shot + self.n_query)
            support_graphs.extend(selected_graphs[:self.k_shot])
            query_graphs.extend(selected_graphs[self.k_shot:])
            support_labels.extend([index] * self.k_shot)
            query_labels.extend([index] * self.n_query)

        sindex = list(range(len(support_graphs)))
        random.shuffle(sindex)

        support_graphs = np.array(support_graphs)[sindex]
        support_labels = np.array(support_labels)[sindex]

        qindex = list(range(len(query_graphs)))
        random.shuffle(qindex)
        query_graphs = np.array(query_graphs)[qindex]
        query_labels = np.array(query_labels)[qindex]

        return np.array(support_graphs), np.array(query_graphs), np.array(support_labels), np.array(query_labels)

    def sample_graph_data(self, graph_ids):
        """
        :param graph_ids: a numpy shape n_way*n_shot/query
        :return:
        """
        edge_index = []
        graph_indicator = []
        node_attr = []

        current = 0

        node_number = 0
        for index, gid in enumerate(graph_ids):
            nodes = self.dataset.graph2nodes[gid]
            new_nodes = list(range(node_number, node_number+len(nodes)))
            node_number = node_number+len(nodes)
            node2new_number = dict(zip(nodes, new_nodes))

            current += np.array([self.dataset.node_attribute_data[node]
                                for node in nodes]).reshape(len(nodes), -1).shape[0]

            node_attr.append(np.array(
                [self.dataset.node_attribute_data[node] for node in nodes]).reshape(len(nodes), -1))
            edge_index.extend([[node2new_number[edge[0]], node2new_number[edge[1]]]
                              for edge in self.dataset.graph2edges[gid]])
            graph_indicator.extend([index]*len(nodes))

        node_attr = np.vstack(node_attr)

        return [torch.from_numpy(node_attr).float(),
                torch.from_numpy(np.array(edge_index)).long(),
                torch.from_numpy(np.array(graph_indicator)).long()]

    def sample_episode(self):
        classes = self.sample_classes()
        support_graphs, query_graphs, support_labels, query_labels = self.sample_graphs_id(
            classes)

        support_data = self.sample_graph_data(support_graphs)
        support_labels = torch.from_numpy(support_labels).long()
        support_data.append(support_labels)

        query_data = self.sample_graph_data(query_graphs)
        query_labels = torch.from_numpy(query_labels).long()
        query_data.append(query_labels)

        return support_data, query_data

    def load_function(self, iter_idx):
        support_data, query_data = self.sample_episode()
        return support_data, query_data

    def get_iterator(self, epoch: int = 0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=self.load_function
        )

        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

        return data_loader

    def __call__(self, epoch: int = 0):
        return self.get_iterator(epoch)

    def __len__(self) -> int:
        return int(self.epoch_size / self.batch_size)


def main():
    setup_seed()

    dataset = GraphDataset()
    loader = FewShotDataLoader(
        dataset=dataset,
        n_way=3,
        k_shot=10,
        n_query=15,
        batch_size=1,
        num_workers=4,
        epoch_size=200
    )

    sample = next(iter(loader(1)))
    support_data, query_data = sample

    (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
    (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

    print("Support Labels --- ", support_label)
    print("Support Labels Size --- ", support_label.size())

    print("Support Nodes --- ", support_nodes)
    print("Support Nodes Size --- ", support_nodes.size())
    print("Task Num --- ", support_nodes.size()[0])

    print("Query Labels --- ", query_label)
    print("Query Labels Size --- ", query_label.size()[1])

    print("Support Nodes - ith --- ", support_nodes[0])
    print("Support Nodes Size - ith --- ", support_nodes[0].size())

    print("Support Edge Index - ith --- ", support_edge_index[0])
    print("Support Edge Index Size - ith --- ", support_edge_index[0].size())
    print("Transposed Support Edge Index - ith --- ", support_edge_index[0].transpose(0, 1))

    print("Support Graph Indicator - ith --- ", support_graph_indicator[0])
    print("Support Graph Indicator Size - ith --- ", support_graph_indicator[0].size())
    
    # for i, data in enumerate(tqdm.tqdm(loader(1)), 1):
    #     support_data, query_data = data

    #     (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
    #     (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

    #     print("Task Num --- ", support_nodes.size()[0])


if __name__ == "__main__":
    main()