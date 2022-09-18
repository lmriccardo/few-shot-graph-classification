# Few-Shot Graph Classification

In this project, of the Deep Learning and applied AI, I want to explore graph classification tasks when training data is not enough in order to build an accurate model, using standard Deep Learning technique. More precisely, I would like to give a brief description of Few-Shot Learning (FSL) and *Meta-Learning* (ML). Finally, I'm going to present some approaches in few-shot learning. First, a *Meta-Learning Framework* based on Fast Weight Adaptation and MAML (Model-Agnostic Meta-Learner), taken from the paper [Adaptive-Step Graph Meta-Learner for Few-Shot Graph Classification](https://arxiv.org/pdf/2003.08246.pdf) (Ning Ma et al.). Second, I'm going to compre it with different GDA (graph data augmentation) techniques used to enrich the dataset for the novel classes (i.e., those with the less amount of data) taken from a second paper named [Graph Data Augmentation for Graph Machine Learning: A Survey](https://arxiv.org/pdf/2202.08871.pdf) (Tong Zhao et al.). 

---

## 1. Introduction

**Few-Shot Learning**

Most of the graph classification task overlook the scarcity of labeled graph in many situations. To overcome this problem, *Few-Shot Learning* is started being used. It is a type of machine learning method where the training data contains limited information. The general practice is to feed the machine learning model with as much data as possible, since this leads to better predictions. However, FSL aims to build accurate machine learning models with less training data. FSL aims to reduce the cost of gain and label a huge amount of data. In this project I'm going to concentrate on *few-shot graph classification*.

*Which is the idea behind Few-Shot Classification*? Let's say we have the train, validation and test set where train/validation and test set do not share any label for their data. First we sample $N$ class from those of the train/validation set and then for each class we sample $K + Q$ sample, for a total of $N \times (K + Q)$ graphs. The first $N \times K$ samples are called **support set**, while the latter $N \times Q$ composed the **query set**.  Given labeled support data, the goal is to predict the labels of query data. Note that in a single task, support data and query data share the same class space. This is also called **N-way-K-shot** learning. At test stage when performing classification tasks on unseen classes, we firstly fine tune the learner on the support data of test classes, then report classification performance on the test query set. You can find more about FSL in [A Survey on Few-Shot Learning](https://arxiv.org/pdf/1904.05046.pdf) (YAQING WANG et al.). 

**Meta-Learning**

Humans learn really quickly from few examples, but what can we say about computers? In particular we can easily classify different objects of the real-world just after having seen very few examples, however current deep learning methods needs a huge amount of information in order to create a very precise model. Moreover, what if the test set has classes that we do not have in the training set? Or what if we want to test the model on a completely different task? Meta-Learning offers solutions to these situations. Why? It is also known as *learn-to-learn*: the goal is, obviously to learn a model that correctly classify already seen samples, but also to learn a model that quickly adapt to new classes and/or tasks with few samples. One of the most famous meta-learner is the so-called [MAML](https://arxiv.org/pdf/1703.03400.pdf).

---

## 2. Used Datasets

I decided to use the same datasets considered in the paper for AS-MAML: TRIANGLES, COIL-DEL, R52 and Letter-High. All of them can be downloaded directly from this [page](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets), which is the origin of these datasets. Downloading from the previous page will result in a ZIP file with: 

- `<dataname>_node_attributes.txt` with the attribute vector for each node of each graph
- `<dataname>_graph_labels.txt` with the class for each graph
- `<dataname>_graph_edges.txt` with the edges for each graph expressed as a pair (node x, node y)
- `<dataname>_graph_indicator.txt` that maps each nodes to its corresponding graph

Each of the dataset has been splitted into *train*, *test* and *validation*, and transformed into a python dictionaries finally saved as `.pickle` files. In this way we have a ready-to-be-used dataset. Moreover, each ZIP dataset containes three files:

- `<dataname>_node_attributes.pickle` with the node attributes saved as a List or a torch Tensor
- `<dataname>_train_set.pickle` with all the train data as python dictionaries
- `<dataname>_test_set.pickle` with all the test data as python dictionaries
- `<dataname>_val_set.pickle` with all the validation data as python dictionaries

These are the link from which you can download the datasets: [TRIANGLES](https://drive.google.com/drive/folders/1Ghdi2dwoqMsqrAwxz4bYZrZI7Y8-B6In?usp=sharing), [COIL-DEL](https://drive.google.com/drive/folders/1m3frg5_MPOPPEoTJO7aSGDKMh-nqOOHL?usp=sharing), [R52](https://drive.google.com/drive/folders/158WZsLUMBBUJRR_RdbHY3I3Ea2yPU8lW?usp=sharing) and [Letter-High](https://drive.google.com/drive/folders/1573PBEW0R8xyZnkpcEBMht2l4p2jbbkm?usp=sharing).

---

## 3. Project structure

In this section I'm going to describe the structure of this project.

```bash
.
├── data                       # Contains the datasets (TRIANGLES, COIL-DEL, R52 and Letter-High)
├── models                     # Contains pre-trained models for each of the different tests done
├── src                        # Source files of the project
│   ├── algorithms             # Contains all the algorithm used in the project
│   │   ├── asmaml
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   └── asmaml.py
│   │   ├── mevolve
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   └── mevolve.py
│   │   ├── flag
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   └── flag.py
│   │   ├── gmixup
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   └── gmixup.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── sampler.py
│   │   └── dataloader.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── conv.py
│   │   ├── gcn4maml.py
│   │   ├── linear.py
│   │   ├── nis.py
│   │   ├── pool.py
│   │   ├── sage4maml.py
│   │   ├── stopcontrol.py
│   │   └── utils.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── kfold.py
│   │   ├── testers.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── config.py
│   └── main.py
├── fsgc.ipynb                 # The notebook of the project (ready-to-go)
├── LICENSE
└── README.md
```