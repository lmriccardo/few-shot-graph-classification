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

However, you can give whathever dataset you want. The only requirement is that it is in a specific form, i.e. the one previously descripted: a folder containing four files name as `<dataname>_node_attributes.txt`, `<dataname>_graph_labels.txt`, `<dataname>_graph_edges.txt` and `<dataname>_graph_indicator.txt`. It is preferred to have the pickle files, since their content changes and we already have pre-defined train, validation and test set. In the other case, in which you provide non-pickle files, you will have to transform those files using some utility functions stored in the `utils.py` file (more in the following section). 

---

## 3. Project structure

In this section I'm going to describe the structure of this project.

```bash
.
├── data                       # Contains the datasets (TRIANGLES, COIL-DEL, R52 and Letter-High)
├── models                     # Contains pre-trained models for each of the different tests done
├── src                        # Source files of the project
│   ├── algorithms             # Contains all the algorithm used in the project
│   │   ├── asmaml             # Contains code for the AS-MAML
│   │   │   ├── __init__.py    
│   │   │   ├── README.md       
│   │   │   └── asmaml.py      
│   │   ├── mevolve            # Contains code for M-Evolve
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   └── mevolve.py     
│   │   ├── flag               # Contains code for FLAG
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   └── flag.py
│   │   ├── gmixup             # Contains code for G-Mixup
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   └── gmixup.py
│   ├── data                   # Contains code for dataset, dataloader and sampler
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── sampler.py
│   │   └── dataloader.py
│   ├── models                 # Contains various convolutional layer and models
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

---

## 4. Usage

To run the project you will need to install all the dependencies, so the suggested procedure is to create a virtual environment first, with the command `python -m venv <venv_name>`, and then install the required libraries:

- `torch==1.12.1` or `torch==1.12.1+cu116` (or other versions of CUDA) (more [info](https://pytorch.org/get-started/locally/))
- `torch-geometric` (more [info](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html))
- `numpy` (latest)
- `matplotlib` (latest)
- `networkx` (latest)
- `sklearn` (latest)

Finally, to run the base project, i.e. entire training and testing with only AS-MAML, just provide to the command-line/terminal the following command

```bash
$> python main.py
``` 

For further options to better configuring the execution of the project, please use the `-h, --help` flag. It will gives you the following output, where you can see which options can be modified and for what. 

```bash
usage: main.py [-h] [-p PATH] [-n NAME] [-d DEVICE] [-l LOG_PATH] [-f] [-s SAVE_PATH] [-m MODEL] [--not-as-maml] [--gmixup] [--flag] [--mevolve]
               [--batch-size BATCH_SIZE] [--outer_lr OUTER_LR] [--inner_lr INNER_LR] [--stop_lr STOP_LR] [--w-decay W_DECAY] [--max-step MAX_STEP]
               [--min-step MIN_STEP] [--penalty PENALTY] [--train-shot TRAIN_SHOT] [--val-shot VAL_SHOT] [--train-query TRAIN_QUERY]
               [--val-query VAL_QUERY] [--train-way TRAIN_WAY] [--test-way TEST_WAY] [--val-episode VAL_EPISODE] [--train-episode TRAIN_EPISODE]
               [--batch-episode BATCH_EPISODE] [--epochs EPOCHS] [--patience PATIENCE] [--grad-clip GRAD_CLIP] [--scis SCIS] [--schs SCHS] [--beta BETA]
               [--n-fold N_FOLD] [--n-xval N_XVAL] [--iters ITERS] [--heuristic HEURISTIC] [--lrts LRTS] [--lrtb LRTB] [--flag-m FLAG_M] [--ass ASS]

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  The path of the dataset (default: /home/fscg/app/data)
  -n NAME, --name NAME  The name of the dataset (default: COIL-DEL)
  -d DEVICE, --device DEVICE
                        The device to use (default: cpu)
  -l LOG_PATH, --log-path LOG_PATH
                        The path where to log (default: None)
  -f, --file-log        If logging to file or not (default: False)
  -s SAVE_PATH, --save-path SAVE_PATH
                        The path where to save pre-trained models (default: /home/fscg/app/models)
  -m MODEL, --model MODEL
                        The name of the model (sage or gcn) (default: sage)
  --not-as-maml         Use AS-MAML or not (default: True)
  --gmixup              Use G-Mixup or not (default: False)
  --flag                Use FLAG or not (default: False)
  --mevolve             Use M-Evolve or not (default: False)
  --batch-size BATCH_SIZE
                        Dimension of a batch (default: 1)
  --outer_lr OUTER_LR   Initial LR for the model (default: 0.001)
  --inner_lr INNER_LR   Initial LR for the meta model (default: 0.01)
  --stop_lr STOP_LR     Initial LR for the Stop model (default: 0.0001)
  --w-decay W_DECAY     The Weight Decay for optimizer (default: 1e-05)
  --max-step MAX_STEP   The Max Step of the meta model (default: 15)
  --min-step MIN_STEP   The Min Step of the meta model (default: 5)
  --penalty PENALTY     Step Penality for the RL model (default: 0.001)
  --train-shot TRAIN_SHOT
                        The number of Shot per Training (default: 10)
  --val-shot VAL_SHOT   The number of shot per Validation (default: 10)
  --train-query TRAIN_QUERY
                        The number of query per Training (default: 15)
  --val-query VAL_QUERY
                        The number of query per Validation (default: 15)
  --train-way TRAIN_WAY
                        The number of way for Training (default: 3)
  --test-way TEST_WAY   The number of way for Test and Val (default: 3)
  --val-episode VAL_EPISODE
                        The number of episode for Val (default: 200)
  --train-episode TRAIN_EPISODE
                        The number of episode for Training (default: 200)
  --batch-episode BATCH_EPISODE
                        The number of batch per episode (default: 5)
  --epochs EPOCHS       The total number of epochs (default: 500)
  --patience PATIENCE   The patience (default: 35)
  --grad-clip GRAD_CLIP
                        The clipping for the gradient (default: 5)
  --scis SCIS           The input dimension for the Stop Control model (default: 2)
  --schs SCHS           The hidden dimension for the Stop Control model (default: 20)
  --beta BETA           The beta used in heuristics of M-Evolve (default: 0.15)
  --n-fold N_FOLD       The number of Fold for the nX-fol-validation (default: 5)
  --n-xval N_XVAL       Number of Cross-fold Validation to run (default: 10)
  --iters ITERS         Number of iterations of M-Evolve (default: 5)
  --heuristic HEURISTIC
                        The Heuristic to use (default: random_mapping)
  --lrts LRTS           The label reliability step thresholds (default: 1000)
  --lrtb LRTB           The beta used for approximation of the tanh (default: 30)
  --flag-m FLAG_M       The number of iterations of FLAG (default: 3)
  --ass ASS             The attack step size (default: 0.008)
```

### 4.1. Docker

Alternatively, I have already created a [Docker Image](https://hub.docker.com/repository/docker/lmriccardo/fsgc) that can be pulled with `docker pull lmriccardo/fsgc:1.0a`. Then, you need to run the container with `docker run --rm -it lmriccardo/fsgc:1.0a` and, finally, run the same python command given above: `python main.py`. 

---

## 5. Algorithms and Models

As I said the goal of this projects is to compare different Graph Data Augmentation techniques for few-shot learning, and more precisely for few-shot classification. For this reason all the techniques that have been chosen regard augmentation for classification, i.e. techniques that try to preserve structural properties of the original graph. The overall idea is to generate new data for already defined labels, this means without additionally labeling those new data, by using some procedures of dropping edge/nodes, change in node features or randomly generate graphs based on so-called *graphons*. To this end I decided to use this three GDA technique, one per type:

- **Model-Evolution**. GDA technique that uses edge dropping based on motifs similarity ([paper](https://arxiv.org/pdf/2007.05700.pdf))
- **FLAG**. GDA technique that uses perturbation attacks to perturb node features ([paper](https://arxiv.org/pdf/2010.09891.pdf))
- **G-Mixup**. GDA technique that uses Mixup on graphs via graphons ([paper](https://arxiv.org/pdf/2202.07179.pdf))

For quick further informations about each of the three techniques I suggest to have a look to their respectively README that you can found at `./src/algorithms/mevolve` (for M-Evolve), `./src/algorithms/flag` (for FLAG) and `./src/algorithms/gmixup` (for G-Mixup). 