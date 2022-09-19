# M-Evolve - Graph Data Augmentation

This is an overview on the M-Evolve GDA techniques from the [Data Augmentation for Graph Classification](https://arxiv.org/pdf/2007.05700.pdf) paper (Zhou et al.). 

Data limitations makes easy to fall into overfitting. To solve this problem, they take an effective approach to study data augmentation on graphs and develop two graph augmentation methos, called *random mapping* and *motif-similarity mapping*, respectively. The idea is to generate more virtual data for small datasets via heuristic modifications of graph structure. Since this new graph are artificial and treated as weakely labeled data, thier reliability remains to be verified. For this reason, they introduce also a data filtering procedure to filter augmented data based on label reliability. 

---

## Problem definition 

Let $\mathcal{G} = (V, E)$ be a generic graph, where $V$ is the set of nodes and $E$ is the set of edges. $\mathcal{G}$ is, in general, represented using the adjacency matrix $\mathbf{A}[n\times n]$ such that $\mathbf{A}[i,j] = 1$ if $(i,j) \in E$ and $\mathbf{A}[i,j] = 0$ otherwise. The overall dataset $\mathcal{D}$ is composed of pairs $(\mathcal{G}_i,\mathbf{y}_i)$ for each $i \in [0, N]$, and it is splitted into train set $\mathcal{D}^\text{train}$, validation set $\mathcal{D}^\text{val}$ and test set $\mathcal{D}^\text{test}$. Finally, we have a classifier $\mathcal{C}$ pre-trained on the train and validation dataset. The goal is to update the classifier $\mathcal{C}$ with augmented data, which are first generated via GDA and then filtered via label reliability. During GDA, we want to map a graph $\mathcal{G} \in \mathcal{D}^\text{train}$ to a new graph $\mathcal{G}'$ with a function $f : (\mathcal{G}, \mathbf{y}) \to (\mathcal{G}', \mathbf{y})$. Then classify the new generated graphs into two groups via label reliability threshold $\theta$, learnt from the validation set. Finally, the new filtered train set $\mathcal{D'}^\text{train}$ is merged with the original train set: $\mathcal{D}^\text{new,train} = \mathcal{D}^\text{train} + \mathcal{D'}^\text{train}$. Then, we finetune the classifier with the new train set and evaluate using the validation set.

---

## Methodology

In this section I'm going to describe which are the methodologies for generating new graph data and for filtering them. The filter operations is very important since provide us a reliability score for each of the newly "constructed" data. 

### Data Generation methods

M-Evolve is based on two heuristics: *random similarity mapping* and *motifs similarity mapping*. Although the former is just a baseline comparision, and for this reason we are going to focus much more on the second, I would like to give you a quick overview of the first one. From now on we are going to call $E^c_\text{del}$ and $E^c_\text{add}$ the set of nodes from which egdes are sampled to be respectively deleted/added. 

**Random mapping**

In this case we set $E^c_\text{del} = E$ and $E^c_\text{add} = \lbrace (v_i, v_j) | \mathbf{A}[i,j] = 0, i \neq j \rbrace$. Finally, we sample $E_\text{del} = \lbrace e_i | i = 1, ..., \lceil m \cdot \beta \rceil \rbrace \subset E^c_\text{del}$ and $E_\text{add} = \lbrace e_i | i = 1, ..., \lceil m \cdot \beta \rceil \rbrace \subset E^c_\text{add}$ and construct the new graph $\mathcal{G}' = (V, (E \cup E_\text{add}) \backslash E_\text{del})$. We can easily find out why this is considered just a baseline: it is random. Due to its randomness structural properties of the original graphs would not be preserved. 

**Motifs-similarity mapping**

This heuristic is based on so-called *motifs*: sub-graphs that repeat themselves in a specific graph or even among various graphs. Each of these sub-graphs, defined by a particular pattern of interactions between vertices, may describe a framework in which particular functions are achieved efficiently. In this case we are considering so-called *open-triads*: lenght-2 paths that induce a triangle. Essentially, open-triad are triples of nodes $(v_i, v_z, v_j)$ connected together by just two edges $(v_i, v_z)$ and $(v_z, v_j)$, such that if the third edge $(v_i, v_j)$ is inserted then we would build a triangle. In this context we call $v_i$ as **head vertex** and $v_j$ as **tail vertex**. Using this motifs we can preserves structural properties. Now, we need to recall that if $\mathbf{A}$ is the adjacency matrix, then $\mathbf{A}^2$ is a matrix such that $\mathbf{A}[i,i] = \text{deg}(v_i)$ while $\mathbf{A}^2[i,j]$ is the number of lenght-2 path connecting $v_i$ and $v_j$. Then, we fill $E^c_\text{add}$ with pairs of head vertex and tail vertex for each open-triad in the graph, i.e., $$E^c_\text{add} = \lbrace (v_i, v_j) | \mathbf{A}[i,j] = 0, \mathbf{A}^2[i,j] \neq 0, i \neq j \rbrace$$
Finally, for what concerning adding edges, we construct $E_\text{add}$ by weight random sampling from $E^c_\text{add}$. The question now is, *How do I sample?* To this end, weights are given by a vertex similarity score using *Resource Allocation Index*: for each $(v_i, v_j)$ the RAI $s_\text{i,j}$ and the respective weight $w_\text{i,j}$ can be computed as


$$s_\text{i,j} = \sum_\{ciao\}$$