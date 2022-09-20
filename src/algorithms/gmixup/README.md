# G-Mixup - Graph Data Augmentation using Mixup

This is an overview of the $\mathcal{G}$-Mixup GDA technique proposed in [G-Mixup: Graph Data Augmentation for Graph Classification](https://arxiv.org/pdf/2202.07179.pdf) by Xiaotian Han et al. 

Mixup improves generalization and robustness of neural networks by interpolating features and labels between two random samples. However, applying Mixup to graph data is quite challenging: 1) different graphs have different number of nodes; 2) different graphs are not really aligned; 3) different graphs have unique topologies in a non-Euclidean space. So, *how can we apply mixup to graph data*? Instead of directly manipulating graphs, we can interpolate so-called graphons in the Euclidean space to get mixed graphons and, finally, we sample new data from them. First, we need to recall that the formal math expression of Mixup is, given two data $(x_i, y_i)$ and $(x_j, y_j)$ the resulting new data would be $x_\text{new} = \lambda \cdot x_i + x_j \cdot (1 - \lambda)$ and $y_\text{new} = \lambda \cdot y_i + y_j \cdot (1 - \lambda)$, where $y_i, y_j$ and $y_\text{new}$ are One-Hot Encoded labels. 

---

## Proposed method

Given two different graph set $\mathcal{G}$ and $\mathcal{H}$ with different labels and topologies, we estimate graphons $W_\mathcal{G}$ and $W_\mathcal{H}$, then mixup and obtain $W_\mathcal{I}$ as additional training graphs. 

**Graph Homomorphism**. Adjacency-preserving mapping between two graphs, i.e., mapping adjacent vertices in one graph to adjacent vertices in another graph. Formally, $\phi : F \to G$ is a map from $V(F)$ to $V(G)$, where if $(u, v) \in E(F)$ then also $(\phi(u), \phi(v)) \in E(G)$. Let, now, denote $\text{hom}(G, H)$ the total number of graph homomorphism between those two graphs, in total we there are $|V(G)|^\text{|V(H)|}$ mapping, but only some of them are homomorphism. Thus, we can define *homomorphism density* as 

$$t(H, G) = \frac{\text{hom}(H, G)}{|V(G)|^\text{|V(H)|}}$$

**Graphons**. They are continuous, bounded and symmetric functions $W : [0,1]^2 \to [0, 1]$ such that given $u_i, u_j \in [0, 1]$ then $W(u_i, u_j)$ is the probability that the edge $(u_i, u_j)$ exists. Thanks to this, graph can be easily extended to a degree distribution function in graphons $d_w (x) = \int_0^1 W(x, y) \mathrm{d}x$. Similarly the concept of homomorphism density can be easily extended from graph to graphons. Given an arbitary motif $F$ its homomorphism density with respect to the function $W$ is defined as 

$$t(F, W) = \int_{[0,1]^\text{V(F)}} \prod_{(i, j) \in E(F)} W(x_i, x_j) \prod_{i \in V(F)}\mathrm{d}x_i$$