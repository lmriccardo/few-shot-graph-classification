# G-Mixup - Graph Data Augmentation using Mixup

This is an overview of the $\mathcal{G}$-Mixup GDA technique proposed in [G-Mixup: Graph Data Augmentation for Graph Classification](https://arxiv.org/pdf/2202.07179.pdf) by Xiaotian Han et al. 

Mixup improves generalization and robustness of neural networks by interpolating features and labels between two random samples. However, applying Mixup to graph data is quite challenging: 1) different graphs have different number of nodes; 2) different graphs are not really aligned; 3) different graphs have unique topologies in a non-Euclidean space. So, *how can we apply mixup to graph data*? Instead of directly manipulating graphs, we can interpolate so-called graphons in the Euclidean space to get mixed graphons and, finally, we sample new data from them. First, we need to recall that the formal math expression of Mixup is, given two data $(x_i, y_i)$ and $(x_j, y_j)$ the resulting new data would be $x_\text{new} = \lambda \cdot x_i + x_j \cdot (1 - \lambda)$ and $y_\text{new} = \lambda \cdot y_i + y_j \cdot (1 - \lambda)$, where $y_i, y_j$ and $y_\text{new}$ are One-Hot Encoded labels. 

---

## Proposed method

