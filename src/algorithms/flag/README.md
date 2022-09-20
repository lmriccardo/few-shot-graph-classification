# FLAG - Adversarial Data Augmentation

This is an overview of the FLAG GDA technique from the [Robust Optimization as Data Augmentation for Large-Scale Graphs](https://arxiv.org/pdf/2010.09891.pdf) by Kezhi Kong et al.

Data augmentation helps neural networks generalize better by enlarging the training set, but it remains an open question how to effectively augment graph data to enhance the performance of GNNs. While most existing graph regularizers focus on manipulating graph topological structures by adding/removing edges, they offer a method to augment node features for better performance. They proposed FLAG (Free Large-scale Adversarial Augmentation on Graphs), which iteratively augments node features with gradient-based adversarial perturbations during training. By making the model invariant to small fluctuations in input data, their method help models generlize out-of-distribution samples and boost model performance at test time. 

---

## Proposed method

In this work they investigate how to effectively improve the generalization of GNNs through a feature-based augmentation. Graph node features are usually constructed as discrete embeddings, such as bag-of-words vectors or categorical variables. As a result, standard hand-crafted augmentations, like flipping and cropping transforms used in computer vision, are not applicable to node features. By hunting for and stamping out small perturbations that cause the classifier to fail, one may hope that adversarial training could benefit standard accuracy. 

**Min-Max Optimization**

Adversarial training is the process of crafting adversarial data points, and then injecting them into training data. This process is often formulated as the following min-max problem $\min_\theta \mathbb{E}[\max_{||\mathbf{\delta}|| \leq \epsilon} L( f_\theta (x + \mathbf{\delta}), y)]$ where the outer minimization uses the SGD, while the inner maximization uses the *Projected GD*. In practice the typical approximation of the inner under $l_\infty$-norm constraint is as follow

$$\delta_\text{t + 1} = \prod_{||\delta|| \leq \epsilon} (\delta_t + \alpha \cdot \text{sgn}(\nabla_\delta L(f_\theta(x + \delta_t), y)))$$

where the perturbation $\delta$ is updated iteratively, and $\prod_{||\delta|| \leq \epsilon}$ performs projection onto the $\epsilon$-ball in the $l_\infty$-norm. For maximum robustness, this iterative updating procedure usually loops $M$ times to craft the worst-case noise, which requires $M$ forward and backward passes end-to-end. Afterwards the most vicious noise $\delta_M$ is applied to the input feature, on which the model weight is optimized. 

**Multi-scale augmentation**

To fully exploit the generalizing ability and enhance the diversity and quality of adversarial perturbations, they proposed to craft multi-scale augmentations. To realize this goal, they leverage the so-called *Free training* technique. PGD is a powerful yet inefficient way of solving the min-max optimization. It runs $M$ full forward and backward passes to craft a refined perturbation $\delta_\text{1:M}$, but the model weight $\theta$ are optimized only with $\delta_M$. This process makes model training $M$ times slower. In contrast, while computing the gradient for the perturbation $\delta$, "free" training simultaneously produces the model parameter $\theta$ on the same backward pass. Note that $X$ is augmented with additive perturbations $\delta_\text{1:M}$, of which each can have a maximum scale of $m\cdot \alpha$, $m \in \lbrace 1, ..., M \rbrace$. This greatly adds to the diversity of the augmentation. However, the "free" algorithm is sub-optimal in terms of min-max optimization that during the batch-relay process, the approximated perturbation computed to maximize the objective $\theta_t$ is used to robustly optimize $\theta_\text{t + 1}$ rather than $\theta_t$. To tackle this problem, instead of directly updating $\theta$ using the "by-product" gradient attained from the gradient ascent step on $\delta$, we can accumulate the gradients and apply them to the model parameters all at once later. Formally, the optimization step is

$$\theta_\text{i + 1} = \theta_i - \frac{\tau}{M} \sum_{t = 1}^M \nabla_\theta L(f_\theta(x + \delta_t), y)$$

where $\tau$ is learning rate and $\delta_1$ is uniform noise.