# FLAG - Adversarial Data Augmentation

This is an overview of the FLAG GDA technique from the [Robust Optimization as Data Augmentation for Large-Scale Graphs](https://arxiv.org/pdf/2010.09891.pdf) by Kezhi Kong et al.

Data augmentation helps neural networks generalize better by enlarging the training set, but it remains an open question how to effectively augment graph data to enhance the performance of GNNs. While most existing graph regularizers focus on manipulating graph topological structures by adding/removing edges, they offer a method to augment node features for better performance. They proposed FLAG (Free Large-scale Adversarial Augmentation on Graphs), which iteratively augments node features with gradient-based adversarial perturbations during training. By making the model invariant to small fluctuations in input data, their method help models generlize out-of-distribution samples and boost model performance at test time. 

---

## Proposed method

In this work they investigate how to effectively improve the generalization of GNNs through a feature-based augmentation. Graph node features are usually constructed as discrete embeddings, such as bag-of-words vectors or categorical variables. As a result, standard hand-crafted augmentations, like flipping and cropping transforms used in computer vision, are not applicable to node features. By hunting for and stamping out small perturbations that cause the classifier to fail, one may hope that adversarial training could benefit standard accuracy. 

**Min-Max Optimization**

Adversarial training is the process of crafting adversarial data points, and then injecting them into training data. This process is often formulated as the following min-max problem

$$\min_\theta \mathbb{E}[\max_{||\mathbf{\delta}||_p \leq \epsilon} \mathcal{L}(f_\theta (x + \mathbf{delta}), y)]$$