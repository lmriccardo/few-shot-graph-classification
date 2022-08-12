"""
For the source of this code check-out https://github.com/NingMa-AI/AS-MAML

Algorithm:

function AS-MAML(
    Input:
        - Task distribution p(T) over {(G_train, y_train)}
    
    Paramters:
        - Graph embedding parameters "e"
        - Classifier parameters "c"
        - Step control parametrs "s"
        - Learning Rates a1, a2, a3
    
    Output:
        - Trained parameters
){
    initialize_random(a1, a2, a3)
    while (not converge) do
    {
        T_i = (D(train, sup), D(train, query)) <- random.sample(p(T))
        T <- get_adaptation_step()
        P' <- P = {e, c}  // Set fast adaptation parameters
        for (t=0 ... T) do
        {
            P' <- P' - a1 * Grad(P', L(T_i, f_P'))
            M_T <- compute_ANI()
            p(t) <- compute_stop_probability()
            Q(t) <- compute_reward(on=D(train, que))
        }

        P <- P - a2 * Grad(P', L(T_i, f_P'))
        for (t=0 ... T) do
        {
            s <- s + a3 * Q(t) * Grad(s, ln(p(t)))
        }
    }
}
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torch_geometric.data as gdata

from utils.utils import get_batch_number
import config


class AdaptiveStepMAML(nn.Module):
    """ The Meta-Learner Class """
    def __init__(self, model, inner_lr, outer_lr, stop_lr, weight_decay) -> None:
        self.net          = model
        self.inner_lr     = inner_lr
        self.outer_lr     = outer_lr
        self.stop_lr      = stop_lr
        self.weight_decay = weight_decay

        self.task_index = 1

        self.stop_prob = 0.5
        self.stop_gate = self.StopControl(config.STOP_CONTROL_INPUT_SIZE, config.STOP_CONTROL_HIDDEN_SIZE)

        self.meta_optim = self.configure_optimizers()

        self.loss      = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optim, mode="min", factor=0.5, 
            patience=config.PATIENCE, verbose=True, min_lr=1e-05
        )

        self.graph_embs = []
        self.graph_labels = []
        self.index = 1

    def compute_loss(self, logits, label):
        """Compute the CrossEntropyLoss"""
        loss = self.loss(logits, label)
        return loss

    def configure_optimizers(self):
        """Configure Optimizers"""
        return optim.Adam([
                           {'params': self.net.parameters(),       'lr': self.outer_lr},
                           {'params': self.stop_gate.parameters(), 'lr': self.stop_lr}],
                          lr=1e-04, weight_decay=self.weight_decay
               )
        
    def compute_loss(self, logits: torch.Tensor, label: torch.Tensor) -> float:
        return self.loss(logits.squeeze(), label.double().squeeze())

    def stop(self, step: int, loss: float, node_score: torch.Tensor):
        ...

    def forward(self, support_data: gdata.batch.Batch, query_data: gdata.batch.Batch):
        task_num = support_data.y.shape[0] // config.BATCH_PER_EPISODES
        query_size = query_data.y.shape[0]

        losses_q = []  # Losses on query data
        corrects, stop_gates, train_losses, train_accs, scores = [], [], [], [], []

        for i in range(task_num):
            i_batch = task_num + 1

            # Get the correct task batch from all the tasks batch
            support_data_task = get_batch_number(
                support_data, 
                i_batch=i_batch, 
                n_way=config.TRAIN_WAY,
                k_shot=config.TRAIN_SHOT
            )

            query_data_task = get_batch_number(
                query_data, 
                i_batch=i_batch, 
                n_way=config.TRAIN_WAY,
                k_shot=config.TRAIN_QUERY
            )

            fast_parameters = list(self.net.parameters())

            for weight in self.net.parameters():
                weight.fast = None
            
            step = 0
            self.stop_prob = 0.1 if self.stop_prob < 0.1 else self.stop_prob

            # Get adaptation step
            ada_step = min(config.MAX_STEP, config.MIN_STEP + int(1.0 / self.stop_prob))

            for k in range(0, ada_step):
                # Run the i-th task and compute the loss
                logits, score, _ = self.net(...)  # TODO: Change accordingly to data structure

                loss = self.compute_loss(logits, support_data_task.y)
                stop_probability = 0
                if config.FLEXIBLE_STEP:
                    stop_probability = self.stop(k, loss, score)
                    self.stop_prob = stop_probability
                
                stop_gates.append(stop_probability)
                scores.append(score.item())

            step = k
            train_losses.append(loss.item())

            # Compute the gradient with respect to the loss
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []
            for index, weight in enumerate(self.net.parameters()):
                if weight.fast:
                    weight.fast = weight - self.inner_lr * grad[index]
                else:
                    weight.fast = weight.fast - self.inner_lr * grad[index]
                
                fast_parameters.append(weight.fast)