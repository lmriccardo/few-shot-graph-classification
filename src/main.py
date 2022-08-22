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
    get_max_acc
)
from models.asmaml.asmaml import AdaptiveStepMAML
from models.asmaml.gcn4maml import GCN4MAML
from models.asmaml.sage4maml import SAGE4MAML

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
    model = SAGE4MAML(num_classes=config.TRAIN_WAY, paper=paper)# .to(config.DEVICE)
    meta_model = AdaptiveStepMAML(model,
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
        setup_seed(epoch)
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
            
            # for support in support_data:
            #     support.to(config.DEVICE)
            
            # for query in query_data:
            #     query.to(config.DEVICE)
            
            accs, step, stop_gates, scores, query_losses = meta_model.finetuning(support_data, query_data)
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
            print('Epoch(***Best***): {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                   'acc_val:{:.2f} ±{:.2f},meta_lr: {:.6f},best {:.2f}\n'.format(
                            epoch, train_loss_avg, train_acc_avg,
                            val_acc_avg, val_acc_ci95, meta_model.get_meta_learning_rate(),
                            max_val_acc
                        )# , file=open("../results/asmaml_gcn.result", mode="a")
            )

            # torch.save({'epoch': epoch, 'embedding':meta_model.state_dict(),
            #             # 'optimizer': optimizer.state_dict()
            #             }, os.path.join(config["save_path"], 'best_model.pth'))
        else :
            print('\nEpoch: {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                    'acc_val:{:.2f} ±{:.2f},meta_lr: {:.6f},best {:.2f}\n'.format(
                            epoch, train_loss_avg, train_acc_avg, val_acc_avg, 
                            val_acc_ci95, meta_model.get_meta_learning_rate(), max_val_acc
                        )# , file=open("../results/asmaml_gcn.result", mode="a")
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
    # train_ds, test_ds, val_ds, data_dir = generate_train_val_test(
    #     # data_dir=data_dir,
    #     download=config.DOWNLOAD_DATASET, # not config.DOWNLOAD_DATASET,
    #     dataset_name="TRIANGLES"
    # )

    data_dir = "../data"
    train_ds, test_ds, val_ds, data_dir = generate_train_val_test(
        data_dir=data_dir,
        download=not config.DOWNLOAD_DATASET,
        dataset_name="TRIANGLES"
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

    # logging.debug("--- Getting the First Sample ---")
    # support, query = next(iter(train_dataloader))
    # print("\n- Support Sample Batch: ", support)
    # print("- Query Sample Batch: ", query)
    # print("- Support Sample Graph Index: ", support.edge_index)
    # print()

    run_train(train_dataloader, validation_dataloader)

    # delete_data_folder(data_dir)


if __name__ == "__main__":
    main()

    # from utils.utils import load_with_pickle
    # print(load_with_pickle("../data/TRIANGLES/TRIANGLES_node_attributes.pickle"))