import torch
import time
from tqdm import tqdm
import numpy as np
import os
import config
from algorithms.asmaml.asmaml1 import AdaptiveStepMAML
from models.sage4maml import SAGE4MAML
import paper

from data.dataset import get_dataset
from data.dataloader import get_dataloader
from utils.utils import configure_logger


def get_max_acc(accs,step,scores,min_step,test_step):
    step=np.argmax(scores[min_step-1:test_step])+min_step-1
    return accs[step]


def run(config, meta_model, train_loader, val_loader):
    device=config["device"]
    t = time.time()
    max_val_acc=0
    for epoch in range(config["epochs"]):
        meta_model.train()
        train_accs, train_final_losses,train_total_losses, val_accs, _ = [], [], [], [],[]
        for i, data in enumerate(tqdm(train_loader(epoch)), 1):
            support_data, _, query_data, _ =data
            # support_data=[item.to(device) for item in support_data]

            if config["double"] == True:
                support_data[0]=support_data[0].double()
                query_data[0] = query_data[0].double()

            # query_data=[item.to(device) for item in query_data]
            accs,step,final_loss,total_loss,stop_gates,scores,_, _ = meta_model(support_data, query_data)
            train_accs.append(accs[step])

            train_final_losses.append(final_loss)
            train_total_losses.append(total_loss)
            #
            if (i+1)%100==0:
                if np.sum(stop_gates) > 0:
                    print("\nstep",len(stop_gates),np.array(stop_gates))
                print("accs{:.6f},final_loss{:.6f},total_loss{:.6f}".format(np.mean(train_accs),np.mean(train_final_losses),
                                                     np.mean(train_total_losses)))
        # validation_stage
        meta_model.eval()
        for i, data in enumerate(tqdm(val_loader(epoch)), 1):
            support_data, _, query_data, _ = data

            if config["double"]==True:
                support_data[0] = support_data[0].double()
                query_data[0] = query_data[0].double()

            # support_data = [item.to(device) for item in support_data]
            # query_data = [item.to(device) for item in query_data]

            accs, step, stop_gates, scores, query_losses = meta_model.finetuning(support_data, query_data)

            val_accs.append(accs[step])
            # train_losses.append(loss)
            if (i+1) % 200 == 0:
                print("\n{}th test".format(i))
                if np.sum(stop_gates)>0:
                    print("stop_prob", len(stop_gates), np.array(stop_gates))
                print("scores", len(scores), np.array(scores))
                print("query_losses", len(query_losses), np.array(query_losses))
                print("accs", step, np.array([accs[i] for i in range(0, step + 1)]))
        val_acc_avg=np.mean(val_accs)
        train_acc_avg=np.mean(train_accs)
        train_loss_avg =np.mean(train_final_losses)
        val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(config["val_episode"])

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            print('\nEpoch(***Best***): {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                   'acc_val:{:.2f} ±{:.2f},meta_lr: {:.6f},time: {:.2f}s,best {:.2f}'
                .format(epoch,train_loss_avg,train_acc_avg,val_acc_avg,val_acc_ci95,
                        meta_model.get_meta_learning_rate(),time.time() - t,max_val_acc))

            torch.save({'epoch': epoch, 'embedding':meta_model.state_dict(),
                        # 'optimizer': optimizer.state_dict()
                        }, os.path.join(config["save_path"], 'COIL-DEL_ASMAML_SAGE_best_model.pth'))
        else :
            print('\nEpoch: {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                    'acc_val:{:.2f} ±{:.2f},meta_lr: {:.6f},time: {:.2f}s,best {:.2f}'
                .format(epoch, train_loss_avg, train_acc_avg, val_acc_avg, val_acc_ci95,
                        meta_model.get_meta_learning_rate(), time.time() - t, max_val_acc))

        meta_model.adapt_meta_learning_rate(train_loss_avg)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))


def test(config, meta_model, test_loader) -> None:
    device=config["device"]
    t = time.time()

    val_accs=[]
        # validation_stage
    meta_model.eval()
    for i, data in enumerate(tqdm(test_loader(1)), 1):
        support_data, _, query_data, _ = data
        if config["double"]==True:
            support_data[0] = support_data[0].double()
            query_data[0] = query_data[0].double()

        # support_data = [item.to(device) for item in support_data]
        # query_data = [item.to(device) for item in query_data]

        accs,step,stop_gates,scores,query_losses= meta_model.finetuning(support_data, query_data)
        val_accs.append(accs[step])
        if i % 100 == 0:
            print("\n{}th test".format(i))
            print("stop_prob", len(stop_gates), [stop_gate for stop_gate in stop_gates])
            print("scores", len(scores), [score for score in scores])
            print("stop_prob", len(query_losses), [query_loss  for query_loss in query_losses])
            print("accs", len(accs), [accs[i] for i in range(0,step+1)])
            print("query_accs{:.2f}".format(np.mean(val_accs)))


    val_acc_avg=np.mean(val_accs)
    val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(config["val_episode"])
    print('\nacc_val:{:.2f} ±{:.2f},time: {:.2f}s'.format(val_acc_avg,val_acc_ci95,time.time() - t))

    return None


def main(test_: bool=False):
    sconfig = {
        "device"      : "cpu",
        "epochs"      : 200,
        "double"      : False,
        "save_path"   : os.path.abspath("../models"),
        "val_episode" : 200
    }

    mm_configuration = {
        "inner_lr"           : config.INNER_LR,
        "train_way"          : config.OUTER_LR,
        "train_shot"         : config.TRAIN_SHOT,
        "train_query"        : config.TRAIN_QUERY,
        "grad_clip"          : config.GRAD_CLIP,
        "batch_per_episodes" : config.BATCH_PER_EPISODES,
        "flexible_step"      : config.FLEXIBLE_STEP,
        "min_step"           : config.MIN_STEP,
        "max_step"           : config.MAX_STEP,
        "step_test"          : config.STEP_TEST,
        "step_penalty"       : config.STEP_PENALITY,
        "use_score"          : config.USE_SCORE,
        "use_loss"           : config.USE_LOSS,
        "outer_lr"           : config.OUTER_LR,
        "stop_lr"            : config.STOP_LR,
        "patience"           : config.PATIENCE,
        "weight_decay"       : config.WEIGHT_DECAY,
        "scis"               : config.STOP_CONTROL_INPUT_SIZE,
        "schs"               : config.STOP_CONTROL_HIDDEN_SIZE
    }

    model = SAGE4MAML(
        num_classes=config.TRAIN_WAY, paper=False,
        num_features=config.NUM_FEATURES["TRIANGLES"]
    )

    meta = AdaptiveStepMAML(model, False, **mm_configuration)

    logger = configure_logger(dataset_name="TRIANGLES")
    # train_ds = paper.get_dataset()
    # val_ds = paper.get_dataset(val=True)
    # train_dl = paper.get_dataloader(train_ds, 3, 10, 15, 200, 1)
    # val_dl = paper.get_dataloader(val_ds, 3, 10, 15, 200, 1)

    train_ds, val_ds, test_ds, _ = get_dataset(logger=logger, dataset_name="TRIANGLES")
    if not test_:
        train_dl = get_dataloader(train_ds, 3, 10, 15, 200, True, 1)
        val_dl = get_dataloader(val_ds, 3, 10, 15, 200, True, 1)
    
        run(sconfig, meta, train_dl, val_dl)
    else:
        meta_model = torch.load("../models/TRIANGLES_AdaptiveStepMAML_SAGE4MAML_bestModel.pth")
        meta.load_state_dict(meta_model["embedding"])
        test_dl = get_dataloader(test_ds, 3, 10, 15, 200, True, 1)
        test(sconfig, meta, test_dl)



if __name__ == "__main__":
    # Train
    # main()

    # Test
    main(True)