import sys
import os

sys.path.append(os.getcwd())

from data.dataset import get_dataset
from utils.utils import configure_logger
from utils.trainer import Trainer
from algorithms.asmaml.asmaml import AdaptiveStepMAML

import argparse
import config


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General arguments for the script
    parser.add_argument('-p', '--path',      help="The path of the dataset",                   type=str,  default=config.DATA_PATH)
    parser.add_argument('-n', '--name',      help="The name of the dataset",                   type=str,  default=config.DEFAULT_DATASET)
    parser.add_argument('-d', '--device',    help="The device to use",                         type=str,  default=config.DEVICE)
    parser.add_argument('-l', '--log-path',  help="The path where to log",                     type=str,  default=config.LOGGING_PATH)
    parser.add_argument('-f', '--file-log',  help="If logging to file or not",                            default=config.FILE_LOGGING, action="store_true")
    parser.add_argument('-s', '--save-path', help="The path where to save pre-trained models", type=str,  default=config.MODELS_SAVE_PATH)
    parser.add_argument('-m', '--model',     help="The name of the model (sage or gcn)",       type=str,  default=config.MODEL_NAME)
    parser.add_argument('--not-as-maml',     help="Use AS-MAML or not",                                   default=True, action="store_false")
    parser.add_argument('--gmixup',          help="Use G-Mixup or not",                                   default=False, action="store_true")
    parser.add_argument('--flag',            help="Use FLAG or not",                                      default=False, action="store_true")
    parser.add_argument('--mevolve',         help="Use M-Evolve or not",                                  default=False, action="store_true")

    # Configurations for the AS-MAML Model
    parser.add_argument('--batch-size',    help="Dimension of a batch",                            type=int,   default=1)
    parser.add_argument('--outer_lr',      help="Initial LR for the model",                        type=float, default=config.OUTER_LR)
    parser.add_argument('--inner_lr',      help="Initial LR for the meta model",                   type=float, default=config.INNER_LR)
    parser.add_argument('--stop_lr',       help="Initial LR for the Stop model",                   type=float, default=config.STOP_LR)
    parser.add_argument('--w-decay',       help="The Weight Decay for optimizer",                  type=float, default=config.WEIGHT_DECAY)
    parser.add_argument('--max-step',      help="The Max Step of the meta model",                  type=int,   default=config.MAX_STEP)
    parser.add_argument('--min-step',      help="The Min Step of the meta model",                  type=int,   default=config.MIN_STEP)
    parser.add_argument('--penalty',       help="Step Penality for the RL model",                  type=float, default=config.STEP_PENALITY)
    parser.add_argument('--train-shot',    help="The number of Shot per Training",                 type=int,   default=config.TRAIN_SHOT)
    parser.add_argument('--val-shot',      help="The number of shot per Validation",               type=int,   default=config.VAL_SHOT)
    parser.add_argument('--train-query',   help="The number of query per Training",                type=int,   default=config.TRAIN_QUERY)
    parser.add_argument('--val-query',     help="The number of query per Validation",              type=int,   default=config.VAL_QUERY)
    parser.add_argument('--train-way',     help="The number of way for Training",                  type=int,   default=config.TRAIN_WAY)
    parser.add_argument('--test-way',      help="The number of way for Test and Val",              type=int,   default=config.TEST_WAY)
    parser.add_argument('--val-episode',   help="The number of episode for Val",                   type=int,   default=config.VAL_EPISODE)
    parser.add_argument('--train-episode', help="The number of episode for Training",              type=int,   default=config.TRAIN_EPISODE)
    parser.add_argument('--batch-episode', help="The number of batch per episode",                 type=int,   default=config.BATCH_PER_EPISODES)
    parser.add_argument('--epochs',        help="The total number of epochs",                      type=int,   default=config.EPOCHS)
    parser.add_argument('--patience',      help="The patience",                                    type=int,   default=config.PATIENCE)
    parser.add_argument('--grad-clip',     help="The clipping for the gradient",                   type=int,   default=config.GRAD_CLIP)
    parser.add_argument('--scis',          help="The input dimension for the Stop Control model",  type=int,   default=config.STOP_CONTROL_INPUT_SIZE)
    parser.add_argument('--schs',          help="The hidden dimension for the Stop Control model", type=int,   default=config.STOP_CONTROL_HIDDEN_SIZE)

    # M-Evolve configurations
    parser.add_argument('--beta',      help="The beta used in heuristics of M-Evolve",      type=float, default=config.BETA)
    parser.add_argument('--n-fold',    help="The number of Fold for the nX-fol-validation", type=int,   default=config.N_FOLD)
    parser.add_argument('--n-xval',    help="Number of Cross-fold Validation to run",       type=int,   default=config.N_CROSSVALIDATION)
    parser.add_argument('--iters',     help="Number of iterations of M-Evolve",             type=int,   default=config.ITERATIONS)
    parser.add_argument('--heuristic', help="The Heuristic to use",                         type=str,   default=config.HEURISTIC)
    parser.add_argument('--lrts',      help="The label reliability step thresholds",        type=int,   default=config.LABEL_REL_THRESHOLD_STEPS)
    parser.add_argument('--lrtb',      help="The beta used for approximation of the tanh",  type=int,   default=config.LABEL_REL_THRESHOLD_BETA)

    # FLAG configurations
    parser.add_argument('--flag-m', help="The number of iterations of FLAG", type=int,   default=config.M)
    parser.add_argument('--ass',    help="The attack step size",             type=float, default=config.ATTACK_STEP_SIZE)


    args = parser.parse_args()
    assert args.model in ["sage", "gcn"], f"Model name: {args.model} has not been implemented yet"
    assert sum([args.mevolve, args.flag, args.gmixup]) < 2, "Cannot use more than one GDA technique at the same time"

    configs = {
        "data_path"     : args.path,
        "data_name"     : args.name,
        "device"        : args.device,
        "log_path"      : args.log_path,
        "file_log"      : args.file_log,
        "save_path"     : args.save_path,
        "model_name"    : args.model,
        "use_asmaml"    : args.not_as_maml,
        "use_gmixup"    : args.gmixup,
        "use_flag"      : args.flag,
        "use_mevolve"   : args.mevolve,
        "batch_size"    : args.batch_size,
        "outer_lr"      : args.outer_lr,
        "inner_lr"      : args.inner_lr,
        "stop_lr"       : args.stop_lr,
        "weight_decay"  : args.w_decay,
        "max_step"      : args.max_step,
        "min_step"      : args.min_step,
        "penalty"       : args.penalty,
        "train_shot"    : args.train_shot,
        "val_shot"      : args.val_shot,
        "train_query"   : args.train_query,
        "val_query"     : args.val_query,
        "train_way"     : args.train_way,
        "test_way"      : args.test_way,
        "val_episode"   : args.val_episode,
        "train_episode" : args.train_episode,
        "batch_episode" : args.batch_episode,
        "epochs"        : args.epochs,
        "patience"      : args.patience,
        "grad_clip"     : args.grad_clip,
        "scis"          : args.scis,
        "schs"          : args.schs,
        "beta"          : args.beta,
        "n_fold"        : args.n_fold,
        "n_xval"        : args.n_xval,
        "iters"         : args.iters,
        "heuristic"     : args.heuristic,
        "lrts"          : args.lrts,
        "lrtb"          : args.lrtb,
        "flag_m"        : args.flag_m,
        "ass"           : args.ass
    }


    logger = configure_logger(file_logging=configs["file_log"], logging_path=configs["log_path"])
    dataset_name = configs["data_name"]
    train_ds, test_ds, val_ds, _ = get_dataset(
        download=False, 
        data_dir=configs["data_path"], 
        logger=logger, 
        dataset_name=dataset_name
    )

    logger.debug("--- Datasets ---")
    print("\n- Train: ", train_ds, file=sys.stdout if not configs["file_log"] else open(logger.handlers[1].baseFilename, mode="a"))
    print("- Test : ", test_ds, file=sys.stdout if not configs["file_log"] else open(logger.handlers[1].baseFilename, mode="a"))
    print("- Validation: ", val_ds, file=sys.stdout if not configs["file_log"] else open(logger.handlers[1].baseFilename, mode="a"))
    print("\n", file=sys.stdout if not configs["file_log"] else open(logger.handlers[1].baseFilename, mode="a"))

    logger.debug("--- Configuration ---")
    configurations = ("\nDEVICE: {device}\n"                             +
                      "DATASET NAME: {dataset_name}\n"                   + 
                      "USE ASMAML: {use_asmaml}\n"                       +
                      "USE_GMIXUP : {use_gmixup}\n"                      +
                      "USE_MEVOLVE : {use_mevolve}\n"                    + 
                      "USE_FLAG : {use_flag}\n"                          + 
                      "TRAIN SUPPORT SIZE: {train_support_size}\n"       +
                      "TRAIN QUERY SIZE: {train_query_size}\n"           +
                      "VALIDATION SUPPORT SIZE: {val_support_size}\n"    +
                      "VALIDATION QUERY SIZE: {val_query_size}\n"        +
                      "TEST SUPPORT SIZE: {test_support_size}\n"         +
                      "TEST QUERY SIZE: {test_query_size}\n"             +
                      "TRAIN EPISODE: {train_episode}\n"                 +
                      "VALIDATION EPISODE: {val_episode}\n"              +
                      "NUMBER OF EPOCHS: {number_of_epochs}\n"           +
                      "BATCH PER EPISODES: {batch_per_episodes}\n"       + 
                      "NUMBER OF FOLDS : {n_folds}\n"                    +
                      "NUMBER OF CROSS-FOLD VALIDATION RUNS: {n_xval}\n" +
                      "M-EVOLVE ITERATIONS : {m_iters}\n"                + 
                      "M-EVOLVE HEURISTIC : {m_heu}\n"                   +
                      "FLAG ITERATIONS : {flag_m}\n"                     
        ).format(
            device=configs["device"], dataset_name=dataset_name,
            use_asmaml=configs["use_asmaml"], use_gmixup=configs["use_gmixup"], 
            use_flag=configs["use_flag"], use_mevolve=configs["use_mevolve"],
            train_support_size="{} x {}".format(configs["train_way"], configs["train_shot"]),
            train_query_size="{} x {}".format(configs["train_way"], configs["train_query"]),
            val_support_size="{} x {}".format(configs["test_way"], configs["val_shot"]),
            val_query_size="{} x {}".format(configs["test_way"], configs["val_query"]),
            test_support_size="{} x {}".format(configs["test_way"], configs["val_shot"]),
            test_query_size="{} x {}".format(configs["test_way"], configs["val_query"]),
            train_episode=configs["train_episode"], val_episode=configs["val_episode"],
            number_of_epochs=configs["epochs"], batch_per_episodes=configs["batch_episode"],
            n_folds=configs["n_fold"] if configs["use_mevolve"] else "M-Evolve not used",
            n_xval=configs["n_xval"] if configs["use_mevolve"] else "M-Evolve not used",
            m_iters=configs["iters"] if configs["use_mevolve"] else "M-Evolve not used",
            m_heu=configs["heuristic"] if configs["use_mevolve"] else "M-Evolve not used",
            flag_m=configs["falg_m"] if configs["use_flag"] else "FLAG not used"
    )

    print(configurations, file=sys.stdout if not configs["file_log"] else open(logger.handlers[1].baseFilename, mode="a"))

    # Run the trainer
    meta_model = AdaptiveStepMAML if configs["use_asmaml"] else None
    save_suffix = "ASMAML_" if configs["use_asmaml"] else "_"
    optimizer = Trainer(
        train_ds=train_ds, val_ds=val_ds, logger=logger, 
        meta_model=meta_model, save_suffix=save_suffix,
        **configs
    )

    optimizer.run()


if __name__ == '__main__':
	main()