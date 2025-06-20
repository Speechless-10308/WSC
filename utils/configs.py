import argparse
from datetime import datetime
from typing import Union

from utils.logger import Logger

def parse_arg_partial():
    parser = argparse.ArgumentParser(description="Weakly supervised contrastive learning for partial labels")


    # dataset
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "cub200"], help="dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="path to dataset")
    parser.add_argument("--partial_rate", type=float, default=0.2, help="partial label rate")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--img_size", type=int, default=32, help="image size")
    parser.add_argument("--partial_type", type=str, default="uniform", choices=["uniform", "hierarchical"], help="partial type")

    # model settings
    parser.add_argument("--model", type=str, default="widenet", choices=["widenet", "resnet18"], help="model name")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--lam", type=float, default=1.0, help="lambda for consistency loss and wsc loss")
    parser.add_argument("--average_entropy_loss", action='store_true', default=False, help="use average entropy loss")
    

    # wsc specific
    parser.add_argument("--alpha", type=float, default=2.0, help="alpha for wsc loss")
    parser.add_argument("--beta", type=float, default=300.0, help="beta for wsc loss")
    parser.add_argument("--lam_consist", type=float, default=0.1, help="lambda for consistency loss")

    # logging
    parser.add_argument("--out", type=str, default="./results", help="output directory")
    parser.add_argument("--trial", type=str, default="1", help="trial number")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--wandb", action="store_true", help="use wandb for logging")

    # resume
    parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: none)")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch (default: 0)")

    # configs
    parser.add_argument("--config_file", type=str, default=None, help="path to config file (default: None)")

    
    args = parser.parse_args()

    if args.config_file:
        import yaml
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    args.out = f"{args.out}/{args.dataset}/p_rate_{args.partial_rate}/"
    now = datetime.now()
    args.time = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"

    return args


def parse_arg_noisy():
    parser = argparse.ArgumentParser(description="Weakly supervised contrastive learning for noisy labels")

    # dataset
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "cifar10n", "cifar100n", "clothing1m"], help="dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="path to dataset")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--crop_ratio", type=float, default=0.875, help="crop ratio")
    parser.add_argument("--img_size", type=int, default=32, help="image size")
    parser.add_argument("--noise_ratio", type=Union[str, float], default=0.2, help="noise ratio")
    parser.add_argument("--noise_type", type=str, default="sym", choices=["sym", "asym", "ins"], help="noise type")

    # model settings
    parser.add_argument("--model", type=str, default="widenet", choices=["preact_resnet18", "resnet50_pretrained", "resnet34"], help="model name")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--average_entropy_loss", action='store_true', default=False, help="use average entropy loss")
    parser.add_argument("--noise_matrix_scale", type=float, default=1.0, help="scale for noise matrix, should be big when number of class is large")
    parser.add_argument("--vol_lambda", type=float, default=0.1, help="lambda for VolMinNet loss")
    parser.add_argument("--mix_alpha", type=float, default=0.2, help="mixup alpha for Mixup loss")

    # wsc specific
    parser.add_argument("--alpha", type=float, default=2.0, help="alpha for wsc loss")
    parser.add_argument("--beta", type=float, default=300.0, help="beta for wsc loss")
    parser.add_argument("--lam", type=float, default=0.1, help="lambda for wsc loss")
    parser.add_argument("--lam_consist", type=float, default=0.1, help="lambda for consistency loss in wsc loss")

    # logging
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--wandb", action="store_true", help="use wandb for logging")

    # resume
    parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: none)")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch (default: 0)")

    # configs
    parser.add_argument("--config_file", type=str, default=None, help="path to config file (default: None)")
    parser.add_argument("--notes", type=str, default="", help="notes for the experiment")

    # multi gpu
    parser.add_argument("--distributed", action='store_true', default=False, help="use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="number of processes in distributed training")
    parser.add_argument("--dist_url", type=str, default="tcp://localhost:12355", help="url used to set up distributed training")

    
    args = parser.parse_args()

    if args.config_file:
        import yaml
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    now = datetime.now()
    args.out = f"./results/{args.dataset}/{args.model}_{args.noise_ratio}_{args.notes}/{now.strftime('%m%d_%H%M')}/"
    
    args.time = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"

    # args.logger = Logger(args.out)

    return args
