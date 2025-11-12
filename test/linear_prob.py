import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
from typing import Union

import torch.utils
import torch.utils.data
from tqdm import tqdm

from utils.logger import Logger
from datasets.imprecise_label import get_sym_noisy_labels
from datasets.base_data import get_data
from models.preact_resnet import preact_resnet18
from datasets.base_datasets import ImgBaseDataset
from utils.metrics import AverageMeter



def parse_args():
    parser = argparse.ArgumentParser(description="Linear probing test for WSC")

    # dataset
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "cifar10n", "cifar100n", "clothing1m"], help="dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="path to dataset")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--crop_ratio", type=float, default=0.875, help="crop ratio")
    parser.add_argument("--img_size", type=int, default=32, help="image size")
    parser.add_argument("--noise_ratio", type=float, default=0.9, help="noise ratio")
    parser.add_argument("--noise_type", type=str, default="sym", choices=["sym", "asym", "ins"], help="noise type")

    # training settings
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")
    parser.add_argument("--model_path", default=None, type=str, help="path to latest checkpoint (default: none)")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--method", type=str, default="wsc", choices=["wsc", "elr+"], help="which method to test")


    args = parser.parse_args()

    now = datetime.now()
    args.out = f"./results/linear_prob/{args.dataset}_{args.noise_ratio}_{args.method}/{now.strftime('%m%d_%H%M')}/"
    args.logger = Logger(args.out)

    return args

def reproducibility(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
    return seed

def main(args):
    # dataset
    train_data, train_targets, test_data, test_targets, _ = get_data(args.data_path, args.dataset)

    args.logger.info(f"Creating noise labels with noise ratio {args.noise_ratio}")
    assert args.dataset in ["cifar10", "cifar100"]
    _, train_data, train_noisy_targets = get_sym_noisy_labels(train_data, train_targets, args.num_classes, args.noise_ratio)

    train_dataset = ImgBaseDataset(args, train_data, train_noisy_targets, "noise")
    test_dataset = ImgBaseDataset(args, test_data, test_targets, "noise")
    # train_dataset = ImgBaseDataset(args, train_data, train_targets, "noise")
    # test_dataset = ImgBaseDataset(args, test_data, test_targets, "noise")

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # load model
    args.logger.info(f"Loading model preact_resnet18 for linear probing")
    model = preact_resnet18(num_classes=args.num_classes)
    model.cuda()

    checkpoint = torch.load(args.model_path)
    if args.method == "wsc":
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif args.method == "elr+":
        model.load_state_dict(checkpoint['state_dict1'], strict=False)

    encoder = model.encoder
    for params in encoder.parameters():
        params.requires_grad = False

    linear_layer = nn.Linear(model.num_features, args.num_classes).cuda()

    # optimizer and scheduler
    args.logger.info("Setting up optimizer and scheduler")
    optimizer = torch.optim.SGD(linear_layer.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    criterion = nn.CrossEntropyLoss().cuda()
    # use l2 loss
    # criterion = nn.MSELoss().cuda()

    is_best = False
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        # training
        linear_layer.train()

        train_loss = AverageMeter()
        
        progress_bar = tqdm(enumerate(train_loader) ,total=len(train_loader), desc=f"Train Epoch: [{epoch + 1}]")

        for i, (inputs, targets, index) in progress_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            with torch.no_grad():
                features = encoder(inputs)

            features = F.avg_pool2d(features, 4)
            features = features.view(features.size(0), -1)
            outputs = linear_layer(features)
            # outputs_prob = F.softmax(outputs, dim=1)
            # targets_one_hot = F.one_hot(targets, num_classes=args.num_classes).float()
            loss = criterion(outputs, targets)
            # loss = criterion(outputs_prob, targets_one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), inputs.size(0))

            progress_bar.set_postfix({"loss": train_loss.avg})

        scheduler.step()
        args.logger.info(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {train_loss.avg:.4f}")

        # validation
        linear_layer.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Val Epoch: [{epoch + 1}]")

        with torch.no_grad():
            for i, (inputs, targets, index) in progress_bar:
                inputs, targets = inputs.cuda(), targets.cuda()

                features = encoder(inputs)
                features = F.avg_pool2d(features, 4)
                features = features.view(features.size(0), -1)
                outputs = linear_layer(features)

                loss = nn.CrossEntropyLoss()(outputs, targets)

                acc = (outputs.argmax(dim=1) == targets).float().mean().item()

                val_loss.update(loss.item(), inputs.size(0))
                val_acc.update(acc, inputs.size(0))

                progress_bar.set_postfix({"loss": val_loss.avg, "acc": val_acc.avg})

        args.logger.info(f"Validation - Loss: {val_loss.avg:.4f}, Acc: {val_acc.avg:.4f}")
        if val_acc.avg > best_acc:
            best_acc = val_acc.avg
            best_epoch = epoch + 1
            is_best = True
        else:
            is_best = False
        
    args.logger.info(f"Best Accuracy: {best_acc:.4f} at epoch {best_epoch}")
    args.logger.info("Training complete.")


if __name__ == "__main__":
    reproducibility(42)
    
    args = parse_args()
    main(args)
    
