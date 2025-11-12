from sklearn.metrics import accuracy_score
from sympy import false
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import sys, os

from tqdm import tqdm
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.base_data import get_data
from models.preact_resnet import preact_resnet18
from datasets.base_datasets import ImgBaseDataset


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

def parse_args():
    parser = argparse.ArgumentParser(description="t-sne visualization for WSC")

    # dataset
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "cifar10n", "cifar100n", "clothing1m"], help="dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="path to dataset")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--crop_ratio", type=float, default=0.875, help="crop ratio")
    parser.add_argument("--img_size", type=int, default=32, help="image size")
    parser.add_argument("--noise_ratio", type=float, default=0.9, help="noise ratio")
    parser.add_argument("--noise_type", type=str, default="sym", choices=["sym", "asym", "ins"], help="noise type")

    # model settings
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")
    parser.add_argument("--model_path", default=None, type=str, help="path to latest checkpoint (default: none)")
    parser.add_argument("--method", type=str, default="wsc", choices=["wsc", "elr+"], help="method for visualization")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")

    # palette setting
    parser.add_argument("--palette", type=str, default="tab10")


    args = parser.parse_args()

    return args

def main(args):
    # dataset
    train_data, train_targets, test_data, test_targets, _ = get_data(args.data_path, args.dataset)

    test_dataset = ImgBaseDataset(args, test_data, test_targets, "noise")

    # dataloader
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # load model
    print(f"Loading model preact_resnet18 for t-sne visualization")
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

    encoder.eval()
    model.eval()

    test_features = []
    test_labels = []

    if args.dataset == "cifar10":
        with torch.no_grad():
            for data, target, index in tqdm(test_loader, desc="Extracting Test Features"):
                data = data.cuda()
                features = encoder(data)
                features = F.avg_pool2d(features, 4)
                features = features.view(features.size(0), -1)
                test_features.append(features.cpu().numpy())
                test_labels.append(target.cpu().numpy())

        test_features = np.concatenate(test_features, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
    elif args.dataset == "cifar100":
        # get the acc first
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, target, index in tqdm(test_loader, desc="Get Accuracy"):
                data = data.cuda()
                logits = model(data)

                pred = logits.argmax(dim=1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(target.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        class_accuracies = {}
        for i in range(100):
            class_mask = (all_labels == i)
            if np.sum(class_mask) > 0:
                class_preds = all_preds[class_mask]
                class_true_labels = all_labels[class_mask]
                accuracy = accuracy_score(class_true_labels, class_preds)
                class_accuracies[i] = accuracy
            else:
                class_accuracies[i] = 0.0

        sorted_classes = sorted(class_accuracies.items(), key=lambda item: item[1], reverse=True)
        top_classes = sorted_classes[:20]
        top_class_indices = [c[0] for c in top_classes]

        with torch.no_grad():
            for data, target, index in tqdm(test_loader, desc="Extracting Test Features"):
                data = data.cuda()
                features = encoder(data)
                features = F.avg_pool2d(features, 4)
                features = features.view(features.size(0), -1)
                test_features.append(features.cpu().numpy())
                test_labels.append(target.cpu().numpy())

        test_features = np.concatenate(test_features, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        mask = np.isin(test_labels, top_class_indices)
        filtered_features = test_features[mask]
        filtered_labels = test_labels[mask]

        print(f"Original number of samples: {len(test_labels)}")
        print(f"Number of samples after filtering for top 20 classes: {len(filtered_labels)}")

        test_features = filtered_features
        test_labels = filtered_labels

    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, random_state=42)
    test_tsne_features = tsne.fit_transform(test_features)

    plt.figure(figsize=(4, 4))
    scatter = sns.scatterplot(
        x=test_tsne_features[:, 0],
        y=test_tsne_features[:, 1],
        hue=test_labels,
        palette=sns.color_palette(args.palette, n_colors=len(np.unique(test_labels))),
        legend=False,
        alpha=0.7,
        s=2,
        linewidth=0,
    )
    plt.title(f"{args.method.upper()} on {args.dataset.upper()}", fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

    plt.savefig(f"tsne_test_{args.method}_{args.dataset}.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

    reproducibility(42)
    args = parse_args()
    main(args)