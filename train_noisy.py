import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import os

from models import preact_resnet18, PretrainedResNet50, ResNet34, NoiseMatrixLayer, inception_resnet_v2
from datasets.imprecise_label import get_sym_noisy_labels, get_cifar10_asym_noisy_labels, get_cifar100_asym_noisy_labels
from datasets.base_data import get_data
from datasets.base_datasets import ImgBaseDataset, ImgThreeViewDataset
from utils.configs import parse_arg_noisy
from utils.others import reproducibility, save_checkpoint, param_groups_weight_decay
from utils.metrics import AverageMeter, accuracy
from utils.validate import validate
from loss import ce_loss, WeakSpectralLoss, SupervisedNoisyLoss

args = parse_arg_noisy()
if args.wandb:
    import wandb
    wandb.init(project="wsc_noisy", config=args)
    wandb.run.name = f"{args.dataset}_noise_rate_{args.noise_ratio}_{args.time}"
    wandb.config.update(args)
    wandb.run.save()

def main(args):

    args.logger.log_args(args)
    # for reproducibility
    args.logger.info(f"Setting random seed to {args.seed}")
    reproducibility(args.seed)

    # load data
    args.logger.info(f"Loading data {args.dataset} from {args.data_path}")
    train_data, train_targets, test_data, test_targets, _ = get_data(args.data_path, args.dataset)

    # get noise labels
    args.logger.info(f"Creating noise labels with noise type {args.noise_type}")
    noise_type = args.noise_type
    if noise_type == "sym":
        assert args.dataset in ["cifar10", "cifar100"]

        _, train_data, train_noisy_targets = get_sym_noisy_labels(train_data, train_targets, args.num_classes, args.noise_ratio)
    elif noise_type == "asym":
        if args.dataset == "cifar10":
            _, train_data, train_noisy_targets = get_cifar10_asym_noisy_labels(train_data, train_targets, args.num_classes, args.noise_ratio)
        elif args.dataset == "cifar100":
            _, train_data, train_noisy_targets = get_cifar100_asym_noisy_labels(train_data, train_targets, args.num_classes, args.noise_ratio)
        else:
            raise NotImplementedError(f"Dataset {args.dataset} is not supported for asymmetric noise.")
    elif noise_type == "ins":
        if args.dataset == "cifar10n":
            noise_file = torch.load(os.path.join(args.data_path, "cifar10n", "CIFAR-10_human.pt"))
            assert args.noise_ratio in ['clean_label', 'worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3']
            train_noisy_targets = noise_file[args.noise_ratio]
        elif args.dataset == "cifar100n":
            noise_file = torch.load(os.path.join(args.data_path, "cifar10n", "CIFAR-10_human.pt"))
            assert args.noise_ratio in ['clean_label', 'noisy_label']
            train_noisy_targets = noise_file[args.noise_ratio]
        else:
            # noisy labels is directly loaded in train_targets
            train_noisy_targets = train_targets
    else:
        raise NotImplementedError(f"Noise type {noise_type} is not supported.")
    
    # create dataset
    args.logger.info(f"Creating dataset {args.dataset}")
    train_dataset = ImgThreeViewDataset(args, train_data, train_noisy_targets, 'noise')
    test_dataset = ImgBaseDataset(args, test_data, test_targets, "noise")

    # create dataloader
    args.logger.info(f"Creating dataloader with batch size {args.batch_size}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    args.logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    # model
    if args.model == "preact_resnet18":
        model = preact_resnet18(num_classes=args.num_classes)
    elif args.model == "resnet50_pretrained":
        model = PretrainedResNet50(num_classes=args.num_classes)
    elif args.model == "resnet34":
        model = ResNet34(num_classes=args.num_classes)
    elif args.model == "inception_resnet_v2":
        model = inception_resnet_v2(num_classes=args.num_classes)
    else:
        raise NotImplementedError(f"Model {args.model} is not supported.")
    
    model = model.cuda()

    args.logger.info(f"Created model {args.model}")

    # our noise method should use noise matrix
    args.logger.info(f"Creating noise matrix with scale {args.noise_matrix_scale}") 
    noise_model = NoiseMatrixLayer(args.num_classes, init=args.noise_matrix_scale)


    def create_projector(in_dim, out_dim):
        if args.model in ["preact_resnet18", "inception_resnet_v2"]:
            squential = nn.Sequential(nn.Linear(in_dim, in_dim),
                            nn.BatchNorm1d(in_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_dim, out_dim),
                            nn.BatchNorm1d(out_dim),
                            )
        else:
            squential = nn.Identity()
        return squential
    
    # projector
    if args.model in ["preact_resnet18", "resnet50_pretrained", "resnet34"]:
        projector = create_projector(512, 256).cuda()
    elif args.model == "inception_resnet_v2":
        projector = create_projector(1536, 256).cuda()

    # optimizer
    args.logger.info(f"Creating optimizer and scheduler")
    per_param_args = param_groups_weight_decay(model, args.weight_decay, no_weight_decay_list={})
    optimizer = torch.optim.SGD(
        per_param_args,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    noise_matrix_optimizer = torch.optim.SGD(
        noise_model.parameters(),
        lr=args.lr,
        momentum=0,
        weight_decay=0
    )

    # scheduler
    if args.dataset == "webvision":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50],
            last_epoch=-1,
        )
    elif args.dataset == "clothing1m":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[7],
            last_epoch=-1,
        )
    else:
        # for cub200, use cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min = 2e-4,
        )

    # loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()
    sup_loss = SupervisedNoisyLoss(args.num_classes)
    weak_spec_loss = WeakSpectralLoss(args.alpha, args.beta, args)

    # resume from checkpoint
    args.logger.info(f"Check if resume from checkpoint")
    if args.resume:
        if os.path.isfile(args.resume):
            args.logger.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            noise_model.load_state_dict(checkpoint['noise_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            noise_matrix_optimizer.load_state_dict(checkpoint['noise_matrix_optimizer'])
        else:
            args.logger.info(f"No checkpoint found at '{args.resume}'")
    
    # training
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        model.train()
        noise_model.train()

        train_loss = train(args, train_loader, model, noise_model, projector, optimizer, noise_matrix_optimizer, sup_loss, weak_spec_loss, epoch)
        print(f"noise matrix:\n {noise_model()}")
        scheduler.step()

        val_loss, val_acc = validate(test_loader, model, criterion, args)

        if val_acc > best_acc:
            best_acc = val_acc
            is_best = True

        args.logger.info(f"Epoch {epoch+1}: LR: {optimizer.param_groups[0]['lr']:.4f} Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} Best Acc: {best_acc:.4f}")

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'noise_model': noise_model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'noise_matrix_optimizer': noise_matrix_optimizer.state_dict(),
        }, is_best, args)


def train(args, train_loader, model, noise_model, projector, optimizer, noise_matrix_optimizer, sup_loss, weak_spec_loss, epoch):
    losses = AverageMeter()
    sup_losses = AverageMeter()
    vol_losses = AverageMeter()
    wsc_losses = AverageMeter()
    consist_losses = AverageMeter()
    l1_losses = AverageMeter()
    l2_losses = AverageMeter()

    progress_bar = tqdm(enumerate(train_loader) ,total=len(train_loader), desc=f"Train Epoch: [{epoch + 1}]")

    for i, (x_w, x_s, x_s_, noise_y, index) in progress_bar:
        x_w = x_w.cuda()
        x_s = x_s.cuda()
        x_s_ = x_s_.cuda()
        noise_y = noise_y.cuda()

        x_aug = torch.cat([x_w, x_s, x_s_], dim=0)
        y_pred, feat = model(x_aug)

        feat = projector(feat)
        if args.model in ["preact_resnet18", "inception_resnet_v2"]:
            feat = F.normalize(feat, dim=1)

        y_pred_w, y_pred_s, y_pred_s_ = y_pred.chunk(3)
        feat_w, feat_s, feat_s_ = feat.chunk(3)

        noise_matrix = noise_model()

        probs_x_w = y_pred_w.softmax(dim=-1).detach()

        noisy_probs_x_w = torch.matmul(y_pred_w.softmax(dim=-1), noise_matrix)
        noisy_probs_x_w = noisy_probs_x_w / noisy_probs_x_w.sum(dim=-1, keepdim=True)

        # supervised loss
        supervised_loss = torch.mean(-torch.sum(F.one_hot(noise_y, args.num_classes) * torch.log(noisy_probs_x_w), dim = -1))

        # VolMinNet loss
        vol_loss = noise_matrix.slogdet().logabsdet

        # consistency loss
        con_loss = ce_loss(y_pred_s, probs_x_w, reduction='mean')
        con_loss_ = ce_loss(y_pred_s_, probs_x_w, reduction='mean')

        # wsc loss
        """
        Note that there's not one way to construct the wsc loss. you will also notice that there's a function called create_noise_matrix_inv in models.utils, which provide another way to do it. Our paper also mentioned that case and the experiments for this way (yes it actually using the enviroment information) is under exploration actually. An example is here:
            >>> true_noisy_matrix_inv = create_noise_matrix_inv(args.num_classes, args.noise_ratio).detach().cuda()
            >>> construced_s = (F.one_hot(y, args.num_classes).float() @ true_noisy_matrix_inv).detach()
            >>> wsc_loss, l1, l2 = weak_spec_loss(feat_s, feat_s_, constructed_s)
            For our implementation, we just use the probabilities of the model output as the constructed s.
        """
        wsc_loss, l1, l2 = weak_spec_loss(feat_s, feat_s_, probs_x_w)

        # total loss
        lam = min(1, float(epoch + 1)/args.epochs) * args.lam
        loss = supervised_loss + con_loss + con_loss_ + args.vol_lambda * vol_loss + lam * wsc_loss

        # compute average entropy loss
        if args.average_entropy_loss:
            avg_prediction = torch.mean(y_pred_w.softmax(dim=-1), dim=0)
            prior_distr = 1.0 / args.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min=1e-6, max=1.0)
            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = args.balance_lam * balance_kl
            loss += entropy_loss

        # update parameters
        loss.backward()

        optimizer.zero_grad()
        noise_matrix_optimizer.zero_grad()

        optimizer.step()
        noise_matrix_optimizer.step()


        # update meter
        losses.update(loss.item(), x_w.size(0))
        sup_losses.update((supervised_loss).item(), x_w.size(0))
        vol_losses.update(args.vol_lambda * vol_loss.item(), x_w.size(0))
        wsc_losses.update(wsc_loss.item(), x_w.size(0))
        consist_losses.update((con_loss + con_loss_).item(), x_w.size(0))
        l1_losses.update(l1.item(), x_w.size(0))
        l2_losses.update(l2.item(), x_w.size(0))

        progress_bar.set_postfix({
                "loss": f"{losses.val:.4f}",
                "sup": f"{sup_losses.val:.4f}",
                "vol": f"{vol_losses.val:.4f}",
                "wsc": f"{wsc_losses.val:.4f}",
                "con": f"{consist_losses.val:.4f}",
            })

        if args.wandb:
            wandb.log({
                    "loss": losses.val,
                    "sup_loss": sup_losses.val,
                    "wsc_loss": wsc_losses.val,
                    "consist_loss": consist_losses.val,
                    "l1_loss": l1_losses.val,
                    "l2_loss": l2_losses.val
                })
            
    return losses.avg

if __name__ == "__main__":
    main(args)