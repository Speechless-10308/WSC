from ast import Not
import copy
import os
from re import L
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from models import ResNet18, NoiseMatrixLayer, resnet18con
from datasets.imprecise_label import get_partial_noisy_labels, get_partial_labels, get_hierarchical_partial_labels
from datasets.base_data import get_data
from datasets.base_datasets import ImgBaseDataset, ImgThreeViewDataset
from utils.configs import parse_arg_mixture
from utils.logger import Logger
from utils.others import reproducibility, save_checkpoint, param_groups_weight_decay
from utils.metrics import AverageMeter, accuracy
from utils.validate import validate

from loss import ce_loss, WeakSpectralLoss

args = parse_arg_mixture()
reproducibility(args.seed)
if args.wandb:
    import wandb
    wandb.init(project="WSC-Mixture", config=args)
    wandb.run.name = f"{args.dataset}_{args.model}_{args.noise_ratio}_{args.partial_rate}_{args.time}"
    wandb.config.update(args)
    wandb.run.save()

def main(args):
    args.logger.log_args(args)

    # load data
    args.logger.info(f"Loading data {args.dataset} from {args.data_path}")
    train_data, train_targets, test_data, test_targets, _ = get_data(args.data_path, args.dataset)

    # make partial labels
    args.logger.info(f"Creating noise partial labels with partial type {args.partial_type}")
    partial_type = args.partial_type
    if partial_type == 'uniform':
        train_data, train_partial_labels = get_partial_labels(train_data, train_targets, args.num_classes, args.partial_rate)
    elif partial_type == "hierarchical":
        if args.dataset == "cifar100":
            train_data, train_partial_labels = get_hierarchical_partial_labels(args, train_data, train_targets, args.num_classes, args.partial_rate)
        else:
            raise NotImplementedError(f"Dataset {args.dataset} is not supported for hierarchical partial.")
    else:
        raise NotImplementedError(f"Partial type {partial_type} is not implemented.")
    # make noisy labels
    train_partial_noisy_targets = get_partial_noisy_labels(train_targets, train_partial_labels, args.noise_ratio)

    # create dataset
    args.logger.info(f"Creating dataset {args.dataset}")
    train_dataset = ImgThreeViewDataset(args, train_data, train_partial_noisy_targets, 'noisy_partial')
    test_dataset = ImgBaseDataset(args, test_data, test_targets, 'noisy_partial')

    # create dataloader
    args.logger.info(f"Creating dataloader {args.dataset}")
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

    if args.model == "resnet18":
        if args.dataset in ["cifar10", "cifar100"]:
            model = ResNet18(num_classes=args.num_classes)
        elif args.dataset == "cub200":
            model = resnet18con()
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented")
    model = model.cuda()

    noise_model = NoiseMatrixLayer(args.num_classes, init=args.noise_matrix_scale)

    per_param_args = param_groups_weight_decay(model, args.weight_decay, no_weight_decay_list={})
    args.logger.info("Creating optimizer and scheduler")
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

    # if args.dataset in ['cifar10', 'cifar100']:
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer,
    #         T_max=int(args.epochs * len(train_loader)), 
    #         eta_min=1e-4
    #     )
    # else:
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer,
    #         milestones=[int(args.epochs * 0.5 * len(train_loader)), int(args.epochs * 0.75 * len(train_loader))],
    #         gamma=0.1
    #     )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs * len(train_loader)), 
        eta_min=args.lr * 0.001
    )

    confidence = copy.deepcopy(train_loader.dataset.targets)
    confidence = confidence / confidence.sum(axis=1)[:, None]

    confidence2 = copy.deepcopy(train_loader.dataset.targets)
    confidence2 = confidence2 / confidence2.sum(axis=1)[:, None]

    criterion = torch.nn.CrossEntropyLoss().cuda()
    consistency_criterion = torch.nn.KLDivLoss(reduction='batchmean').cuda()
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

    best_acc = 0
    ema_p = 0.999
    p_hat = torch.ones((args.num_classes, )) / args.num_classes
    p_hat = p_hat.cuda()

    for epoch in range(args.start_epoch, args.epochs):
        is_best = False

        model.train()
        noise_model.train()

        try:
            if args.beta_up:
                weak_spec_loss.beta = args.beta * ((epoch - args.wp) / (args.epochs - args.wp))
        except:
            pass


        loss_dict, confidence, confidence2 = train(args, train_loader, model, noise_model, optimizer, noise_matrix_optimizer, consistency_criterion, weak_spec_loss, scheduler, confidence, confidence2, epoch)

        args.logger.info(f"noise matrix:\n {noise_model()}")

        val_loss, val_acc = validate(test_loader, model, criterion, args)

        if val_acc > best_acc:
            best_acc = val_acc
            is_best = True

        args.logger.info(f"Epoch {epoch+1}: LR: {optimizer.param_groups[0]['lr']:.4f} Train Loss: {dict2str(loss_dict)}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} Best Acc: {best_acc:.4f}")

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


def train(args, train_loader, model, noise_model, optimizer, noise_matrix_optimizer, consistency_criterion, weak_spec_loss, scheduler, confidence, confidence2, epoch):
    losses = AverageMeter()
    sup_losses = AverageMeter()
    vol_losses = AverageMeter()
    wsc_losses = AverageMeter()
    consist_losses = AverageMeter()
    l1_losses = AverageMeter()
    l2_losses = AverageMeter()

    progress_bar = tqdm(enumerate(train_loader) ,total=len(train_loader), desc=f"Train Epoch: [{epoch + 1}]")

    for i, (x_w, x_s, x_s_, partial_noise_y, index) in progress_bar:
        x_w = x_w.cuda()
        x_s = x_s.cuda()
        x_s_ = x_s_.cuda()
        partial_noise_y = partial_noise_y.cuda()

        # mixup trial
        # Lambda = np.random.beta(8.0, 8.0)
        # idx_rp = torch.randperm(x_w.shape[0])
        # x_w_rp = x_w[idx_rp]
        # x_s_rp = x_s[idx_rp]
        # x_s_rp_ = x_s_[idx_rp]

        # x_w_mix = Lambda * x_w + (1 - Lambda) * x_w_rp
        # x_s_mix = Lambda * x_s + (1 - Lambda) * x_s_rp
        # x_s_mix_ = Lambda * x_s_ + (1 - Lambda) * x_s_rp_


        x_aug = torch.cat([x_w, x_s, x_s_], dim=0)
        y_pred, feat = model(x_aug)

        y_pred_w, y_pred_s, y_pred_s_ = y_pred.chunk(3)
        _, zq1, zq2 = feat.chunk(3)
        # y_pred_w, y_pred_s, y_pred_s_, y_pred_w_mix, y_pred_s_mix, y_pred_s_mix_ = y_pred.chunk(6)
        # _, zq1, zq2, _, _, _ = feat.chunk(6)

        noise_matrix = noise_model()
        probs_x_w = y_pred_w.softmax(dim=-1).detach()

        noisy_probs_x_w = torch.matmul(y_pred_w.softmax(dim=-1), noise_matrix)
        noisy_probs_x_w = noisy_probs_x_w / noisy_probs_x_w.sum(dim=-1, keepdim=True)

        # Supervised partial noise loss
        sup_partial_noise_loss = torch.mean(-torch.sum(torch.log(1.0000001 - noisy_probs_x_w) * (1 - partial_noise_y), dim=1))

        # VolMinNet loss
        vol_loss = noise_matrix.slogdet().logabsdet

        # for consistency loss
        noisy_probs_x_w_log = torch.log(noisy_probs_x_w)

        noisy_probs_x_s = torch.matmul(y_pred_s.softmax(dim=-1), noise_matrix)
        noisy_probs_x_s = noisy_probs_x_s / noisy_probs_x_s.sum(dim=-1, keepdim=True)
        noisy_probs_x_s_log = torch.log(noisy_probs_x_s)

        noisy_probs_x_s_ = torch.matmul(y_pred_s_.softmax(dim=-1), noise_matrix)
        noisy_probs_x_s_ = noisy_probs_x_s_ / noisy_probs_x_s_.sum(dim=-1, keepdim=True)
        noisy_probs_x_s_log_ = torch.log(noisy_probs_x_s_)

        y_pred_w_probas = torch.softmax(y_pred_w, dim=-1)
        y_pred_s_probas = torch.softmax(y_pred_s, dim=-1)
        y_pred_s_probas_ = torch.softmax(y_pred_s_, dim=-1)

        if args.dataset in ['cifar100', 'cifar10']:
            cur_confidence = torch.tensor(confidence[index]).float().cuda()
            con_loss = consistency_criterion(noisy_probs_x_w_log, cur_confidence)
            con_loss += consistency_criterion(noisy_probs_x_s_log, cur_confidence)
            con_loss += consistency_criterion(noisy_probs_x_s_log_, cur_confidence)
        elif args.dataset in ['cub200']:
            con_loss = ce_loss(
                torch.cat([y_pred_w, y_pred_s, y_pred_s_], dim=0),
                torch.cat([torch.tensor(confidence[index]).float().cuda(), torch.tensor(confidence[index]).float().cuda(), torch.tensor(confidence[index]).float().cuda(),], dim=0),
                reduction='mean'
                )
        else:
            raise NotImplementedError(f"Dataset {args.dataset} is not implemented for consistency loss.")
        
        # con_loss = ce_loss(y_pred_s, probs_x_w, reduction='mean')
        # con_loss += ce_loss(y_pred_s_, probs_x_w, reduction='mean')

        # with torch.no_grad():
        #     p_hat = ema_p * p_hat + (1 - ema_p) * probs_x_w.mean(dim=0)
        #     pseudo_label = probs_x_w / p_hat
        #     pseudo_label = pseudo_label / pseudo_label.sum(dim=-1, keepdim=True)

        # con_loss = ce_loss(y_pred_s, pseudo_label, reduction='mean')
        # con_loss += ce_loss(y_pred_s_, pseudo_label, reduction='mean')
        # cur_confidence = torch.tensor(confidence[index]).float().cuda()
        # con_loss = consistency_criterion(noisy_probs_x_w_log, cur_confidence)
        # con_loss += consistency_criterion(noisy_probs_x_s_log, cur_confidence)
        # con_loss += consistency_criterion(noisy_probs_x_s_log_, cur_confidence)

        
        
        if args.dataset in ['cifar100', 'cifar10']:
            loss = sup_partial_noise_loss + lam * (wsc_loss + con_loss) + args.vol_lambda * vol_loss
        elif args.dataset == 'cub200':
            loss = sup_partial_noise_loss + args.vol_lambda * vol_loss + con_loss
        # loss = sup_partial_noise_loss + lam * con_loss + args.vol_lambda * vol_loss
        # loss = sup_partial_noise_loss + args.vol_lambda * vol_loss

        # wsc_loss
        if epoch >= args.wp:
            q = q_calculate(confidence2, index)
            
            wsc_loss, l1, l2 = weak_spec_loss(zq1, zq2, q)
            lam = min(((epoch - args.wp) / (args.epochs - args.wp)) * args.lam, args.lam)
            loss += lam * wsc_loss
        else:
            wsc_loss, l1, l2 = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        if args.average_entropy_loss:
            avg_prediction = torch.mean(y_pred_w_probas, dim=0)
            prior_distr = 1.0 / args.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min=1e-6, max=1.0)
            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = args.balance_lam * balance_kl
            loss += entropy_loss

        optimizer.zero_grad()
        noise_matrix_optimizer.step()
        loss.backward()
        optimizer.step()
        noise_matrix_optimizer.zero_grad()

        scheduler.step()

        if args.dataset in ['cifar10', 'cifar100']:
            confidence = confidence_update(confidence, noisy_probs_x_w, noisy_probs_x_s, noisy_probs_x_s_, partial_noise_y, index, args)
            confidence2 = confidence_update(confidence2, y_pred_w_probas, y_pred_s_probas, y_pred_s_probas_, partial_noise_y, index, args)
        else:
            # for cub200
            confidence = confidence_update2(confidence, noisy_probs_x_w, partial_noise_y, index, args)
            confidence2 = confidence_update2(confidence2, y_pred_w_probas, partial_noise_y, index, args)

        losses.update(loss.item(), x_w.size(0))
        sup_losses.update((sup_partial_noise_loss).item(), x_w.size(0))
        vol_losses.update(args.vol_lambda * vol_loss.item(), x_w.size(0))
        wsc_losses.update(wsc_loss.item(), x_w.size(0))
        consist_losses.update((con_loss).item(), x_w.size(0))
        l1_losses.update(l1.item(), x_w.size(0))
        l2_losses.update(l2.item(), x_w.size(0))

        progress_bar.set_postfix({
                "loss": f"{losses.val:.4f}",
                "sup": f"{sup_losses.val:.4f}",
                "vol": f"{vol_losses.val:.4f}",
                "wsc": f"{wsc_losses.val:.4f}",
                "con": f"{consist_losses.val:.4f}",
                "l1": f"{l1_losses.val:.4f}",
                "l2": f"{l2_losses.val:.4f}",
            })
        
        if args.wandb:
            wandb.log({
                    "train/loss": losses.val,
                    "train/sup_loss": sup_losses.val,
                    "train/wsc_loss": wsc_losses.val,
                    "train/consist_loss": consist_losses.val,
                    "train/l1_loss": l1_losses.val,
                    "train/l2_loss": l2_losses.val
                })
            
    loss_dict = {
        "loss": losses.avg,
        "sup_loss": sup_losses.avg,
        "vol_loss": vol_losses.avg,
        "wsc_loss": wsc_losses.avg,
        "consist_loss": consist_losses.avg
    }
            
    return loss_dict, confidence, confidence2


def confidence_update2(confidence, y_pred, part_y, index, args):
    y_pred_probs = y_pred.detach()
    revisedY0 = part_y.clone()
    revisedY0 = revisedY0 * y_pred_probs
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(args.num_classes, 1).transpose(0, 1)
    confidence[index, :] = revisedY0.cpu().numpy()
    return confidence


def confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index, args):
    y_pred_aug0_probas = y_pred_aug0_probas.detach()
    y_pred_aug1_probas = y_pred_aug1_probas.detach()
    y_pred_aug2_probas = y_pred_aug2_probas.detach()

    revisedY0 = part_y.clone()
    revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(args.num_classes, 1).transpose(0, 1)

    confidence[index, :] = revisedY0.cpu().numpy()
    return confidence


def q_calculate(confidence, index):
    return torch.tensor(confidence[index]).float().cuda()


def dict2str(d):
    return ', '.join([f"{k}: {v:.4f}" for k, v in d.items()])


if __name__ == "__main__":
    main(args)