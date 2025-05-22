import os
import torch
import numpy as np
import argparse
import copy
from tqdm import tqdm
import torch.nn.functional as F

from models import build_wideresnet, resnet18con
from datasets.base_data import get_data
from datasets.imprecise_label import get_partial_labels
from utils.configs import parse_arg_partial
from utils.metrics import AverageMeter, accuracy
from utils.others import reproducibility, save_checkpoint
from utils.validate import validate
from datasets.base_datasets import ImgBaseDataset, ImgThreeViewDataset
from loss import WeakSpectralLoss, SupervisedPartialLoss, ce_loss


args = parse_arg_partial()
if args.wandb:
    import wandb
    wandb.init(project="wsc_partial", config=args)
    wandb.run.name = f"{args.dataset}_p_rate_{args.partial_rate}_trial_{args.trial}_{args.time}"
    wandb.config.update(args)
    wandb.run.save()



def main(args: argparse.Namespace):
    # for reproducibility
    reproducibility(args.seed)

    # load dataset
    train_data, train_target, test_data, test_target = get_data(args.data_path, args.dataset)

    # make partial label dataset
    train_data, train_partial_target = get_partial_labels(train_data, train_target, args.num_classes, args.partial_rate)

    train_dataset = ImgThreeViewDataset(args, train_data, train_partial_target)
    test_dataset = ImgBaseDataset(args, test_data, test_target)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # model
    if args.model == "widenet":
        model = build_wideresnet(28, 2, 0.0, args.num_classes)
    elif args.model == "resnet18":
        # it is special for cub200
        model = resnet18con()
    else: 
        raise NotImplementedError(f'{args.model} is not implemented')
    
    model = model.cuda()

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # scheduler
    if args.dataset in ['cifar10', 'cifar100']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[160, 180],
            last_epoch=-1,
        )
    else:
        # for cub200, use cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min = args.lr * 0.001,
        )

    # criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()
    consistency_criterion = torch.nn.KLDivLoss(reduction='batchmean').cuda()
    weak_spec_loss = WeakSpectralLoss(args.alpha, args.beta, args)
    supervised_loss = SupervisedPartialLoss()

    confidence = copy.deepcopy(train_loader.dataset.targets)
    confidence = confidence / confidence.sum(axis=1)[:, None]

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            confidence = checkpoint['confidence']

    # training
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        
        train_loss = train(train_loader, model, optimizer, epoch, supervised_loss, consistency_criterion, weak_spec_loss, confidence, args)
        scheduler.step()

        val_loss, val_acc = validate(test_loader, model, criterion, args)
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            is_best = True
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'confidence': confidence,
        }, is_best, args)
        

def train(train_loader, model, optimizer, epoch, sup_loss, consistency_criterion, weak_spec_loss, confidence, args):
    '''
    train for one epoch
    '''

    losses = AverageMeter()
    sup_losses = AverageMeter()
    wsc_losses = AverageMeter()
    consist_losses = AverageMeter()
    l1_losses = AverageMeter()
    l2_losses = AverageMeter()

    model.train()
    progress_bar = tqdm(enumerate(train_loader) ,total=len(train_loader), desc=f"Train Epoch: [{epoch}]", ncols=0)

    if args.dataset == "cub200":
        weak_spec_loss.beta = min((epoch / args.epochs) * args.beta, args.beta)

    for i, (x_w, x_s, x_s_, part_y, index) in progress_bar:
        part_y = part_y.float().cuda()
        x_aug = torch.cat((x_w, x_s, x_s_), dim=0).cuda()
        y_pred_aug, feat = model(x_aug)
        y_pred_w, y_pred_s, y_pred_s_ = y_pred_aug.chunk(3)
        _, zq1, zq2 = feat.chunk(3)

        # supervised loss
        supervised_loss = sup_loss(y_pred_w, part_y)
        loss = supervised_loss

        # wsc loss
        q = q_calculate(confidence, index)
        wsc_loss, l1, l2 = weak_spec_loss(zq1, zq2, q)
        lam = min((epoch / args.epochs) * args.lam, args.lam)

        loss += lam * wsc_loss

        # for consistency loss
        y_pred_w_probas_log = torch.log_softmax(y_pred_w, dim=-1)
        y_pred_s_probas_log = torch.log_softmax(y_pred_s, dim=-1)
        y_pred_s_probas_log_ = torch.log_softmax(y_pred_s_, dim=-1)

        y_pred_w_probas = torch.softmax(y_pred_w, dim=-1)
        y_pred_s_probas = torch.softmax(y_pred_s, dim=-1)
        y_pred_s_probas_ = torch.softmax(y_pred_s_, dim=-1)

        if args.dataset in ['cifar100', 'cifar10']:
            consist_loss0 = consistency_criterion(y_pred_w_probas_log, torch.tensor(confidence[index]).float().cuda())
            consist_loss1 = consistency_criterion(y_pred_s_probas_log, torch.tensor(confidence[index]).float().cuda())
            consist_loss2 = consistency_criterion(y_pred_s_probas_log_, torch.tensor(confidence[index]).float().cuda())
            consist_loss = consist_loss0 + consist_loss1 + consist_loss2

            loss += lam * consist_loss
        elif args.dataset in ['cub200']:
            consist_loss = ce_loss(
                torch.cat([y_pred_w, y_pred_s, y_pred_s_], dim=0),
                torch.cat([torch.tensor(confidence[index]).float().cuda(), torch.tensor(confidence[index]).float().cuda(), torch.tensor(confidence[index]).float().cuda(),], dim=0),
                reduction='mean'
                )
            
            loss += consist_loss
        else:
            raise NotImplementedError(f'{args.dataset} is not implemented')
        
        '''
        so the final loss is:
            >>> loss = supervised_loss + lam * wsc_loss + lam * consist_loss
        or:
            >>> loss = supervised_loss + lam * wsc_loss + consist_loss
        '''
        
        if args.average_entropy_loss:
            avg_prediction = torch.mean(y_pred_w_probas, dim=0)
            prior_distr = 1.0 / args.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min=1e-6, max=1.0)
            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = 0.1 * balance_kl
            loss += entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.dataset == 'cifar100':
            confidence_update(confidence, y_pred_w_probas, y_pred_s_probas, y_pred_s_probas_, part_y, index, args)
        else:
            confidence_update2(confidence, y_pred_w_probas, part_y, index, args)

        losses.update(loss.item(), x_w.size(0))
        sup_losses.update(sup_loss.item(), x_w.size(0))
        wsc_losses.update(wsc_loss.item(), x_w.size(0))
        consist_losses.update(consist_loss.item(), x_w.size(0))
        l1_losses.update(l1.item(), x_w.size(0))
        l2_losses.update(l2.item(), x_w.size(0))

        progress_bar.set_postfix({
            "loss": losses.avg,
            "sup_loss": sup_losses.avg,
            "wsc_loss": wsc_losses.avg,
            "consist_loss": consist_losses.avg,
            "l1_loss": l1_losses.avg,
            "l2_loss": l2_losses.avg
        })

        if args.wandb:
            wandb.log({
                "train/loss": losses.avg,
                "train/sup_loss": sup_losses.avg,
                "train/wsc_loss": wsc_losses.avg,
                "train/consist_loss": consist_losses.avg,
                "train/l1_loss": l1_losses.avg,
                "train/l2_loss": l2_losses.avg
            })
    
    return losses.avg


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


def confidence_update2(confidence, y_pred, part_y, index, args):
    y_pred_probs = y_pred.detach()
    revisedY0 = part_y.clone()
    revisedY0 = revisedY0 * y_pred_probs
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(args.num_classes, 1).transpose(0, 1)
    confidence[index, :] = revisedY0.cpu().numpy()


def q_calculate(confidence, index):
    return torch.tensor(confidence[index]).float().cuda()

if __name__ == "__main__":
    main(args)

