#############################################################
##  Warning: This is not finished! Do not use it for now!  ##
#############################################################

from sched import scheduler
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import models

from loss import WeakSpectralLoss
from utils.metrics import AverageMeter, accuracy, count_parameters
from utils.configs import parse_arg_semi
from utils.others import reproducibility, param_groups_layer_decay, param_groups_weight_decay, get_cosine_schedule_with_warmup
from utils.validate import validate
from datasets.imprecise_label import get_semisup_labels
from datasets.base_data import get_data
from datasets.base_datasets import ImgBaseDataset, ImgThreeViewDataset, ImgTwoViewBaseDataset

args = parse_arg_semi()
if args.wandb:
    import wandb
    wandb.init(project="wsc_semi", config=args)
    wandb.run.name = f"{args.dataset}_label_count_{args.noise_ratio}_{args.time}"
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

    # get semi-supervise data
    args.logger.info(f"labeled data number: {args.num_labels}")
    lb_train_data, lb_train_targets, ulb_train_data, ulb_train_targets = get_semisup_labels(
        train_data, train_targets, num_classes=args.num_classes, num_labels=args.num_labels
    )
    train_lb_dataset = ImgThreeViewDataset(args, lb_train_data, lb_train_targets, "semi")
    train_ulb_dataset = ImgThreeViewDataset(args, ulb_train_data, ulb_train_targets, "semi")
    test_dataset = ImgBaseDataset(args, test_data, test_targets, "semi")

    # get data loader
    args.logger.info(f"Creating data loader with batch size {args.batch_size}")
    train_lb_loader = DataLoader(
        train_lb_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    train_ulb_loader = DataLoader(
        train_ulb_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # get model
    model = getattr(models, args.model)(num_classes=args.num_classes, pretrained=args.use_pretrain, pretrained_path=args.pretrain_path)

    model = model.cuda()
    args.logger.info(f"Model {args.model} created, with {count_parameters(model)} trainable parameters")

    # optimizer
    no_decay = {}
    if hasattr(model, 'no_weight_decay'):
        no_decay = model.no_weight_decay()

    if args.layer_decay != 1.0:
        per_param_args = param_groups_layer_decay(model, args.lr, args.weight_decay, no_weight_decay_list=no_decay, layer_decay=args.layer_decay)
    else:
        per_param_args = param_groups_weight_decay(model, args.weight_decay, no_weight_decay_list=no_decay)
    
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            per_param_args,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.AdamW(
            per_param_args,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    
    # scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=args.num_train_iters,
        num_warmup_steps=args.num_warmup_iters,
    )

    # loss
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    weak_spec_loss = WeakSpectralLoss(args.alpha, args.beta, args)

    # TODO: add resume

    # training
    best_acc = 0.0
    iters = 0
    for epoch in range(args.start_epoch, args.epochs):

        if iters >= args.num_train_iters:
            break

        model.train()

        train_loss, l1, l2 = train(args, train_lb_loader, train_ulb_loader, model, optimizer, scheduler, weak_spec_loss, iters, epoch)

        val_loss, val_acc = validate(test_loader, model, criterion, args)
        
        args.logger.info(f"Epoch {epoch + 1}: LR: {optimizer.param_groups[0]['lr']:.4f} Train Loss: {train_loss:.4f} L1 Loss: {l1:.4f} L2 Loss: {l2:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        if args.wandb:
            wandb.log({
                "train/loss": train_loss,
                "train/l1": l1,
                "train/l2": l2,
                "val/loss": val_loss,
                "val/acc": val_acc
            })


def train(args, train_lb_loader, train_ulb_loader, model, optimizer, scheduler, weak_spec_loss, iters, epoch):
    losses = AverageMeter()
    l1_losses = AverageMeter()
    l2_losses = AverageMeter()

    progress_bar = tqdm(zip(train_lb_loader, train_ulb_loader), total=len(train_lb_loader), desc=f"Train Epoch: [{epoch + 1}]")

    for (lb_w, lb_s, lb_s_, lb_y), (ulb_w, ulb_s, ulb_s_, _) in progress_bar:
        if iters >= args.num_train_iters:
            break

        lb_w, lb_s, lb_s_, lb_y = lb_w.cuda(), lb_s.cuda(), lb_s_.cuda(), lb_y.cuda()
        ulb_w, ulb_s, ulb_s_ = ulb_w.cuda(), ulb_s.cuda(), ulb_s_.cuda()

        inputs = torch.cat([lb_w, lb_s, lb_s_, ulb_w, ulb_s, ulb_s_], dim=0)
        output_dict = model(inputs)
        logits, feats = output_dict["logits"], output_dict["feats"]
        feats = F.normalize(feats, dim=1)
        logits_lb_w, _, _, _, _, _ = logits.chunk(6)
        _ ,feat_lb_s, feat_lb_s_, _, feat_ulb_s, feat_ulb_s_ = feats.chunk(6)

        probs_lb_w = logits_lb_w.softmax(dim=-1).detach()

        wsc_loss, l1, l2 = weak_spec_loss(feat_lb_s, feat_lb_s_, probs_lb_w, feat_ulb_s, feat_ulb_s_)

        total_loss = args.lam * wsc_loss

            # update parameters
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        l1_losses.update(l1.item(), lb_w.size(0))
        l2_losses.update(l2.item(), lb_w.size(0))
        losses.update(total_loss.item(), lb_w.size(0))

        iters += 1
        progress_bar.set_postfix({
                "loss": f"{losses.avg:.4f}",
                "l1": f"{l1_losses.avg:.4f}",
                "l2": f"{l2_losses.avg:.4f}",
            })

        if args.wandb:
            wandb.log({
                    "train/loss": losses.val,
                    "train/l1": l1_losses.val,
                    "train/l2": l2_losses.val,
                })
            
    return losses.avg, l1_losses.avg, l2_losses.avg


if __name__ == "__main__":
    main(args)