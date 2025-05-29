import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

import os

from models import preact_resnet18, PretrainedResNet50, ResNet34, NoiseMatrixLayer, inception_resnet_v2
from datasets.imprecise_label import get_sym_noisy_labels, get_cifar10_asym_noisy_labels, get_cifar100_asym_noisy_labels
from datasets.base_data import get_data
from datasets.base_datasets import ImgBaseDataset, ImgThreeViewDataset
from utils.configs import parse_arg_noisy
from utils.others import reproducibility, save_checkpoint, param_groups_weight_decay
from utils.metrics import AverageMeter, accuracy
from utils.validate import validate
from utils.logger import Logger
from loss import ce_loss, WeakSpectralLoss, SupervisedNoisyLoss



def main_worker(local_rank, ngpu_per_node, args):
    # initialize distributed training
    args.local_rank = local_rank
    args.rank = local_rank
    args.world_size = ngpu_per_node


    if args.distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=local_rank,
        )
    else:
        torch.cuda.set_device(local_rank)

    is_main_process = (not args.distributed) or (dist.get_rank() == 0)

    if args.wandb and is_main_process:
        import wandb
        wandb.init(project="wsc_noisy", config=args)
        wandb.run.name = f"{args.dataset}_noise_rate_{args.noise_ratio}_{args.time}"
        wandb.config.update(args)
        wandb.run.save()

    if is_main_process:
        args.logger = Logger(args.out)
        args.logger.log_args(args)

        # for reproducibility
        args.logger.info(f"Setting random seed to {args.seed}")
    
    reproducibility(args.seed)

    # load data
    if is_main_process:
        args.logger.info(f"Loading data {args.dataset} from {args.data_path}")
    train_data, train_targets, test_data, test_targets, _ = get_data(args.data_path, args.dataset)

    # get noise labels
    if is_main_process:
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
            noise_file = torch.load(os.path.join(args.data_path, "cifar100n", "CIFAR-100_human.pt"))
            assert args.noise_ratio in ['clean_label', 'noisy_label']
            train_noisy_targets = noise_file[args.noise_ratio]
        else:
            # noisy labels is directly loaded in train_targets
            train_noisy_targets = train_targets
    else:
        raise NotImplementedError(f"Noise type {noise_type} is not supported.")
    
    # create dataset
    if is_main_process:
        args.logger.info(f"Creating dataset {args.dataset}")
    train_dataset = ImgThreeViewDataset(args, train_data, train_noisy_targets, 'noise')
    test_dataset = ImgBaseDataset(args, test_data, test_targets, "noise")

    if is_main_process:
        args.logger.info(f"w_transform: {train_dataset.w_transform}")
        args.logger.info(f"s_transform: {train_dataset.s_transform}")
        args.logger.info(f"test_transform: {test_dataset.test_transform}")

    # create dataloader
    # args.logger.info(f"Creating dataloader with batch size {args.batch_size}")
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    if is_main_process:
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
    
    model = model.cuda(local_rank)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank,])
    if is_main_process:
        args.logger.info(f"Created model {args.model}")

    # our noise method should use noise matrix
    if is_main_process:
        args.logger.info(f"Creating noise matrix with scale {args.noise_matrix_scale}") 

    noise_model = NoiseMatrixLayer(args.num_classes, init=args.noise_matrix_scale)
    noise_model = noise_model.cuda(local_rank)
    noise_model = nn.parallel.DistributedDataParallel(noise_model, device_ids=[local_rank,])

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
        projector = create_projector(512, 256).cuda(local_rank)
    elif args.model == "inception_resnet_v2":
        projector = create_projector(1536, 256).cuda(local_rank)

    # optimizer
    per_param_args = param_groups_weight_decay(model, args.weight_decay, no_weight_decay_list={})
    if is_main_process:
        args.logger.info(f"Creating optimizer with weight decay {args.weight_decay} and momentum {args.momentum}")
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
            milestones=[50 * len(train_loader),],
            last_epoch=-1,
        )
    elif args.dataset == "clothing1m":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[7 * len(train_loader),],
            last_epoch=-1,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.epochs * len(train_loader)),
            eta_min = 2e-4,
        )

    # loss function
    criterion = torch.nn.CrossEntropyLoss().cuda(local_rank)
    sup_loss = SupervisedNoisyLoss(args.num_classes)
    weak_spec_loss = WeakSpectralLoss(args.alpha, args.beta, args)

    # resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.distributed:
                if is_main_process:
                    args.logger.info(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_acc = checkpoint['best_acc']
                model.module.load_state_dict(checkpoint['state_dict'])
                noise_model.module.load_state_dict(checkpoint['noise_model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                noise_matrix_optimizer.load_state_dict(checkpoint['noise_matrix_optimizer'])
            else:
                args.logger.info(f"Loading checkpoint '{args.resume}' on local rank {local_rank}")
                checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
                args.start_epoch = checkpoint['epoch']
                best_acc = checkpoint['best_acc']
                model.load_state_dict(checkpoint['state_dict'])
                noise_model.load_state_dict(checkpoint['noise_model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                noise_matrix_optimizer.load_state_dict(checkpoint['noise_matrix_optimizer'])
        else:
            if is_main_process:
                args.logger.error(f"No checkpoint found at '{args.resume}'")
    
    # training
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        is_best = False
        model.train()
        noise_model.train()

        train_loss = train(args, train_loader, model, noise_model, projector, optimizer, noise_matrix_optimizer, sup_loss, weak_spec_loss, scheduler, epoch, local_rank)

        if is_main_process:
            print(f"noise matrix:\n {noise_model(None)}")
        # scheduler.step()

        val_loss, val_acc = validate(test_loader, model, criterion, args)

        if is_main_process:
            if val_acc > best_acc:
                best_acc = val_acc
                is_best = True

            args.logger.info(f"Epoch {epoch+1}: LR: {optimizer.param_groups[0]['lr']:.4f} Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} Best Acc: {best_acc:.4f}")
            if args.distributed:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': model.module.state_dict(),
                    'noise_model': noise_model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'noise_matrix_optimizer': noise_matrix_optimizer.state_dict(),
                }, is_best, args)
            else:
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


def train(args, train_loader, model, noise_model, projector, optimizer, noise_matrix_optimizer, sup_loss, weak_spec_loss, scheduler, epoch, local_rank):
    losses = AverageMeter()
    sup_losses = AverageMeter()
    vol_losses = AverageMeter()
    wsc_losses = AverageMeter()
    consist_losses = AverageMeter()
    l1_losses = AverageMeter()
    l2_losses = AverageMeter()

    is_main_process = (not args.distributed) or dist.get_rank() == 0

    progress_bar = tqdm(enumerate(train_loader) ,total=len(train_loader), desc=f"Train Epoch: [{epoch + 1}]", disable=not is_main_process)

    for i, (x_w, x_s, x_s_, noise_y, index) in progress_bar:
        x_w = x_w.cuda(local_rank)
        x_s = x_s.cuda(local_rank)
        x_s_ = x_s_.cuda(local_rank)
        noise_y = noise_y.cuda(local_rank)

        x_aug = torch.cat([x_w, x_s, x_s_])
        y_pred, feat = model(x_aug)
        feat = projector(feat)
        if args.model in ["preact_resnet18", "inception_resnet_v2"]:
            feat = F.normalize(feat, dim=1)

        y_pred_w, y_pred_s, y_pred_s_ = y_pred.chunk(3)
        feat_w, feat_s, feat_s_ = feat.chunk(3)

        noise_matrix = noise_model(None)

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

        For the distributed training, we need to gather the features and probabilities from all the processes.
        """
        if args.distributed:
            feat_s_list = [torch.zeros_like(feat_s) for _ in range(dist.get_world_size())]
            feat_s__list = [torch.zeros_like(feat_s_) for _ in range(dist.get_world_size())]
            probs_x_w_list = [torch.zeros_like(probs_x_w) for _ in range(dist.get_world_size())]

            dist.all_gather(feat_s_list, feat_s)
            dist.all_gather(feat_s__list, feat_s_)
            dist.all_gather(probs_x_w_list, probs_x_w)

            feat_s_all = torch.cat(feat_s_list, dim=0)
            feat_s__all = torch.cat(feat_s__list, dim=0)
            probs_x_w_all = torch.cat(probs_x_w_list, dim=0)

            if dist.get_rank() == 0:
                # print(f"shape: feat_s_all: {feat_s_all.shape}, feat_s__all: {feat_s__all.shape}, probs_x_w_all: {probs_x_w_all.shape}")
                wsc_loss, l1, l2 = weak_spec_loss(feat_s_all, feat_s__all, probs_x_w_all)
            else:
                wsc_loss = torch.tensor(0.0, device=feat_s.device)
                l1 = torch.tensor(0.0, device=feat_s.device)
                l2 = torch.tensor(0.0, device=feat_s.device)

            # broadcast wsc_loss, l1, l2 to all processes
            dist.broadcast(wsc_loss, src=0)
            dist.broadcast(l1, src=0)
            dist.broadcast(l2, src=0)
        else:
            wsc_loss, l1, l2 = weak_spec_loss(feat_s, feat_s_, probs_x_w)

        # total loss
        lam = min(1, float(epoch)/float(args.epochs)) * args.lam
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

        optimizer.step()
        noise_matrix_optimizer.step()

        optimizer.zero_grad()
        noise_matrix_optimizer.zero_grad()

        scheduler.step()        

        # update meter
        losses.update(loss.item(), x_w.size(0))
        sup_losses.update((supervised_loss).item(), x_w.size(0))
        vol_losses.update(args.vol_lambda * vol_loss.item(), x_w.size(0))
        wsc_losses.update(wsc_loss.item(), x_w.size(0))
        consist_losses.update((con_loss).item(), x_w.size(0))
        l1_losses.update(l1.item(), x_w.size(0))
        l2_losses.update(l2.item(), x_w.size(0))

        if is_main_process:
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
                import wandb
                wandb.log({
                        "train/loss": losses.val,
                        "train/sup_loss": sup_losses.val,
                        "train/wsc_loss": wsc_losses.val,
                        "train/consist_loss": consist_losses.val,
                        "train/l1_loss": l1_losses.val,
                        "train/l2_loss": l2_losses.val
                    })
            
    return losses.avg


if __name__ == "__main__":
    args = parse_arg_noisy()
    ngpu_per_node = torch.cuda.device_count()
    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpu_per_node, args=(ngpu_per_node, args))
    else:
        # for single GPU training
        args.local_rank = 0
        main_worker(args.local_rank, ngpu_per_node, args)
    # main(args)
    