import torch
import torch.distributed as dist

from sklearn.metrics import confusion_matrix
import numpy as np
from utils.metrics import AverageMeter, accuracy
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def validate(test_loader, model, criterion, args):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    is_distributed = hasattr(args, "distributed") and args.distributed
    is_main_process = (not is_distributed) or (dist.is_initialized() and dist.get_rank() == 0)

    progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc='Test',
        disable=not is_main_process,
    )

    y_true = []
    y_pred = []
    with torch.no_grad():
        confusion_matrix = torch.zeros(args.num_classes, args.num_classes).cuda()
        for i, (x, y, index) in progress_bar:
            x = x.cuda()
            y = y.cuda()

            output = model(x)
            if isinstance(output, dict):
                output = output["logits"]
            loss = criterion(output, y)

            output = output.float()
            loss = loss.float()

            # Compute confusion matrix
            pred = output.argmax(dim=1)
            for t, p in zip(y.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # acc1 = accuracy(output, y)[0]
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(output, dim=-1)[1].cpu().tolist())

            losses.update(loss.item(), x.size(0))
            # top1.update(acc1, x.size(0))
            if is_main_process:
                progress_bar.set_postfix({
                    'loss': losses.avg,
                })
    if is_distributed and dist.is_initialized():
        y_true_tensor = torch.tensor(y_true, dtype=torch.long, device='cuda')
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.long, device='cuda')

        y_true_list = [torch.zeros_like(y_true_tensor) for _ in range(dist.get_world_size())]
        y_pred_list = [torch.zeros_like(y_pred_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(y_true_list, y_true_tensor)
        dist.all_gather(y_pred_list, y_pred_tensor)

        if is_main_process:
            y_true_all = torch.cat(y_true_list).cpu().numpy()
            y_pred_all = torch.cat(y_pred_list).cpu().numpy()
            acc = accuracy_score(y_true_all, y_pred_all)
        else:
            acc = None

        loss_tensor = torch.tensor(losses.sum, device='cuda')
        count_tensor = torch.tensor(losses.count, device='cuda')

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        global_loss = (loss_tensor / count_tensor).item() if count_tensor.item() > 0 else 0.0

        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
    else:
        acc = accuracy_score(np.array(y_true), np.array(y_pred))
        global_loss = losses.avg

    # Normalize confusion matrix
    if is_main_process:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = confusion_matrix.cpu().numpy()
        print("Confusion Matrix:")
        print(confusion_matrix)

        if args.wandb:
            import wandb
            wandb.log(
                {
                    "test/loss": global_loss,
                    "test/accuracy": acc * 100.0,
                    "test/confusion_matrix": wandb.Table(
                        data=confusion_matrix.tolist(),
                    ),
                }
            )
        
        return global_loss, acc * 100.0
    else:
        return global_loss, None