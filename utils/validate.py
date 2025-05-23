import enum
import torch
from utils.metrics import AverageMeter, accuracy
from tqdm import tqdm


def validate(test_loader, model, criterion, args):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc='Test',
    )

    with torch.no_grad():
        for i, (x, y, index) in progress_bar:
            x = x.cuda()
            y = y.cuda()

            output = model(x)
            loss = criterion(output, y)

            output = output.float()
            loss = loss.float()

            acc1 = accuracy(output, y)[0]
            
            losses.update(loss.item(), x.size(0))
            top1.update(acc1, x.size(0))

            progress_bar.set_postfix({
                'loss': losses.avg,
                'acc1': top1.avg
            })

            if args.wandb:
                import wandb
                wandb.log({
                    "test/loss": losses.avg,
                    "test/acc1": top1.avg
                })
        
    return losses.avg, top1.avg