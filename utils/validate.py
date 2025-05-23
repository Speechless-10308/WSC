import enum
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from utils.metrics import AverageMeter, accuracy
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def validate(test_loader, model, criterion, args):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc='Test',
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

            progress_bar.set_postfix({
                'loss': losses.avg,
            })

            if args.wandb:
                import wandb
                wandb.log({
                    "test/loss": losses.avg,
                    "test/acc1": top1.avg
                })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)


    # Normalize confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix = confusion_matrix.cpu().numpy()
    print("Confusion Matrix:")
    print(confusion_matrix)
        
    return losses.avg, acc * 100.0