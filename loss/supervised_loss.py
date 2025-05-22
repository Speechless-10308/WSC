import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedPartialLoss(nn.Module):
    def __init__(self):
        super(SupervisedPartialLoss, self).__init__()
        
    def forward(self, y_pred, part_y):
        y_pred_probas = F.softmax(y_pred, dim=1)
        complement_probas = 1.0000001 - y_pred_probas
        weighted_sum = torch.sum(torch.log(complement_probas)* (1-part_y), dim=1)
        loss = -torch.mean(weighted_sum)
        return loss
    
class SupervisedNoisyLoss(nn.Module):
    def __init__(self, num_classes=10):
        super(SupervisedNoisyLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, noisy_probs, target):
        one_hot_target = F.one_hot(target, num_classes=self.num_classes)
        noise_loss = torch.mean(
            -torch.sum(
                one_hot_target * torch.log(noisy_probs), dim=-1
            )
        )
        return noise_loss
    
    
def ce_loss(logits, targets, reduction='none'):
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
    

