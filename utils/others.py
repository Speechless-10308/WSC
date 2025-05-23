import torch
import numpy as np
import shutil
import os
from datetime import datetime

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

def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    filename = os.path.join(args.out, filename)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.out, 'model_best.pth.tar'))
    print(f"Checkpoint saved to {filename}")
    print(f"Best model saved to {os.path.join(args.out, 'model_best.pth.tar')}")