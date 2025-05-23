import os 
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets as vision_datasets


from .utils import CUB200, Clothing1M
from .base_sampler import TrainIterDistributedSampler


def get_data(data_dir='./data', data_name='cifar10'):
    if data_name == 'cifar10' or data_name == 'cifar10n':
        train_dataset = vision_datasets.CIFAR10(root=data_dir, train=True, download=True)
        train_data, train_targets = train_dataset.data, train_dataset.targets
        test_dataset = vision_datasets.CIFAR10(root=data_dir, train=False, download=True)
        test_data, test_targets = test_dataset.data, test_dataset.targets
        extra_data = None 
    elif data_name == 'cifar100' or data_name == 'cifar100n':
        train_dataset = vision_datasets.CIFAR100(root=data_dir, train=True, download=True)
        train_data, train_targets = train_dataset.data, train_dataset.targets
        test_dataset = vision_datasets.CIFAR100(root=data_dir, train=False, download=True)
        test_data, test_targets = test_dataset.data, test_dataset.targets
        extra_data = None 
    elif data_name == 'svhn':
        train_dataset = vision_datasets.SVHN(root=data_dir, split='train', download=True)
        train_data, train_targets = train_dataset.data.transpose([0, 2, 3, 1]), train_dataset.labels
        test_dataset = vision_datasets.SVHN(root=data_dir, split='test', download=True)
        test_data, test_targets = test_dataset.data.transpose([0, 2, 3, 1]), test_dataset.labels
        extra_dataset = vision_datasets.SVHN(root=data_dir, split='extra', download=True)
        extra_data, extra_targets = extra_dataset.data.transpose([0, 2, 3, 1]), extra_dataset.labels
        train_data = np.concatenate([train_data, extra_data], axis=0)
        train_targets = np.concatenate([train_targets, extra_targets], axis=0)
        extra_data = None 
    elif data_name == 'stl10':
        train_dataset = vision_datasets.STL10(root=data_dir, split='train', download=True)
        train_data, train_targets = train_dataset.data.transpose([0, 2, 3, 1]), train_dataset.labels.astype(np.int64)
        test_dataset = vision_datasets.STL10(root=data_dir, split='test', download=True)
        test_data, test_targets = test_dataset.data.transpose([0, 2, 3, 1]), test_dataset.labels.astype(np.int64)
        extra_dataset = vision_datasets.STL10(root=data_dir, split='unlabeled', download=True)
        extra_data = extra_dataset.data.transpose([0, 2, 3, 1])
    elif data_name == 'cub':
        train_dataset = CUB200(root=data_dir, train=True, download=True)
        train_data, train_targets = train_dataset.train_data, train_dataset.train_labels
        test_dataset = CUB200(root=data_dir, train=False, download=True)
        test_data, test_targets = test_dataset.test_data, test_dataset.test_labels
        extra_data = None
    elif data_name == 'clothing1m':
        train_dataset = Clothing1M(root=data_dir, mode='all', transform=None, num_samples=3000 * 64)
        train_data, train_targets = train_dataset.train_imgs, train_dataset.train_labels
        test_dataset = Clothing1M(root=data_dir, mode='test', transform=None)
        test_data, test_targets = test_dataset.test_imgs, test_dataset.test_labels
        extra_data = None 
    elif data_name == 'mnist':
        train_dataset = vision_datasets.MNIST(root=data_dir, train=True, download=True)
        train_data, train_targets = train_dataset.data.numpy(), train_dataset.targets.numpy()
        test_dataset = vision_datasets.MNIST(root=data_dir, train=False, download=True)
        test_data, test_targets = test_dataset.data.numpy(), test_dataset.targets.numpy()
        extra_data = None 
    elif data_name == 'fmnist':
        train_dataset = vision_datasets.FashionMNIST(root=data_dir, train=True, download=True)
        train_data, train_targets = train_dataset.data.numpy(), train_dataset.targets.numpy()
        test_dataset = vision_datasets.FashionMNIST(root=data_dir, train=False, download=True)
        test_data, test_targets = test_dataset.data.numpy(), test_dataset.targets.numpy()
        extra_data = None 
    elif data_name == 'imagenet100' or data_name == 'imagenet1k':
        train_dataset = vision_datasets.ImageFolder(root=os.path.join(data_dir, data_name, 'train'))
        train_data, train_targets = [], []
        for path, target in train_dataset.samples:
            train_data.append(path)
            train_targets.append(target)
        test_dataset = vision_datasets.ImageFolder(root=os.path.join(data_dir, data_name, 'val'))
        test_data, test_targets = [], []
        for path, target in test_dataset.samples:
            test_data.append(path)
            test_targets.append(target)
        extra_data = None
    else:
        raise NotImplementedError

    return train_data, train_targets, test_data, test_targets, extra_data



def get_dataloader(dataset,
                   num_epochs=100,
                   num_train_iter=None,
                   batch_size=32,
                   shuffle=False,
                   num_workers=4,
                   pin_memory=False,
                   drop_last=False,
                   distributed=False,
                   collate_fn=None):

    if distributed:
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        batch_size = batch_size // num_replicas
        if num_train_iter is not None:
            per_epoch_steps = num_train_iter // num_epochs
            num_samples = per_epoch_steps * batch_size * num_replicas
            sampler = TrainIterDistributedSampler(dataset, num_replicas=num_replicas, rank=rank, num_samples=num_samples)
        else:
            sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)
        shuffle = False
    else:
        sampler = None 

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader