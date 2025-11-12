import copy
import torch
import numpy as np 
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision import datasets as vision_datasets
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence

from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy
from datasets.rand_aug import RandAugment


norm_mean_std_dict = {
    'cifar100': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
    'clothing1m': [(0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)],
    'mnist': [(0.1307,), (0.3081, )],
    'fmnist': [(0.1307,), (0.3081, )],
    'stl10': [[x / 255 for x in [112.4, 109.1, 98.6]], [x / 255 for x in [68.4, 66.6, 68.5]]]
}


def compose_transform(img_size=32, 
                      crop_ratio=0.875, 
                      is_train=True,
                      resize='rpn',
                      autoaug='randaug',
                      rand_erase=True,
                      hflip=True,
                      norm_mean_std=[(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
                      args=None):

    if not is_train:
        transform_list = []
        if resize == 'resize_crop':
            transform_list.extend([
                transforms.Resize((int(img_size / crop_ratio), int(img_size / crop_ratio))),
                transforms.CenterCrop((img_size, img_size)),
            ])
        else:
            transform_list.append(transforms.Resize((img_size, img_size)))
            
        transfrom = transforms.Compose([
            *transform_list,
            transforms.ToTensor(),
            transforms.Normalize(*norm_mean_std),
        ])
        return transfrom
    
    transform_list = []
    if resize == 'rpc':
        transform_list.append(transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)))
    elif resize == 'resize_rpc':
        transform_list.append(transforms.Resize((int(img_size / crop_ratio), int(img_size / crop_ratio))))
        transform_list.append(transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)))
    elif resize == 'resize_crop':
        transform_list.append(transforms.Resize((int(img_size / crop_ratio), int(img_size / crop_ratio))))
        transform_list.append(transforms.RandomCrop((img_size, img_size)))
    elif resize == 'resize_crop_pad':
        transform_list.append(transforms.Resize((img_size, img_size)))
        transform_list.append(transforms.RandomCrop((img_size, img_size), padding=int(img_size * (1 - crop_ratio)), padding_mode='reflect'))
    
    if hflip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if autoaug == 'randaug':
        transform_list.append(RandAugment(3, 5))
        rand_erase = False
    elif autoaug == 'autoaug_cifar':
        transform_list.append(transforms.AutoAugment(transforms. AutoAugmentPolicy.CIFAR10))  
    elif autoaug == 'autoaug':
        transform_list.append(transforms.AutoAugment()) 
    elif autoaug is None:
        rand_erase = False
    else:
        raise NotImplementedError
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(*norm_mean_std),
    ])
    

    if rand_erase and autoaug != 'randaug' and autoaug is not None:
        # transform_list.append(CutoutDefault(scale=cutout))
        transform_list.append(transforms.RandomErasing())
    
    
    transform = transforms.Compose(transform_list)
    return transform

def get_img_transform(args, types='partial'):
    if types == 'partial':
        if args.dataset in ['cifar10', 'cifar100']:
            w_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            s_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.ToPILImage(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif args.dataset == 'cub200':
            w_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            s_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.img_size),
                transforms.RandomHorizontalFlip(),
                RandAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        test_transform = transforms.Compose([
            transforms.Resize(int(args.img_size / 0.875)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return w_transform, s_transform, test_transform

    elif types == 'noise':
        if args.dataset in ["cifar10", "cifar100", "cifar10n", "cifar100n"]:
            resize = "resize_crop_pad"
            autoaug = "autoaug_cifar"
            test_resize = "resize"
        elif args.dataset == "clothing1m":
            resize = "resize_crop"
            autoaug = "autoaug"
            test_resize = "resize_crop"
        elif args.dataset == "webvision":
            resize = "resize_rpc"
            autoaug = "autoaug"
            test_resize = "resize"
        else:
            resize = "rpc"
            autoaug = "randaug"
            test_resize = "resize_crop"

        w_transform = compose_transform(args.img_size, args.crop_ratio, True, resize=resize, autoaug=None, norm_mean_std=norm_mean_std_dict.get(args.dataset, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]), args=args)

        s_transform = compose_transform(args.img_size, args.crop_ratio, True, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(args.dataset, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]), args=args)

        test_transform = compose_transform(args.img_size, args.crop_ratio, False, resize=test_resize, norm_mean_std=norm_mean_std_dict.get(args.dataset, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]), args=args)

        return w_transform, s_transform, test_transform
    
    elif types == 'semi':
        if args.dataset in ["cifar100", "cifar10", "stl10"]:
            s_transform = transforms.Compose([
                transforms.Resize(args.crop_ratio),
                transforms.RandomCrop(args.crop_ratio, padding=int(args.crop_ratio * (1 - args.crop_ratio)), padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                RandAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize(*norm_mean_std_dict.get(args.dataset, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)])),
            ])
            w_transform = transform_weak = transforms.Compose([
                transforms.Resize(args.crop_ratio),
                transforms.RandomCrop(args.crop_ratio, padding=int(args.crop_ratio * (1 - args.crop_ratio)), padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*norm_mean_std_dict.get(args.dataset, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)])),
            ])
        else:
            raise NotImplementedError(f"Unknown dataset: {args.dataset}")
        test_transform  = transforms.Compose([
            transforms.Resize(args.crop_ratio),
            transforms.ToTensor(),
            transforms.Normalize(*norm_mean_std_dict.get(args.dataset, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)])),
        ])
        return w_transform, s_transform, test_transform
    
    elif types == 'noisy_partial':
        resize = "resize_crop_pad"
        aug = "randaug"
        test_resize = "resize"

        w_transform = compose_transform(args.img_size, args.crop_ratio, True, resize=resize, autoaug=None, norm_mean_std=norm_mean_std_dict.get(args.dataset, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]), args=args)
        s_transform = compose_transform(args.img_size, args.crop_ratio, True, resize=resize, autoaug=aug, norm_mean_std=norm_mean_std_dict.get(args.dataset, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]), args=args)
        test_transform = compose_transform(args.img_size, args.crop_ratio, False, resize=test_resize, norm_mean_std=norm_mean_std_dict.get(args.dataset, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]), args=args)

        return w_transform, s_transform, test_transform

    else:
        raise NotImplementedError(f"Unknown imprecise label type: {types}")


class ImgThreeViewDataset(Dataset):
    def __init__(self, args, data, targets, types='partial', is_train=True, num_classes=10, class_map=None):
        super().__init__()
        self.data = data 
        self.targets = targets
        self.num_classes = num_classes
        self.class_map = class_map
        self.types = types
        self.w_transform, self.s_transform, self.test_transform = get_img_transform(args, self.types)
        args.logger.info(f"w_transform: {self.w_transform}")
        args.logger.info(f"s_transform: {self.s_transform}")

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if isinstance(data, str):
            data = Image.open(data).convert('RGB')   
        else:
            data = Image.fromarray(data)
        
        data_aug_w = self.w_transform(data)
        data_aug_s = self.s_transform(data)
        data_aug_s_ = self.s_transform(data)
        
        return data_aug_w, data_aug_s, data_aug_s_, target, index
    
    def __len__(self):
        return len(self.data)
    
class ImgBaseDataset(Dataset):
    def __init__(self, args, data, targets, types='partial', is_train=True, num_classes=10):
        super(ImgBaseDataset, self).__init__()

        self.data = data 
        self.targets = targets
        self.num_classes = num_classes
        self.types = types

        _, _, self.test_transform = get_img_transform(args, self.types)
        # args.logger.info(f"test_transform: {self.test_transform}")

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if isinstance(data, str):
            data = Image.open(data).convert('RGB')   
        else:
            data = Image.fromarray(data)
        
        data_aug = self.test_transform(data)
        
        return data_aug, target, index
    
    def __len__(self):
        return len(self.data)

# class ImgBaseDataset(Dataset):
#     def __init__(self, data_name, data, targets, is_train=True, num_classes=10, class_map=None,
#                  img_size=32, crop_ratio=0.875, autoaug='randaug', resize='rpc',
#                  return_target=True, return_idx=False, return_keys=['x_lb', 'y_lb']):
#         super(ImgBaseDataset, self).__init__()

#         self.data_name = data_name
#         self.data = data 
#         self.targets = targets
#         self.num_classes = num_classes
#         self.return_target = return_target
#         self.return_keys = return_keys
#         self.return_idx = return_idx
#         self.class_map = class_map
    
#         if self.class_map is not None and len(self.class_map) != num_classes:
#             print("select data from %s" % str(self.class_map))
#             selected_data = []
#             selected_targets = []
#             for idx in range(len(targets)):
#                 if targets[idx] not in self.class_map:
#                     continue
#                 selected_data.append(data[idx])
#                 selected_targets.append(targets[idx])
#             self.data = selected_data
#             self.targets = selected_targets
        
#         self.transform = get_img_transform(img_size, crop_ratio, is_train=is_train, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(data_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
    
#     def __getitem__(self, index):
#         data, target = self.data[index], self.targets[index]
#         if isinstance(data, str):
#             data = Image.open(data).convert('RGB')   
#         else:
#             data = Image.fromarray(data)
        
#         data_aug = self.transform(data)
        
#         if self.class_map is not None:
#             target = self.class_map[target]
        
#         if self.return_idx:
#             return_items = [index, data_aug]
#         else:
#             return_items = [data_aug]
            
#         if self.return_target:
#             return_items.append(target)
            
#         return_dict = {k:v for k,v in zip(self.return_keys, return_items)}
#         return return_dict

#     def __len__(self):
#         return len(self.data)


# class ImgTwoViewBaseDataset(ImgBaseDataset):
#     def __init__(self, data_name, data, targets, is_train=True, num_classes=10, class_map=None,
#                  img_size=32, crop_ratio=0.875, autoaug='randaug', resize='rpc',
#                  return_target=True, 
#                  return_idx=False,
#                  return_keys=['x_ulb_w', 'x_ulb_s', 'y_ulb']):
#         super().__init__(data_name, data, targets, is_train, num_classes, class_map, img_size, crop_ratio, None, resize, return_target, return_idx, return_keys)
#         self.strong_transform = get_img_transform(img_size, crop_ratio, is_train=is_train, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(data_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
    
#     def __getitem__(self, index):
#         data, target = self.data[index], self.targets[index]
#         if isinstance(data, str):
#             data = Image.open(data).convert('RGB')   
#         else:
#             data = Image.fromarray(data)
#         data_aug_w = self.transform(data)
#         data_aug_s = self.strong_transform(data)
#         if self.class_map is not None:
#             target = self.class_map[target]
        
#         if self.return_idx:
#             return_items = [index, data_aug_w, data_aug_s]
#         else:
#             return_items = [data_aug_w, data_aug_s]
            
#         if self.return_target:
#             return_items.append(target)
            
#         return_dict = {k:v for k,v in zip(self.return_keys, return_items)}
#         return return_dict

# class ImgThreeViewBaseDataset(ImgTwoViewBaseDataset):
#     def __init__(self, data_name, data, targets, is_train=True, num_classes=10, class_map=None,
#                  img_size=32, crop_ratio=0.875, autoaug='randaug', resize='rpc',
#                  return_target=True, 
#                  return_idx=False,
#                  return_keys=['x_ulb_w', 'x_ulb_s', 'x_ulb_s_', 'y_ulb']):
#         super().__init__(data_name, data, targets, is_train, num_classes, class_map, img_size, crop_ratio, None, resize, return_target, return_idx, return_keys)
#         self.strong_transform = get_img_transform(img_size, crop_ratio, is_train=is_train, resize=resize, autoaug=autoaug, norm_mean_std=norm_mean_std_dict.get(data_name, [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]))
    
#     def __getitem__(self, index):
#         data, target = self.data[index], self.targets[index]
#         if isinstance(data, str):
#             data = Image.open(data).convert('RGB')   
#         else:
#             data = Image.fromarray(data)
#         data_aug_w = self.transform(data)
#         data_aug_s = self.strong_transform(data)
#         data_aug_s_ = self.strong_transform(data)
#         if self.class_map is not None:
#             target = self.class_map[target]
        
#         if self.return_idx:
#             return_items = [index, data_aug_w, data_aug_s, data_aug_s_]
#         else:
#             return_items = [data_aug_w, data_aug_s, data_aug_s_]
            
#         if self.return_target:
#             return_items.append(target)
            
#         return_dict = {k:v for k,v in zip(self.return_keys, return_items)}
#         return return_dict

    
