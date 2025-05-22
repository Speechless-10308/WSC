import os 
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch


class WebVision(torch.utils.data.Dataset): 
    def __init__(self, root_dir, transform, mode, num_class, pred=[], probability=[], log=''): 
        self.root = os.path.join(root_dir, 'webvision') + '/'
        self.transform = transform
        self.mode = mode  
     
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(os.path.join(self.root, 'val_images_256', img))
                    self.val_labels.append(target)                            
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(os.path.join(self.root, img))
                    self.train_labels.append(target)          
            self.train_imgs = train_imgs       
                    
    def __getitem__(self, index):
        if self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target, index