import logging
import math

import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data

from augmentation import RandAugment
import random
import copy

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
aff_mean = (0.5863, 0.4595, 0.4030)
aff_std = (0.2715, 0.2424, 0.2366)

def get_data(args):
    transform_labeled = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    path_meta_bal = '../data/labeled_10_balance.csv'
    path_meta_train = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_balance = pd.read_csv(path_meta_bal)
    DF_balance = DF_balance.loc[DF_balance['expression'] < 7]

    DF = pd.read_csv(path_meta_train)
    DFselect = DF.loc[DF['expression'] < 7]

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 7]

    label_dataset = AfData(data=DF_balance, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DFselect, directory=orginal_dataset_dir,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset


def get_raf_data(args):
    transform_labeled = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])

    path_label='../data/raf_train.csv'
    path_unlabel='../data/raf_aff_train.csv'
    path_test='../data/raf_test.csv'
    DF_label=pd.read_csv(path_label)
    DF_unlabel=pd.read_csv(path_unlabel)
    DF_test=pd.read_csv(path_test)
    label_dataset_dir='/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images'
    unlabel_dataset_dir='/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images'
    test_dataset_dir='/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images'
    label_dataset = RafData(data=DF_label, directory=label_dataset_dir, transform=transform_labeled)
    unlabel_dataset = RafData(data=DF_unlabel, directory=unlabel_dataset_dir,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = RafData(data=DF_test, directory=test_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset


class AfData(data.Dataset):
    def __init__(self, data, directory, transform):
        self.data = data
        self.directory = directory
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        path = os.path.join(self.directory, self.data.iloc[idx]['subDirectory_filePath'])
        x,y,w,h = self.data.iloc[idx]['face_x'],self.data.iloc[idx]['face_y'],self.data.iloc[idx]['face_width'],self.data.iloc[idx]['face_height']
        target = self.data.iloc[idx]['expression']
        image = Image.open(path).convert('RGB')
        #image = cv2.imread(path, cv2.COLOR_BGR2RGB)
        cropped = image.crop((x, y, x+w, y+h))
        img = self.transform(cropped)
        return img, target

class RafData(data.Dataset):
    def __init__(self,data,directory,transform):
        self.data = data
        self.directory = directory
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        path = os.path.join(self.directory, self.data.iloc[idx]['subDirectory_filePath'])   
        target = self.data.iloc[idx]['expression']
        image = Image.open(path).convert('RGB')
        img = self.transform(image)
        return img,target

class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 10, 10  # default

        self.ori = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(args.resize),
            transforms.Resize(256),
            transforms.RandomCrop(args.resize),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=args.resize,
            #               padding=int(args.resize*0.125),
            #                 padding_mode='reflect')
#             transforms.CenterCrop(args.resize)
        ])
        self.aug = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(args.resize),
            transforms.Resize(256),
            transforms.RandomCrop(args.resize),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=args.resize,
            #                       padding=int(args.resize*0.125),
            #                      padding_mode='reflect'),
#             transforms.CenterCrop(args.resize),
            RandAugment(n=n, m=m)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


DATASET_GETTERS = {'get_data': get_data,
             'get_raf_data': get_raf_data}
