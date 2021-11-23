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
from PIL import Image
import random
import copy

logger = logging.getLogger(__name__)

normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
aff_mean = (0.5863, 0.4595, 0.4030)
aff_std = (0.2715, 0.2424, 0.2366)


def get_10_balance(args):
    transform_labeled = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        #         transforms.RandomCrop(size=args.resize,
        #                               padding=int(args.resize*0.125),
        #                               padding_mode='reflect'),
        #         transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    path_meta_bal = '/home/cseadmin/MPL/labeled_10_balance.csv'
    path_meta_train = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_balance = pd.read_csv(path_meta_bal)

    DF = pd.read_csv(path_meta_train)
    DFselect = DF.loc[DF['expression'] < 8]

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_balance, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DFselect, directory=orginal_dataset_dir,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_10_balance_7(args):
    transform_labeled = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.resize),
        transforms.RandomHorizontalFlip(),

        #         transforms.RandomCrop(size=args.resize,
        #                               padding=int(args.resize*0.125),
        #                               padding_mode='reflect'),
        #         transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    path_meta_bal = '/home/cseadmin/MPL/labeled_10_balance.csv'
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


DATASET_GETTERS = {
                   'bal': get_10_balance,
                   'bal_7': get_10_balance_7,                   
                   }
