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

def get_mix(args):
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
    path_meta_train = '/home/cseadmin/MPL/mix_pl.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF = pd.read_csv(path_meta_train)
    # DFselect = DF.loc[DF['expression'] < 8]

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]
    label_dataset = AfData_pl(data=DF, directory=orginal_dataset_dir, transform=transform_labeled)
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, test_dataset

def get_mix_bal(args):
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
    path_meta_train = '/home/cseadmin/MPL/mix_pl_bal.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF = pd.read_csv(path_meta_train)
    # DFselect = DF.loc[DF['expression'] < 8]

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]
    label_dataset = AfData_pl(data=DF, directory=orginal_dataset_dir, transform=transform_labeled)
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, test_dataset

def get_10_imbalance(args):
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
    path_meta_imb = '/home/cseadmin/MPL/label_10_imb.csv'
    path_meta_train = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF = pd.read_csv(path_meta_train)
    DFselect = DF.loc[DF['expression'] < 8]

    DF_imb = pd.read_csv(path_meta_imb)

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_imb, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DFselect, directory=orginal_dataset_dir,transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def pseudo_label(args):
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
        transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    path_meta_bal = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    #path_meta_train = '/home/cseadmin/MPL/label_10_bal_rest_clean.csv'
    path_meta_train = '/home/data/lzy/WebFace260M_full/webface260m_bal.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_balance = pd.read_csv(path_meta_bal)

    DF = pd.read_csv(path_meta_train)
    
    #DFselect = DF.loc[DF['expression'] < 8]
    DFselect = DF
    #DFselect = DF.sample(n=257532, random_state=1, axis=0).reset_index(drop=True)

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_balance, directory=orginal_dataset_dir, transform=transform_labeled)
    
    unlabel_dataset = WebData(data=DFselect, directory='/home/data/lzy',
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    '''
    unlabel_dataset = AfData(data=DFselect, directory=orginal_dataset_dir,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    '''
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset


def pseudo_label_7(args):
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
        transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    path_meta_bal = '/home/cseadmin/MPL/labeled_10_balance.csv'
    path_meta_train = '/home/cseadmin/MPL/label_10_bal_rest.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_balance = pd.read_csv(path_meta_bal)
    DF_balance =DF_balance.loc[DF_balance['expression'] < 7]

    DF = pd.read_csv(path_meta_train)
    DFselect = DF.loc[DF['expression'] < 7]

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 7]

    label_dataset = AfData(data=DF_balance, directory=orginal_dataset_dir, transform=transform_labeled)

    # unlabel_dataset = WebData(data=DFselect, directory='/home/data/lzy',
    #                           transform=TransformMPL(args, mean=aff_mean, std=aff_std))

    unlabel_dataset = AfData(data=DFselect, directory=orginal_dataset_dir,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))

    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

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

def get_10_balance_clean(args):
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
    path_meta_bal = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    path_meta_train = '/home/cseadmin/MPL/aligned_affectnet_train.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
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

def get_10_balance_clean_origin(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(args.resize),
        #         transforms.RandomCrop(size=args.resize,
        #                               padding=int(args.resize*0.125),
        #                               padding_mode='reflect'),
        #         transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    path_meta_bal = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    path_meta_train = '/home/cseadmin/MPL/aligned_affectnet_train.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
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

def get_10_balance_clean_origin_2(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(args.resize),
        #         transforms.RandomCrop(size=args.resize,
        #                               padding=int(args.resize*0.125),
        #                               padding_mode='reflect'),
        #         transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    path_meta_bal = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    path_meta_train = '/home/cseadmin/MPL/aligned_affectnet_train.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_balance = pd.read_csv(path_meta_bal)

    DF = pd.read_csv(path_meta_train)
    DFselect = DF.loc[DF['expression'] < 8]

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_balance, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DFselect, directory=orginal_dataset_dir,
                             transform=TransformMPL_origin(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_10_balance_and_rest_clean(args):
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
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    path_meta_bal = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    path_meta_rest = '/home/cseadmin/MPL/label_10_bal_rest_clean.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_balance = pd.read_csv(path_meta_bal)
    DF = pd.read_csv(path_meta_rest)
    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_balance, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DF, directory=orginal_dataset_dir,
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

def get_10_balance_and_rest_7(args):
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
    path_meta_train = '/home/cseadmin/MPL/label_10_bal_rest.csv'
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

def get_10_balance_and_rest(args):
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
    path_meta_rest = '/home/cseadmin/MPL/label_10_bal_rest.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_balance = pd.read_csv(path_meta_bal)
    DF = pd.read_csv(path_meta_rest)
    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_balance, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DF, directory=orginal_dataset_dir,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_5_balance_clean(args):
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
    path_meta_bal = '/home/cseadmin/MPL/labeled_5_balance_clean.csv'
    path_meta_train = '/home/cseadmin/MPL/aligned_affectnet_train.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
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

def get_5_balance_clean_7(args):
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
    path_meta_bal = '/home/cseadmin/MPL/labeled_5_balance_clean.csv'
    path_meta_train = '/home/cseadmin/MPL/aligned_affectnet_train.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
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

def get_1_balance_clean(args):
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
    path_meta_bal = '/home/cseadmin/MPL/labeled_1_balance_clean.csv'
    path_meta_train = '/home/cseadmin/MPL/aligned_affectnet_train.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
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

def get_1_balance_clean_7(args):
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
    path_meta_bal = '/home/cseadmin/MPL/labeled_1_balance_clean.csv'
    path_meta_train = '/home/cseadmin/MPL/aligned_affectnet_train.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
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

def get_toy_example(args):
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
    path_meta_toy_10 = '/home/cseadmin/MPL/toy_10.csv'
    path_meta_toy_90 = '/home/cseadmin/MPL/toy_90.csv'
    # path_meta_train = '../AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_10 = pd.read_csv(path_meta_toy_10)
    DF_90 = pd.read_csv(path_meta_toy_90)
    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]
    label_dataset = AfData(data=DF_10, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DF_90, directory=orginal_dataset_dir, transform=TransformMPL_n(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_toy_val(args):
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
    path_meta_toy_10 = '/home/cseadmin/MPL/toy_10.csv'
    path_meta_toy_90 = '/home/cseadmin/MPL/toy_90.csv'
    # path_meta_train = '../AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_10 = pd.read_csv(path_meta_toy_10)
    DF_90 = pd.read_csv(path_meta_toy_90)
    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]
    label_dataset = AfData(data=DF_10, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DF_90, directory=orginal_dataset_dir,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_toy_val_off(args):
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
    path_meta_toy_10 = '/home/cseadmin/MPL/toy_10.csv'
    path_meta_toy_90 = '/home/cseadmin/MPL/toy_90.csv'
    # path_meta_train = '../AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_10 = pd.read_csv(path_meta_toy_10)
    DF_90 = pd.read_csv(path_meta_toy_90)
    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]
    label_dataset = AfData(data=DF_10, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DF_90, directory=orginal_dataset_dir,
                             transform=TransformMPL_off(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_all(args):
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

    path_meta_unlabel = '/home/data/lzy/WebFace260M_full/webface260m_full.csv'
    path_meta_label = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'
    webface_dir = '/home/data/lzy'

    DF_label = pd.read_csv(path_meta_label)
    DF_label = DF_label.loc[DF_label['expression'] < 8]

    DF_unlabel = pd.read_csv(path_meta_unlabel)

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]
    label_dataset = AfData(data=DF_label, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = WebData(data=DF_unlabel, directory=webface_dir,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_simple(args):
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

    path_meta_unlabel = '/home/data/lzy/WebFace260M_full/webface260m_full.csv'
    path_meta_aff_10 = '/home/cseadmin/MPL/labeled_10_balance.csv'
    path_meta_aff_100 = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'
    webface_dir = '/home/data/lzy'

    aff_10 = pd.read_csv(path_meta_aff_10)

    aff_100 = pd.read_csv(path_meta_aff_100)
    aff_100 = aff_100.loc[aff_100['expression'] < 8]

    webface = pd.read_csv(path_meta_unlabel)

    DF_unlabel = pd.concat([aff_100, webface]).reset_index(drop=True)

    DF_unlabel = DF_unlabel.sample(frac=1, random_state=1).reset_index(drop=True)

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=aff_10, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = mixData(data=DF_unlabel, directory1=orginal_dataset_dir, directory2=webface_dir,
                              transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_affectnet(args):
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

    path_meta_label = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF = pd.read_csv(path_meta_label)
    DF = DF.loc[DF['expression'] < 8]

    DF2 = pd.read_csv(path_meta_val)
    DF2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF, directory=orginal_dataset_dir, transform=transform_labeled)
    test_dataset = AfData(data=DF2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, test_dataset

def get_affectnet_7(args):
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

    path_meta_label = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/training.csv'
    path_meta_val = '/home/data/lzy/AffectNet/Manually_Annotated_file_lists/validation.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF = pd.read_csv(path_meta_label)
    DF = DF.loc[DF['expression'] < 7]

    DF2 = pd.read_csv(path_meta_val)
    DF2 = DF2.loc[DF2['expression'] < 7]

    label_dataset = AfData(data=DF, directory=orginal_dataset_dir, transform=transform_labeled)
    test_dataset = AfData(data=DF2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, test_dataset

def get_balance_supplement(args):
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

    path_label_balance = '/home/cseadmin/MPL/mix_label_balance.csv'
    path_meta_unlabel = '/home/cseadmin/MPL/aligned_affectnet_train.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'
    webface_dir = '/home/data/lzy'

    DF_bal = pd.read_csv(path_label_balance)

    DF_unlabel = pd.read_csv(path_meta_unlabel)

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = mixBalData(data=DF_bal, directory1=orginal_dataset_dir, directory2=webface_dir, transform=transform_labeled)
    unlabel_dataset = AfData(data=DF_unlabel, directory=orginal_dataset_dir,
                              transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_balance_expansion(args):
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

    path_label_balance = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    path_meta_unlabel = '/home/cseadmin/MPL/unlabeled_expansion.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'
    webface_dir = '/home/data/lzy'

    DF_bal = pd.read_csv(path_label_balance)

    DF_unlabel = pd.read_csv(path_meta_unlabel)

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_bal, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = mixBalData(data=DF_unlabel, directory1=orginal_dataset_dir, directory2=webface_dir,
                              transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_balance_expansion_2(args):
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

    path_label_balance = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    path_meta_unlabel = '/home/cseadmin/MPL/unlabeled_expansion_2.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'
    webface_dir = '/home/data/lzy'

    DF_bal = pd.read_csv(path_label_balance)

    DF_unlabel = pd.read_csv(path_meta_unlabel)

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_bal, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = mixBalData(data=DF_unlabel, directory1=orginal_dataset_dir, directory2=webface_dir,
                              transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_balance_expansion_3(args):
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

    path_label_balance = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    path_meta_unlabel = '/home/cseadmin/MPL/unlabeled_expansion_57k.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'
    webface_dir = '/home/data/lzy'

    DF_bal = pd.read_csv(path_label_balance)

    DF_unlabel = pd.read_csv(path_meta_unlabel)

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_bal, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = mixBalData(data=DF_unlabel, directory1=orginal_dataset_dir, directory2=webface_dir,
                              transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_balance_expansion_rank(args):
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

    path_label_balance = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    path_meta_unlabel = '/home/cseadmin/MPL/unlabeled_expansion_rank.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'
    webface_dir = '/home/data/lzy'

    DF_bal = pd.read_csv(path_label_balance)

    DF_unlabel = pd.read_csv(path_meta_unlabel)

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_bal, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = mixBalData(data=DF_unlabel, directory1=orginal_dataset_dir, directory2=webface_dir,
                              transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

def get_rafdb(args):
    transform_labeled = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])

    path_meta_label = '/home/cseadmin/gh/raf_train.csv'
    path_meta_val = '/home/cseadmin/gh/raf_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF = pd.read_csv(path_meta_label)

    DF2 = pd.read_csv(path_meta_val)

    label_dataset = rafData(data=DF, directory=orginal_dataset_dir, transform=transform_labeled)
    test_dataset = rafData(data=DF2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, test_dataset


def get_webface(args):
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
    path_meta_bal = '/home/cseadmin/MPL/labeled_10_balance_clean.csv'
    # path_meta_train = '/home/cseadmin/MPL/label_10_bal_rest_clean.csv'
    path_meta_train = '/home/data/lzy/WebFace260M_full/webface260m_bal.csv'
    path_meta_val = '/home/cseadmin/MPL/aligned_affectnet_test.csv'
    orginal_dataset_dir = '/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images/'

    DF_balance = pd.read_csv(path_meta_bal)

    DF = pd.read_csv(path_meta_train)

    # DFselect = DF.loc[DF['expression'] < 8]
    DFselect = DF

    DF2 = pd.read_csv(path_meta_val)
    DFselect2 = DF2.loc[DF2['expression'] < 8]

    label_dataset = AfData(data=DF_balance, directory=orginal_dataset_dir, transform=transform_labeled)
    unlabel_dataset = WebData(data=DFselect, directory='/home/data/lzy',
                              transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = AfData(data=DFselect2, directory=orginal_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset

class rafData(data.Dataset):
    def __init__(self, data, directory, transform):
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
        return img, target

class WebData(data.Dataset):
    def __init__(self, data, directory, transform):
        self.data = data
        self.directory = directory
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # path = os.path.join(self.directory, self.data.iloc[idx]['0'])
        path = self.directory + self.data.iloc[idx]['0']
        # path = path.replace('\\', '/')
        image = Image.open(path).convert('RGB')
        img = self.transform(image)
        return img

class mixData(data.Dataset):
    def __init__(self, data, directory1, directory2, transform):
        self.data = data
        self.directory1 = directory1
        self.directory2 = directory2
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if not self.data.iloc[idx]['subDirectory_filePath'] is np.nan:
            path = os.path.join(self.directory1, self.data.iloc[idx]['subDirectory_filePath'])
            x, y, w, h = self.data.iloc[idx]['face_x'], self.data.iloc[idx]['face_y'], self.data.iloc[idx][
                'face_width'], self.data.iloc[idx]['face_height']
            image = Image.open(path).convert('RGB')
            # image = cv2.imread(path, cv2.COLOR_BGR2RGB)
            cropped = image.crop((x, y, x + w, y + h))
            img = self.transform(cropped)
            return img
        else:
            path = self.directory2 + self.data.iloc[idx]['0']
            image = Image.open(path).convert('RGB')
            img = self.transform(image)
            return img

class mixBalData(data.Dataset):
    def __init__(self, data, directory1, directory2, transform):
        self.data = data
        self.directory1 = directory1
        self.directory2 = directory2
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if pd.isnull(self.data.iloc[idx]['probability']):
            path = os.path.join(self.directory1, self.data.iloc[idx]['subDirectory_filePath'])
            x, y, w, h = self.data.iloc[idx]['face_x'], self.data.iloc[idx]['face_y'], self.data.iloc[idx][
                'face_width'], self.data.iloc[idx]['face_height']
            image = Image.open(path).convert('RGB')
            target = self.data.iloc[idx]['expression']
            # image = cv2.imread(path, cv2.COLOR_BGR2RGB)
            cropped = image.crop((x, y, x + w, y + h))
            img = self.transform(cropped)
            return img, target
        else:
            path = self.directory2 + self.data.iloc[idx]['subDirectory_filePath']
            image = Image.open(path).convert('RGB')
            img = self.transform(image)
            target = self.data.iloc[idx]['expression']
            return img, target

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

class AfData_pl(data.Dataset):
    def __init__(self,data,directory, transform):
        self.data = data
        self.directory = directory
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        path = os.path.join(self.directory, self.data.iloc[idx]['subDirectory_filePath'])
        x, y, w, h = self.data.iloc[idx]['face_x'], self.data.iloc[idx]['face_y'], self.data.iloc[idx]['face_width'], self.data.iloc[idx]['face_height']
        target = self.data.iloc[idx]['expression']
        pseudo_label = self.data.iloc[idx]['p_label']
        image = Image.open(path).convert('RGB')
        cropped = image.crop((x, y, x+w, y+h))
        img = self.transform(cropped)
        if pseudo_label == -1:
            target = target
        else:
            target = pseudo_label
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

class TransformMPL_origin(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 10, 10  # default

        self.ori = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(args.resize),
            transforms.Resize(args.resize),
            # transforms.RandomCrop(args.resize),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=args.resize,
            #               padding=int(args.resize*0.125),
            #                 padding_mode='reflect')
#             transforms.CenterCrop(args.resize)
        ])
        self.aug = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(args.resize),
            # transforms.Resize(256),
            # transforms.RandomCrop(args.resize),
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

DATASET_GETTERS = {'ex': get_mix,
                   'ex_mix_bal': get_mix_bal,
                   'imb': get_10_imbalance,
                   'bal': get_10_balance,
                   'toy': get_toy_example,
                   'aug_on': get_toy_val,
                   'aug_off': get_toy_val_off,
                   'all': get_all,
                   'bal_1_9': get_10_balance_and_rest,
                   'bal_7': get_10_balance_7,
                   'bal_7_1_9':get_10_balance_and_rest_7,
                   'simple_all': get_simple,
                   'affectnet': get_affectnet,
                   'bal_clean': get_10_balance_clean,
	               'pseudo_label': pseudo_label,
                   'bal_supplement': get_balance_supplement,
                   'bal_expansion': get_balance_expansion,
                   'raf_db': get_rafdb,
                   'bal_clean_1_9': get_10_balance_and_rest_clean,
                   'bal_expansion_2': get_balance_expansion_2,
                   'bal_expansion_3': get_balance_expansion_3,
                   'webface': get_webface,
                   'bal_clean_origin': get_10_balance_clean_origin,
                   'bal_clean_origin_2': get_10_balance_clean_origin_2,
                   'pseudo_label_7': pseudo_label_7,
                   'bal_5_clean': get_5_balance_clean,
                   'bal_1_clean': get_1_balance_clean,
                   'bal_expansion_rank': get_balance_expansion_rank,
                   'bal_5_clean_7': get_5_balance_clean_7,
                   'bal_1_clean_7': get_1_balance_clean_7,
                   'affectnet_7': get_affectnet_7,
                   }
