import logging
import os
import random

import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import ConcatDataset, Subset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C

from RandAugment.augmentations import *
from RandAugment.common import get_logger, RoundRobinDataLoader, copy_and_replace_transform
from RandAugment.dataset.noised_cifar10 import NoisedCIFAR10, TargetNoisedCIFAR10

from RandAugment.augmentations import Lighting
from RandAugment.adaptive_loader import AdaptiveLoaderByLabel

logger = get_logger('RandAugment')
logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def get_dataloaders(dataset, batch, dataroot, split=0.15, split_idx=0, get_meta_optimizer_factory=None, distributed=False, summary_writer=None):
    dataset_info = {}
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        dataset_info['mean'] = _CIFAR_MEAN
        dataset_info['std'] = _CIFAR_STD
        dataset_info['img_dims'] = (3,32,32)
    elif 'imagenet' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    logger.debug('augmentation: %s' % C.get()['aug'])
    if C.get()['aug'] == 'randaugment':
        assert not C.get()['randaug'].get('corrected_sample_space') and not C.get()['randaug'].get('google_augmentations')
        transform_train.transforms.insert(0, get_randaugment(C.get()['randaug']['N'], C.get()['randaug']['M'], C.get()['batch']))
    elif C.get()['aug'] in ['default', 'inception', 'inception320']:
        pass
    else:
        raise ValueError('not found augmentations. %s' % C.get()['aug'])

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if 'preprocessor' in C.get():
        print("Not using any transforms in dataset, since preprocessor is active.")
        transform_train = transforms.ToTensor()
        transform_test = transforms.ToTensor()

    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'noised_cifar10':
        total_trainset = NoisedCIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'targetnoised_cifar10':
        total_trainset = TargetNoisedCIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
        extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_train)
        total_trainset = ConcatDataset([trainset, extraset])
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'imagenet':
        total_trainset = torchvision.datasets.ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform_train)
        testset = torchvision.datasets.ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    if 'throwaway_share_of_ds' in C.get():
        share = C.get()['throwaway_share_of_ds']
        indices = list(range(len(total_trainset)))
        random.shuffle(indices)
        train_subset_inds = indices[:int((1.-share) * len(total_trainset))]
        total_trainset = Subset(total_trainset, train_subset_inds)

    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    if distributed:
        assert split == 0.0, "Split not supported for distributed training."
        train_sampler = DistributedSampler(total_trainset)
        test_sampler = DistributedSampler(testset, shuffle=False)
        test_train_sampler = DistributedSampler(total_trainset, shuffle=False)
    else:
        test_sampler = None
        test_train_sampler = None

    if 'adaptive_trainloader' in C.get():
        val_bs = C.get().get('val_batch',0)
        trainloader = AdaptiveLoaderByLabel(total_trainset, get_meta_optimizer_factory(), batch-val_bs, val_bs, summary_writer=summary_writer)
    elif 'val_step_trainloader_val_share' in C.get():
        share = C.get()['val_step_trainloader_val_share']
        indices = list(range(len(total_trainset)))
        random.shuffle(indices)
        val_subset_inds, train_subset_inds = indices[:int(share*len(total_trainset))], indices[int(share*len(total_trainset)):]
        val_ds, train_ds = Subset(total_trainset, val_subset_inds), Subset(total_trainset, train_subset_inds)
        if distributed:
            tra_sampler, val_sampler = DistributedSampler(train_ds), DistributedSampler(val_ds)
        else:
            tra_sampler = val_sampler = None
        trainloader = RoundRobinDataLoader(
            torch.utils.data.DataLoader(
                train_ds, batch_size=batch, shuffle=tra_sampler is None, num_workers=1 if distributed else 32,
                pin_memory=True,
                sampler=tra_sampler, drop_last=True),
            torch.utils.data.DataLoader(
                val_ds, batch_size=batch, shuffle=val_sampler is None, num_workers=1 if distributed else 32,
                pin_memory=True,
                sampler=val_sampler, drop_last=True)
        )
        if distributed:
            # will be used to step later
            class SamplerWrapper:
                def __init__(self, *samplers):
                    self.samplers = samplers

                def set_epoch(self, *args, **kwargs):
                    for s in self.samplers:
                        s.set_epoch(*args, **kwargs)
            train_sampler = SamplerWrapper(tra_sampler, val_sampler)
    elif 'different_trainloader_for_val' in C.get():
        trainloader = RoundRobinDataLoader(
            torch.utils.data.DataLoader(
                total_trainset, batch_size=batch, shuffle=train_sampler is None, num_workers=1 if distributed else 32,
                pin_memory=True,
                sampler=train_sampler, drop_last=True),
            torch.utils.data.DataLoader(
                total_trainset, batch_size=batch, shuffle=train_sampler is None, num_workers=1 if distributed else 32,
                pin_memory=True,
                sampler=train_sampler, drop_last=True)
        )

    else:
        trainloader = torch.utils.data.DataLoader(
            total_trainset, batch_size=batch, shuffle=train_sampler is None, num_workers=1 if distributed else 32, pin_memory=True,
            sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=1 if distributed else 16, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=1 if distributed else 32, pin_memory=True,
        drop_last=False, sampler=test_sampler
    )
    # We use this 'hacky' solution s.t. we do not need to keep the dataset twice in memory.
    test_total_trainset = copy_and_replace_transform(total_trainset, transform_test)
    test_trainloader = torch.utils.data.DataLoader(
        test_total_trainset, batch_size=batch, shuffle=False, num_workers=1 if distributed else 32, pin_memory=True,
        drop_last=False, sampler=test_train_sampler
    )
    return train_sampler, trainloader, validloader, testloader, test_trainloader, dataset_info


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
