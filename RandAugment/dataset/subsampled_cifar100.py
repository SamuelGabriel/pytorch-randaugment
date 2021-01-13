import random

from torchvision.datasets.cifar import CIFAR100, Image
from torch.utils.data import Subset
import numpy as np

from RandAugment.common import stratified_split

def get_tenclass_CIFAR100_trainandval(*args, **kwargs):
    cifar100 = CIFAR100(*args, **kwargs)
    inds_of_examples_from_first_ten_classes = [i for i,t in enumerate(cifar100.targets) if t < 10]
    val_cifar100 = CIFAR100(*args, train=False, **kwargs)
    val_inds_of_examples_from_first_ten_classes = [i for i,t in enumerate(val_cifar100.targets) if t < 10]
    return Subset(cifar100, inds_of_examples_from_first_ten_classes), \
           Subset(val_cifar100, val_inds_of_examples_from_first_ten_classes)

def get_fiftyexample_CIFAR100_trainandval(*args, **kwargs):
    cifar100 = CIFAR100(*args, **kwargs)
    _, inds_for_examples_from_subsample = stratified_split(cifar100.targets, .1)
    return Subset(cifar100, inds_for_examples_from_subsample), CIFAR100(*args, train=False, **kwargs)


