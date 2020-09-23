import random

from torchvision.datasets.cifar import CIFAR10, Image
import numpy as np

def get_white_noise_image():
    pil_map = Image.fromarray(np.random.randint(0,255,(32,32,3),dtype=np.dtype('uint8')))
    return pil_map

class NoisedCIFAR10(CIFAR10):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.extra_labels = np.random.choice(10,len(self))
        self.noise_labels = (0,)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        extra_label = self.extra_labels[index]
        if extra_label in self.noise_labels:
            img = get_white_noise_image()
            if self.transform is not None:
                img = self.transform(img)

        return img, target, extra_label

class TargetNoisedCIFAR10(CIFAR10):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.extra_labels = np.random.choice(10,len(self))
        self.noise_labels = (0,)
        self.noisy_targets = np.random.choice(10,len(self))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        extra_label = self.extra_labels[index]
        if extra_label in self.noise_labels:
            target = self.noisy_targets[index] #

        return img, target, extra_label
