from copy import deepcopy
import random

import torch
from torch import nn
from torchvision import transforms, utils, datasets

from RandAugment.preprocessors import StandardCIFARPreprocessor, ImagePreprocessor
from RandAugment import google_augmentations
from RandAugment.common import HWCByteTensorToPILImage, PILImageToHWCByteTensor, sigmax

class Augmenter(nn.Module):
    def __init__(self, hidden_dimension, dataset_info):
        super().__init__()
        self.conv1 = nn.Conv2d(6,hidden_dimension,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(hidden_dimension,3,kernel_size=3,padding=1)
        self.alpha = nn.Parameter(torch.zeros([]), requires_grad=True)
        self.bn = nn.BatchNorm2d(3, affine=False)
        self.dataset_info = dataset_info


    def forward(self, imgs): # tensor of standard preprocessed images
        random_noise = torch.zeros_like(imgs) #torch.normal(0.,1.,imgs.shape)
        hidden = self.conv1(torch.cat([imgs,random_noise],1))
        activated = torch.tanh(hidden)
        out = self.conv2(activated)
        return self.bn(out*self.alpha+imgs)

    def log(self, writer, t):
        pass

def denormalize(img, mean, std):
    mean, std = torch.tensor(mean).to(img.device), torch.tensor(std).to(img.device)
    return img.mul_(std[:,None,None]).add_(mean[:,None,None])

def normalize(img, mean, std):
    mean, std = torch.tensor(mean).to(img.device), torch.tensor(std).to(img.device)
    return img.sub_(mean[:,None,None]).div_(std[:,None,None])

class RestrictedAugmenter(nn.Module):
    def __init__(self, dataset_info):
        super().__init__()
        self.weight_tranform = lambda w: torch.softmax(w,0)
        num_augs = google_augmentations.num_augmentations()
        weights = torch.zeros(num_augs)
        weights[0] = 1. # at init this leads to behaviour like no augmentation was used
        self.weights = nn.Parameter(weights)
        self.dataset_info = dataset_info


    def forward(self, imgs):
        imgs = [transforms.ToPILImage()(denormalize(img, self.dataset_info['mean'], self.dataset_info['std'])) for img in imgs]
        rg = lambda: random.randint(0,google_augmentations.PARAMETER_MAX)
        augmented_images = torch.stack([torch.stack([transforms.ToTensor()(google_augmentations.apply_augmentation(aug_idx, rg(), img)) for aug_idx in range(google_augmentations.num_augmentations())]) for img in imgs])
        augmented_images = augmented_images * self.weight_tranform(self.weights)[:,None,None,None]
        augmented_images = augmented_images.sum(1)
        return normalize(augmented_images, self.dataset_info['mean'], self.dataset_info['std'])

    def log(self, writer, t):
        for i, w in enumerate(self.weight_tranform(self.weights)):
            writer.add_scalar(f'AugProb/weightofaug{i}', w, t)


class DifferentiableLearnedPreprocessor(ImagePreprocessor):
    def __init__(self, dataset_info, hidden_dimension, optimizer_creator, cutout=0,
                 uniaug_val=False, old_preprocessor_val=False,
                 summary_prefix='', **kwargs):
        super().__init__(**kwargs)
        print('using learned preprocessor')

        if hidden_dimension == 0:
            self.augmenter = RestrictedAugmenter(dataset_info)
        else:
            self.augmenter = Augmenter(hidden_dimension,dataset_info)
        self.standard_cifar_preprocessor = StandardCIFARPreprocessor(dataset_info, cutout=cutout,
                                                                     device=self.device)

        self.optimizer = optimizer_creator(self.augmenter.parameters())
        self.uniaug_val = uniaug_val
        self.old_preprocessor_val = old_preprocessor_val
        if self.old_preprocessor_val:
            self.old_preprossors = [deepcopy(self.augmentation_sampler)]
            self.num_model_updates = 0
        self.summary_prefix = summary_prefix

    def val_augmentations(self, imgs):
        assert sum([self.uniaug_val, self.old_preprocessor_val]) <= 1

        if not imgs:
            return []

        if self.uniaug_val:
            def uni_aug_augmentations():
                num_transf = self.possible_num_sequential_transforms[random.randint(0,len(self.possible_num_sequential_transforms)-1)]
                transforms = []
                for t_i in range(num_transf):
                    transforms.append((random.randint(0,self.num_transforms-1), random.randint(0,google_augmentations.PARAMETER_MAX)))
                return transforms
            return [uni_aug_augmentations() for i in imgs]
        elif self.old_preprocessor_val:
            with torch.no_grad():
                sampled_op_idxs, sampled_scales = self.old_preprossors[random.randint(0,len(self.old_preprossors)-1)](imgs)
                return [[(i, s) for i, s in zip(i_s, s_s)] for i_s, s_s in zip(sampled_op_idxs, sampled_scales)]
        else:
            return [[(0,1.)] for i in imgs]


    def forward(self, imgs, step, validation_step=False, labels=None):
        if not hasattr(self, 'comparison_images'):
            self.comparison_images = imgs[:10]
        self.t = step
        if self.training:
            if validation_step:
                activated_transforms_for_batch = self.val_augmentations(imgs) # [self.val_augmentations() for _ in range(self.bs)]
            else:
                imgs = self.standard_cifar_preprocessor(imgs, step)
                t_imgs = self.augmenter(imgs)
                self.write_summary(step)
                return t_imgs
        else:
            activated_transforms_for_batch = [[(0, 1.)] for i in imgs]  # do not apply any augmentation when evaluating
        google_augmentations.blend_images = imgs
        t_imgs = []
        for i, (img, augs) in enumerate(zip(imgs, activated_transforms_for_batch)):
            for op, scale in augs:
                img = google_augmentations.apply_augmentation(op, scale, img)
            t_imgs.append(img)
        return self.standard_cifar_preprocessor(t_imgs, step)

    def compute_weights(self, rewards):
        if self.normalize_reward:
            rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)
        return rewards

    def step(self, reward):
        assert reward is None, reward
        torch.nn.utils.clip_grad_value_(self.augmenter.parameters(), 5.)
        self.optimizer.step()
        self.augmenter.zero_grad()
        if self.old_preprocessor_val:
            self.num_model_updates += 1
            log2_num_model_updates = self.num_model_updates.bit_length() - 1
            log2_max_num_saved_augmenters = 8 # 8 is log2(256)
            last_2power_of_256 = 0 if log2_num_model_updates < log2_max_num_saved_augmenters else 2**log2_num_model_updates
            if self.num_model_updates == last_2power_of_256:
                self.old_preprossors = [p for i, p in enumerate(self.old_preprossors) if i % 2 == 0]
                assert len(self.old_preprossors) == 2**(log2_max_num_saved_augmenters-1), f'{len(self.old_preprossors)}'
            offset = 2**(max(log2_num_model_updates - log2_max_num_saved_augmenters + 1, 0))
            if self.num_model_updates % offset == 0:
                self.old_preprossors.append(deepcopy(self.augmenter))

    def reset_state(self):
        pass

    def write_summary(self, step):
        if step % 101 == 0 and self.summary_writer is not None and self.training:
            self.augmenter.log(self.summary_writer, step)
            with torch.no_grad():
                imgs = self.standard_cifar_preprocessor(self.comparison_images, step)
                t_imgs = denormalize(self.augmenter(imgs), self.augmenter.dataset_info['mean'], self.augmenter.dataset_info['std'])
                self.summary_writer.add_image('augmented_images', utils.make_grid(t_imgs), step)

