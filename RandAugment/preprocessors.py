import torch
import torchvision
from torch import nn
from time import time
from PIL import Image
import multiprocessing
from abc import abstractmethod, ABCMeta
from torchvision import transforms
from RandAugment import google_augmentations, augmentations
from RandAugment.networks.convnet import SeqConvNet
from RandAugment.common import sigmax, log_sigmax, exploresoftmax, log_exploresoftmax, replace_parameters, recursive_backpack_memory_cleanup, ListDataLoader, le_softmax, log_le_softmax
from RandAugment.data import _IMAGENET_PCA
from RandAugment.augmentations import Lighting
from theconf import Config as C
from RandAugment.preprocessor_samplers import *

from copy import deepcopy
import random

class Preprocessor(nn.Module):
    def __init__(self, device='cpu:0', summary_writer=None):
        super().__init__()
        self.summary_writer = summary_writer
        self.device = device

    def has_parameters(self):
        p = self.parameters()
        try:
            first = next(p)
            return True
        except StopIteration:
            return False

class ImagePreprocessor(Preprocessor, metaclass=ABCMeta):
    """Objects of this class can even hold weights that are trained with the meta loss.
    """
    @abstractmethod
    def forward(self, imgs):
        """Receives a batch of PIL img and returns a tensor with bs x c x h x w format.
        Depending on the `training` flag, should either perform the transformation
        for training or test.
        The test transform should _not be stochastic_.

        Args:
            imgs (List[PIL]): the image as it is in the data set
        Returns:
            Float Tensor bs x c x h x w
        """
        pass

def compute_dataset_stats(ds):
    mean = 0.
    std = 0.
    for img,label in ds:
        t = torchvision.transforms.ToTensor()(img)
        if len(t.shape) == 2:
            t.unsqueeze_(0)
        data = t.view(t.shape[0], -1)
        mean += data.mean(1)
        std += data.std(1)
    return mean / float(len(ds)), std / float(len(ds))


class StandardCIFARPreprocessor(ImagePreprocessor):
    def __init__(self, dataset_info, cutout, **kwargs):
        super().__init__(**kwargs)
        img_dims = dataset_info['img_dims']
        channels = img_dims[0]
        image_size_hw = tuple(img_dims[1:])

        self.normalize = transforms.Normalize(dataset_info['mean'], dataset_info['std'])
        self.rand_crop = transforms.RandomCrop(image_size_hw, padding=(4, 4), padding_mode='constant')
        self.maybe_mirror = transforms.RandomHorizontalFlip(p=.5)
        if cutout:
            self.eraser = augmentations.CutoutDefault(cutout)
        else:
            self.eraser = lambda x: x

    def forward(self, imgs, step, validation_step=False):
        assert not validation_step
        with torch.no_grad():
            img_dims = torchvision.transforms.ToTensor()(imgs[0]).shape
            batch_tensor = torch.zeros((len(imgs),) + tuple(img_dims))
            for i, img in enumerate(imgs):
                if self.training:
                    img = self.rand_crop(img)
                    img = self.maybe_mirror(img)
                t = torchvision.transforms.ToTensor()(img)
                t = self.normalize(t)
                if self.training:
                    t = self.eraser(t)
                batch_tensor[i] = t

            return batch_tensor.to(self.device)

class StandardImagenetPreprocessor(ImagePreprocessor):
    def __init__(self, cutout, **kwargs):
        super().__init__(**kwargs)
        assert not cutout, "Please do not specify cutout with imagenet, it is not implemented."
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(), # was missing in some experiments
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def forward(self, imgs, step, validation_step=False):
        assert not validation_step
        img_dims = torchvision.transforms.ToTensor()(imgs[0]).shape
        batch_tensor = torch.zeros((len(imgs),) + tuple(img_dims))
        for i, img in enumerate(imgs):
            if self.train:
                t_img = self.transform_train(img)
            else:
                t_img = self.transform_test(img)
            batch_tensor[i] = t_img
        return batch_tensor.to(self.device)

def get_standard_preprocessor(dataset_info, cutout, **kwargs):
    dataset = C.get()['dataset']
    if 'cifar' in dataset:
        return StandardCIFARPreprocessor(dataset_info, cutout, **kwargs)
    elif 'imagenet' in dataset:
        return StandardImagenetPreprocessor(cutout, **kwargs)
    else:
        raise ValueError(f"Dataset {dataset} does not have a standard preprocessor.")


class AugmentationSampler(nn.Module):
    def __init__(self, hidden_dimension, num_transforms, num_scales, q_residual, q_zero_init, scale_embs_zero_init,
                 label_smoothing_rate, usemodel_Dout_imgdims=(False, 10, None), dist_functions=(torch.softmax, torch.log_softmax)):
        super().__init__()
        self.dist, self.log_dist = dist_functions
        self.op_embs = nn.Parameter(torch.normal(0., 1., (num_transforms, hidden_dimension), requires_grad=True))
        if scale_embs_zero_init:
            self.scale_embs = nn.Parameter(torch.zeros(num_scales, hidden_dimension, requires_grad=True))
        else:
            self.scale_embs = nn.Parameter(torch.normal(0., 1., (num_scales, hidden_dimension), requires_grad=True))
        use_model, D_out, img_dims = usemodel_Dout_imgdims
        if use_model:
            self.input_image = nn.Parameter(torch.normal(0., 1., img_dims, requires_grad=True))
            self.post_processor = nn.Sequential(
                nn.Linear(D_out, hidden_dimension),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dimension, hidden_dimension)
            )
            if q_zero_init:
                w = self.post_processor[2].weight
                b = self.post_processor[2].bias
                torch.nn.init.zeros_(w)
                torch.nn.init.zeros_(b)
        else:
            if q_zero_init:
                self.q = nn.Parameter(torch.zeros(hidden_dimension, requires_grad=True))
            else:
                self.q = nn.Parameter(torch.normal(0., 1., hidden_dimension, requires_grad=True))
        self.label_smoothing_rate = label_smoothing_rate
        self.q_residual = q_residual

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad

    def compute_q(self, model=None):
        if model is not None:
            model_trains = model.training
            model.eval()
            o = model(self.input_image.unsqueeze(0))
            if model_trains:
                model.train()
            return self.post_processor(o).squeeze(0)
        else:
            return self.q

    def forward(self, num_samples, model=None):
        self.q = self.compute_q(model)
        self.op_logits = self.op_embs @ self.q
        self.p_op = self.dist(self.op_logits, 0)
        self.log_p_op = self.log_dist(self.op_logits, 0)
        sampled_op_idxs = torch.multinomial(self.p_op, num_samples, replacement=True)
        hidden = self.op_embs[sampled_op_idxs]
        if self.q_residual:
            hidden += self.q
        scale_logits = hidden @ self.scale_embs.t()
        p_scale = self.dist(scale_logits, 1)
        log_p_scale = self.log_dist(scale_logits, 1)
        sampled_scales = torch.multinomial(p_scale, 1, replacement=True).squeeze()
        self.logps = self.log_p_op[sampled_op_idxs] + log_p_scale[torch.arange(log_p_scale.shape[0]), sampled_scales]
        if self.label_smoothing_rate:
            self.logps = (self.log_p_op.mean() * len(sampled_op_idxs) + log_p_scale.mean(1).sum(
                0)) * self.label_smoothing_rate \
                         + self.logps * (1. - self.label_smoothing_rate)
        return sampled_op_idxs.cpu(), sampled_scales.cpu()


class LearnedPreprocessorRandaugmentSpace(ImagePreprocessor):
    def __init__(self, dataset_info, hidden_dimension, optimizer_creator, bs, val_bs, entropy_alpha, scale_entropy_alpha=0.,
                 importance_sampling=False, cutout=0, normalize_reward=True, model_for_online_tests=None,
                 D_out=10, q_zero_init=True, q_residual=False, scale_embs_zero_init=False, scale_embs_zero_strength_bias=0., label_smoothing_rate=0.,
                 sigmax_dist=False, exploresoftmax_dist=False, use_images_for_sampler=False, use_labels_for_sampler=False, uniaug_val=False,  old_preprocessor_val=False, current_preprocessor_val=False,
                 update_every_k_steps=1, **kwargs):
        super().__init__(**kwargs)
        print('using learned preprocessor')
        assert not use_images_for_sampler and not uniaug_val and not old_preprocessor_val and not current_preprocessor_val
        assert scale_embs_zero_strength_bias==0.

        img_dims = dataset_info['img_dims']
        self.num_transforms = google_augmentations.num_augmentations()
        self.num_scales = google_augmentations.PARAMETER_MAX + 1
        self.standard_cifar_preprocessor = get_standard_preprocessor(dataset_info,cutout,device=self.device)
        self.model = model_for_online_tests

        dists = (torch.softmax, torch.log_softmax)
        assert not (sigmax_dist and exploresoftmax_dist)
        if sigmax_dist:
            dists = (sigmax, log_sigmax)
        if exploresoftmax_dist:
            dists = (exploresoftmax, log_exploresoftmax)

        self.augmentation_sampler = AugmentationSampler(hidden_dimension, self.num_transforms, self.num_scales,
                                                        q_residual, q_zero_init, scale_embs_zero_init,
                                                        label_smoothing_rate,
                                                        usemodel_Dout_imgdims=(self.model is not None, D_out, img_dims),
                                                        dist_functions=dists).to(self.device)
        self.agumentation_sampler_copies = []

        self.normalize_reward = normalize_reward
        self.optimizer = optimizer_creator(self.augmentation_sampler.parameters())
        self.entropy_alpha = entropy_alpha
        self.scale_entropy_alpha = scale_entropy_alpha
        self.importance_sampling = importance_sampling
        self.bs = bs
        self.val_bs = val_bs
        self.update_every_k_steps = update_every_k_steps

    def forward(self, imgs, step, validation_step=False):
        assert not validation_step
        self.t = step
        if self.training:
            aug_sampler = deepcopy(self.augmentation_sampler)
            self.agumentation_sampler_copies.append(aug_sampler)
            sampled_op_idxs, sampled_scales = aug_sampler(self.bs, self.model)
            activated_transforms_for_batch = list(zip(sampled_op_idxs, sampled_scales)) + [(0, 1.) for _ in range(self.val_bs)]
            self.write_summary(aug_sampler, step)
        else:
            activated_transforms_for_batch = [(0, 1.) for i in imgs]  # do not apply any augmentation when evaluating
        t_imgs = []
        for i, (img, (op, scale)) in enumerate(zip(imgs, activated_transforms_for_batch)):
            img = google_augmentations.apply_augmentation(op, scale, img)
            t_imgs.append(img)
        if self.importance_sampling and self.training:
            w = 1. / (self.num_transforms * self.num_scales * torch.exp(aug_sampler.logps.detach()))
            return self.standard_cifar_preprocessor(t_imgs, step), w * (len(w) / torch.sum(w))
        return self.standard_cifar_preprocessor(t_imgs, step)

    def compute_weights(self, rewards):
        if self.normalize_reward:
            rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)
        return rewards

    def step(self, rewards):
        assert len(self.agumentation_sampler_copies) <= 2
        aug_sampler = self.agumentation_sampler_copies.pop(0)  # pops the oldest state first (queue-style)
        if self.t % 100 == 0 and self.summary_writer is not None and self.training:
            self.summary_writer.add_scalar(f'Alignment/AverageAlignment', rewards.mean(), self.t)
            self.summary_writer.add_scalar(f'Alignment/MaxAlignment', rewards.max(), self.t)
            self.summary_writer.add_scalar(f'Alignment/MinAlignment', rewards.min(), self.t)

        with torch.no_grad():
            if self.importance_sampling:
                rewards *= 1. / ((self.num_transforms * self.num_scales) ** 2 * torch.exp(2 * self.logps))
            weights = self.compute_weights(rewards).detach().to(self.device)

        aug_sampler.zero_grad()
        loss = -aug_sampler.logps @ weights.detach() / float(len(weights))
        if self.entropy_alpha:
            neg_entropy = aug_sampler.p_op @ (
                        aug_sampler.log_p_op * (aug_sampler.p_op != 0.))  # here we could have logp = -inf and p = 0.
            loss += self.entropy_alpha * neg_entropy
        if self.scale_entropy_alpha:
            hidden = aug_sampler.op_embs
            if aug_sampler.q_residual:
                hidden = aug_sampler.op_embs + aug_sampler.q
            scale_logits = hidden @ aug_sampler.scale_embs.t()
            log_p_scale = aug_sampler.log_dist(scale_logits, 1)
            p_scale = aug_sampler.dist(scale_logits, 1)
            avg_neg_entropy = torch.einsum('bh,bh->b', p_scale, (
                        log_p_scale * (p_scale != 0.))).mean()  # here we could have logp = -inf and p = 0.
            loss += self.scale_entropy_alpha * avg_neg_entropy

        if self.model:
            # This is needed so that we do not write to the gradient buffers of the model
            for p in self.model.parameters():
                assert p.requires_grad
                p.requires_grad = False
        loss.backward()
        if self.model:
            for p in self.model.parameters():
                p.requires_grad = True
        torch.nn.utils.clip_grad_value_(aug_sampler.parameters(), 5.)
        if self.t % self.update_every_k_steps == 0:
            self.augmentation_sampler.add_grad_of_copy(aug_sampler)
            self.optimizer.step()
            self.augmentation_sampler.zero_grad()
        else:
            self.augmentation_sampler.add_grad_of_copy(aug_sampler)
        del aug_sampler

    def reset_state(self):
        del self.agumentation_sampler_copies
        self.agumentation_sampler_copies = []

    def write_summary(self, aug_sampler, step):
        if step % 100 == 0 and self.summary_writer is not None and self.training:
            with torch.no_grad():
                q = aug_sampler.compute_q()  # need to compute q here, because of print in first step
                op_logits = aug_sampler.op_embs @ q
                p = aug_sampler.dist(op_logits, 0)
                for i, p_ in enumerate(p):
                    self.summary_writer.add_scalar(f'PreprocessorWeights/p_{i}', p_, step)
                hidden = aug_sampler.op_embs
                if aug_sampler.q_residual:
                    hidden += q
                scale_logits = hidden @ aug_sampler.scale_embs.t()
                p_scale = aug_sampler.dist(scale_logits, 1)
                for aug_idx, scale_dist in enumerate(p_scale):
                    self.summary_writer.add_scalar(f'MaxScale/aug_{aug_idx}', torch.argmax(scale_dist), step)
                for aug_idx, scale_dist in enumerate(p_scale):
                    self.summary_writer.add_scalar(f'AvgScale/aug_{aug_idx}', scale_dist @ torch.arange(scale_dist.shape[-1],dtype=scale_dist.dtype,device=scale_dist.device), step)
                for i, p_ in enumerate(p_scale.mean(0)):
                    self.summary_writer.add_scalar(f'PreprocessorWeightsScale/p_{i}', p_, step)



# THIS is the one used

class LearnedRandAugmentPreprocessor(ImagePreprocessor):
    def __init__(self, dataset_info, hidden_dimension, optimizer_creator, bs, val_bs, entropy_alpha, scale_entropy_alpha=0.,
                 importance_sampling=False, cutout=0, normalize_reward=True, model_for_online_tests=None,
                 D_out=10, q_zero_init=True, q_residual=False, scale_embs_zero_init=False, scale_embs_zero_strength_bias=0., label_smoothing_rate=0.,
                 possible_num_sequential_transforms=[1,2,3,4], sigmax_dist=False, exploresoftmax_dist=False, use_images_for_sampler=False, use_labels_for_sampler=False, uniaug_val=False, old_preprocessor_val=False, current_preprocessor_val=False,
                 aug_probs=False, use_non_embedding_sampler=False, ppo=False, scale_trainable=True, standard_augmentation_on_validation_steps=True, delete_every_x_steps=0, use_rnn_sampler=False, use_simple_sampler=False, summary_prefix='',
                 update_every_k_steps=1,le_softmax_dist=False,**kwargs):
        super().__init__(**kwargs)
        print('using learned preprocessor')

        self.num_transforms = google_augmentations.num_augmentations()
        self.num_scales = google_augmentations.PARAMETER_MAX + 1
        self.standard_cifar_preprocessor = get_standard_preprocessor(dataset_info, cutout, device=self.device)
        self.model = model_for_online_tests

        dists = (torch.softmax, torch.log_softmax)
        assert sum([sigmax_dist, exploresoftmax_dist, le_softmax_dist]) <= 1
        if sigmax_dist:
            dists = (sigmax, log_sigmax)
        if exploresoftmax_dist:
            dists = (exploresoftmax, log_exploresoftmax)
        if le_softmax_dist:
            print('use le softmax')
            dists = (le_softmax, log_le_softmax)

        def init_augmentation_sampler():
            if use_non_embedding_sampler:
                aug_sampler_type = NonEmbeddingRandAugmentationSampler
            elif use_rnn_sampler:
                aug_sampler_type = RNNAugmentationSampler
            elif use_simple_sampler:
                aug_sampler_type = SimpleAugmentationSampler
            else:
                aug_sampler_type = RandAugmentationSampler
            self.augmentation_sampler = aug_sampler_type(hidden_dimension, self.num_transforms, self.num_scales, torch.tensor(possible_num_sequential_transforms),
                                                            q_residual, q_zero_init, scale_embs_zero_init, scale_embs_zero_strength_bias,
                                                            label_smoothing_rate, dists,use_images_for_sampler, use_labels_for_sampler, aug_probs, scale_trainable, dataset_info).to(self.device)
        self.init_augmentation_sampler = init_augmentation_sampler
        init_augmentation_sampler()
        self.agumentation_sampler_copies = []
        self.bs = bs
        self.val_bs = val_bs
        self.ppo = ppo

        self.normalize_reward = normalize_reward
        self.optimizer = optimizer_creator(self.augmentation_sampler.parameters())
        self.entropy_alpha = entropy_alpha
        self.scale_entropy_alpha = scale_entropy_alpha
        self.importance_sampling = importance_sampling
        self.possible_num_sequential_transforms = possible_num_sequential_transforms
        self.standard_augmentation_on_validation_steps = standard_augmentation_on_validation_steps
        self.delete_every_x_steps = delete_every_x_steps
        self.uniaug_val = uniaug_val
        self.aug_probs = aug_probs
        self.old_preprocessor_val = old_preprocessor_val
        if self.old_preprocessor_val:
            self.old_preprossors = [deepcopy(self.augmentation_sampler)]
            self.num_model_updates = 0
        self.current_preprocessor_val = current_preprocessor_val
        self.summary_prefix = summary_prefix
        self.update_every_k_steps= update_every_k_steps

    def val_augmentations(self, imgs):
        assert sum([self.uniaug_val, self.old_preprocessor_val, self.current_preprocessor_val]) <= 1

        if not imgs:
            return []

        if self.uniaug_val:
            def uni_aug_augmentations():
                num_transf = self.possible_num_sequential_transforms[random.randint(0,len(self.possible_num_sequential_transforms)-1)]
                transforms = []
                for t_i in range(num_transf):
                    if not self.aug_probs or random.random() > .5:
                        transforms.append((random.randint(0,self.num_transforms-1), random.randint(0,google_augmentations.PARAMETER_MAX)))
                return transforms
            return [uni_aug_augmentations() for i in imgs]
        elif self.current_preprocessor_val:
            with torch.no_grad():
                sampled_op_idxs, sampled_scales = self.augmentation_sampler(imgs)
                return [[(i,s) for i,s in zip(i_s,s_s)] for i_s,s_s in zip(sampled_op_idxs,sampled_scales)]
        elif self.old_preprocessor_val:
            with torch.no_grad():
                sampled_op_idxs, sampled_scales = self.old_preprossors[random.randint(0,len(self.old_preprossors)-1)](imgs)
                return [[(i, s) for i, s in zip(i_s, s_s)] for i_s, s_s in zip(sampled_op_idxs, sampled_scales)]
        else:
            return [[(0,1.)] for i in imgs]


    def forward(self, imgs, step, validation_step=False, labels=None):
        self.t = step
        if self.training:
            if validation_step:
                assert not self.val_bs
                activated_transforms_for_batch = self.val_augmentations(imgs) # [self.val_augmentations() for _ in range(self.bs)]
            else:
                aug_sampler = deepcopy(self.augmentation_sampler)
                self.agumentation_sampler_copies.append(aug_sampler)
                sampled_op_idxs, sampled_scales = aug_sampler(imgs[:self.bs],labels)
                activated_transforms_for_batch = [[(i.item(),s.item()) for i,s in zip(i_s,s_s)] for i_s,s_s in zip(sampled_op_idxs,sampled_scales)] + self.val_augmentations(imgs[self.bs:]) # [self.val_augmentations() for _ in range(self.val_bs)]
                self.write_summary(aug_sampler, step)
        else:
            activated_transforms_for_batch = [[(0, 1.)] for i in imgs]  # do not apply any augmentation when evaluating
        t_imgs = []
        google_augmentations.blend_images = imgs
        #print(activated_transforms_for_batch)
        #print(sum([l[0]==0 for ll in activated_transforms_for_batch for l in ll])/sum(1 for ll in activated_transforms_for_batch for l in ll))
        #total_ops = sum(1 for ll in activated_transforms_for_batch for l in ll)
        #aug_type_counts = Counter([l[0] for ll in activated_transforms_for_batch for l in ll])
        #print(sorted([(k,v/total_ops) for k,v in aug_type_counts.items()]))
        #aug_type_counts_in_first = Counter([ll[0][0] for ll in activated_transforms_for_batch])
        #print(sorted([(k,v/len(activated_transforms_for_batch)) for k,v in aug_type_counts_in_first.items()]))
        #strength_counts = Counter([l[1] for ll in activated_transforms_for_batch for l in ll])
        #print(sorted([(k,v/total_ops) for k,v in strength_counts.items()]))
        for i, (img, augs) in enumerate(zip(imgs, activated_transforms_for_batch)):
            for op, scale in augs:
                img = google_augmentations.apply_augmentation(op, scale, img)
            t_imgs.append(img)

        if self.importance_sampling and self.training and not validation_step:
            w = 1. / (self.num_transforms * self.num_scales * torch.exp(aug_sampler.logps.detach()))
            return self.standard_cifar_preprocessor(t_imgs, step), w * (len(w) / torch.sum(w))

        if self.training and validation_step and not self.standard_augmentation_on_validation_steps:
            self.standard_cifar_preprocessor.training = False
            #t_imgs = [google_augmentations.flip_ud.pil_transformer(1.,0)(img) for img in t_imgs] #to check what the model actually does
            r = self.standard_cifar_preprocessor(t_imgs, step)
            self.standard_cifar_preprocessor.training = True
            return r
        self.update_now = self.t % self.update_every_k_steps == 0
        return self.standard_cifar_preprocessor(t_imgs, step)

    def compute_weights(self, rewards):
        if self.normalize_reward:
            rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)
        return rewards

    def step(self, rewards, curr_lr_factor=None):
        assert len(self.agumentation_sampler_copies) <= 2
        aug_sampler = self.agumentation_sampler_copies.pop(0)  # pops the oldest state first (queue-style)
        if self.t % 100 == 0 and self.summary_writer is not None and self.training:
            self.summary_writer.add_scalar(f'{self.summary_prefix}Alignment/AverageAlignment', rewards.mean(), self.t)
            self.summary_writer.add_scalar(f'{self.summary_prefix}Alignment/MaxAlignment', rewards.max(), self.t)
            self.summary_writer.add_scalar(f'{self.summary_prefix}Alignment/MinAlignment', rewards.min(), self.t)
            if hasattr(aug_sampler, 'aug_logits') and aug_sampler.aug_logits is not None:
                ps = torch.sigmoid(aug_sampler.aug_logits)
                for i,p in enumerate(ps):
                    self.summary_writer.add_scalar(f'{self.summary_prefix}AugProbs/aug{i}', p,
                                                   self.t)

        with torch.no_grad():
            if self.importance_sampling:
                rewards *= 1. / ((self.num_transforms * self.num_scales) ** 2 * torch.exp(2 * self.logps))
            weights = self.compute_weights(rewards).detach().to(self.device)

        aug_sampler.zero_grad()
        if self.ppo:
            replace_parameters(self.optimizer, self.augmentation_sampler.parameters(), aug_sampler.parameters())
            update(aug_sampler, self.optimizer, weights, self.entropy_alpha)
            recursive_backpack_memory_cleanup(aug_sampler)
            self.augmentation_sampler.load_state_dict(aug_sampler.state_dict())
            replace_parameters(self.optimizer, aug_sampler.parameters(), self.augmentation_sampler.parameters())
        else:
            loss = -aug_sampler.logps @ weights.detach() / float(len(weights))
            if self.entropy_alpha:
                ent = aug_sampler.get_numaug_entropy()
                loss -= self.entropy_alpha * ent
            if self.scale_entropy_alpha:
                ent = aug_sampler.get_scale_entropy()
                loss -= self.scale_entropy_alpha * ent

            loss.backward()
            torch.nn.utils.clip_grad_value_(aug_sampler.parameters(), 5.)
            if self.update_now:
                self.augmentation_sampler.add_grad_of_copy(aug_sampler)
                self.optimizer.step()
                self.augmentation_sampler.zero_grad()
                self.update_now = False
            else:
                self.augmentation_sampler.add_grad_of_copy(aug_sampler)
        del aug_sampler
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
                self.old_preprossors.append(deepcopy(self.augmentation_sampler))

    def reset_state(self):
        del self.agumentation_sampler_copies
        if self.delete_every_x_steps and hasattr(self,'t') and self.t % self.delete_every_x_steps == 0:
            self.init_augmentation_sampler()
        self.agumentation_sampler_copies = []

    def write_summary(self, aug_sampler, step):
        if step % 101 == 0 and self.summary_writer is not None and self.training:
            with torch.no_grad():
                if hasattr(aug_sampler, 'get_numaug_distribution'):
                    for i,p in enumerate(aug_sampler.get_numaug_distribution()):
                        self.summary_writer.add_scalar(f'{self.summary_prefix}NumAugs/{i}', p, step)
                if hasattr(aug_sampler, 'get_scale_distribution'):
                    p_scale = aug_sampler.get_scale_distribution()

                    for aug_idx, scale_dist in enumerate(p_scale):
                        self.summary_writer.add_scalar(f'{self.summary_prefix}AvgScale/aug_{aug_idx}', scale_dist @ torch.arange(scale_dist.shape[-1],dtype=scale_dist.dtype,device=scale_dist.device), step)
                    for aug_idx, scale_dist in enumerate(p_scale):
                        self.summary_writer.add_scalar(f'{self.summary_prefix}MaxScale/aug_{aug_idx}', torch.argmax(scale_dist), step)
                if hasattr(aug_sampler, 'get_aug_distribution'):
                    p_aug = aug_sampler.get_aug_distribution()
                    for aug_idx, aug_p in enumerate(p_aug):
                        self.summary_writer.add_scalar(f'{self.summary_prefix}AugProb/aug_{aug_idx}', aug_p, step)

class LearnedPreprocessorEnsemble(ImagePreprocessor):
    def __init__(self, dataset_info, hidden_dimension, optimizer_creator, bs, val_bs, *args, num_preprocessors=8, **kwargs):
        super().__init__()
        self.bs = bs
        self.val_bs = val_bs
        self.preprocessors = nn.ModuleList()
        assert bs % num_preprocessors == 0
        for i in range(num_preprocessors):
            self.preprocessors.append(
                LearnedRandAugmentPreprocessor(dataset_info,hidden_dimension,optimizer_creator,bs//num_preprocessors,
                                               val_bs if i == num_preprocessors - 1 else 0,
                                               *args, summary_prefix=f'Prepr{i}', **kwargs)
            )

    def forward(self, imgs, step, validation_step=False, labels=None):
        offset = 0
        out_imgs = []
        for i, prepr in enumerate(self.preprocessors):
            if i == len(self.preprocessors) - 1:
                end = self.bs + self.val_bs
            else:
                end = offset + self.bs // len(self.preprocessors)
            img_subbatch = imgs[offset:end]
            labels_subbatch = labels[offset:end] if labels is not None else None
            if not img_subbatch and not self.training:
                break
            out_imgs.append(prepr(img_subbatch, step, validation_step, labels=labels_subbatch))
            offset = end
        return torch.cat(out_imgs, dim=0)

    def step(self, rewards, curr_lr_factor=None):
        offset = 0
        for prepr in self.preprocessors:
            end = offset + self.bs // len(self.preprocessors)
            prepr.step(rewards[offset:end], curr_lr_factor)
            offset = end
        assert end == len(rewards)

    def reset_state(self):
        for prepr in self.preprocessors:
            prepr.reset_state()



