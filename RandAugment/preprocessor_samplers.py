import random

import torch
from torch import nn

import torchvision
from torchvision import transforms

from RandAugment.networks.convnet import SeqConvNet
from RandAugment.common import ListDataLoader

class Sampler(nn.Module):
    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad


class RandAugmentationSampler(nn.Module):
    def __init__(self, hidden_dimension, num_transforms, num_scales, possible_num_sequential_transforms, q_residual, q_zero_init, scale_embs_zero_init, scale_embs_zero_strength_bias,
                 label_smoothing_rate, distribution_functions, use_images, use_labels, aug_probs, scale_logits_trainable, dataset_info):
        assert scale_logits_trainable
        super().__init__()
        self.dist, self.log_dist = distribution_functions
        if aug_probs:
            self.aug_logits = nn.Parameter(torch.zeros(num_transforms, requires_grad=True))
        else:
            self.aug_logits = None
        self.op_embs = nn.Parameter(torch.normal(0., 1., (num_transforms, hidden_dimension), requires_grad=True))
        self.num_transforms_embs = nn.Parameter(torch.normal(0., 1., (len(possible_num_sequential_transforms), hidden_dimension), requires_grad=True))
        self.has_input = use_labels or use_images

        scale_embs_shape = (num_transforms,num_scales,hidden_dimension) if self.has_input else (num_scales,hidden_dimension)
        if scale_embs_zero_init:
            self.scale_embs = nn.Parameter(torch.zeros(scale_embs_shape, requires_grad=True))
        else:
            self.scale_embs = nn.Parameter(torch.normal(0., 1., scale_embs_shape, requires_grad=True))
        self.label_smoothing_rate = label_smoothing_rate
        self.q_residual = q_residual
        self.use_images = use_images
        self.use_labels = use_labels
        if self.use_images:
            self.normalize = transforms.Normalize(dataset_info['mean'], dataset_info['std'])
            self.convnet = SeqConvNet(hidden_dimension)
            if q_zero_init:
                with torch.no_grad():
                    self.convnet.final_fc.weight.zero_()
                    self.convnet.final_fc.bias.zero_()
        elif self.use_labels:
            if q_zero_init:
                self.q_params = nn.Parameter(torch.zeros(dataset_info['num_labels'],hidden_dimension),requires_grad=True)
            else:
                self.q_params = nn.Parameter(torch.normal(0.,1.,(dataset_info['num_labels'],hidden_dimension)),requires_grad=True)
        else:
            if q_zero_init:
                self.q_param = nn.Parameter(torch.zeros(hidden_dimension, requires_grad=True))
            else:
                self.q_param = nn.Parameter(torch.normal(0., 1., hidden_dimension, requires_grad=True))

        self.possible_num_sequential_transforms = possible_num_sequential_transforms
        self.scale_embs_zero_strength_bias = scale_embs_zero_strength_bias

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad

    def compute_q(self, imgs=None, labels=None):
        if self.use_images:
            assert imgs is not None
            image_batch = torch.stack([self.normalize(torchvision.transforms.ToTensor()(i)) for i in imgs]).to(self.convnet.final_fc.weight.device)
            return self.convnet(image_batch)
        elif self.use_labels:
            return torch.stack([self.q_params[l] for l in labels])
        else:
            return self.q_param.unsqueeze(0).expand(len(imgs),-1)

    def forward(self, imgs, labels=None):
        self.imgs = imgs
        num_samples = len(imgs)
        num_ops = len(self.op_embs)
        max_num_sequential_transforms = len(self.num_transforms_embs)
        augmentation_inds = torch.randint(num_ops,(num_samples,max_num_sequential_transforms))
        self.augmentation_inds = augmentation_inds
        self.q = self.compute_q(imgs, labels) # num_samples x hidden

        self.num_transforms_logits = self.q @ self.num_transforms_embs.t() # num_samples x max_num_sequential_transforms
        self.p_num_transforms = self.dist(self.num_transforms_logits, -1)
        sampled_transform_indices = torch.multinomial(self.p_num_transforms, 1, replacement=True).squeeze(1) # num_samples
        self.sampled_numaug_indices = sampled_transform_indices
        sampled_num_transforms = self.possible_num_sequential_transforms[sampled_transform_indices]
        augmentation_mask = torch.arange(max_num_sequential_transforms).unsqueeze(0).expand(num_samples,max_num_sequential_transforms).to(sampled_num_transforms.device) >= sampled_num_transforms.unsqueeze(1) # num_samples x max_num_seq_transforms
        self.augmentation_mask = augmentation_mask

        self.scale_logits = scale_logits = self.get_scale_logits(augmentation_inds, q=self.q) # num_samples x max_num_seq_tran x num_scales
        p_scale = self.dist(scale_logits, 2)
        sampled_scales = torch.multinomial(p_scale.view(-1,p_scale.shape[2]), 1, replacement=True).view(num_samples, max_num_sequential_transforms)
        self.sampled_scales = sampled_scales
        log_ps_of_sampled_scales = self.get_scale_logps(sampled_scales, scale_logits) # num_samples x max_num_seq_tran

        augmentation_inds[augmentation_mask] = 0 # index of identity augmentation
        log_ps_of_sampled_scales[augmentation_mask] = 0.

        if self.aug_logits is not None:
            aug_logits = self.aug_logits # num_ops
            aug_ps = torch.sigmoid(aug_logits)
            keep = torch.bernoulli(aug_ps[augmentation_inds]) # num_samples x max_num_seq_trans
            self.keep = keep
            aug_logps = self.get_aug_logps(keep, augmentation_inds, augmentation_mask) # num_samples x max_num_seq_trans
            augmentation_inds[keep == 0.0] = 0
            log_ps_of_sampled_scales[keep == 0.0] = 0.0
            self.logps = self.get_numaug_logps(sampled_transform_indices, self.num_transforms_logits) + log_ps_of_sampled_scales.sum(1) + aug_logps
        else:
            self.logps = self.get_numaug_logps(sampled_transform_indices, self.num_transforms_logits) + log_ps_of_sampled_scales.sum(1)
        return augmentation_inds.cpu(), sampled_scales.cpu()

    def evaluate(self, imgs, augmentation_inds, sampled_numaug_indices, augmentation_mask, sampled_scales, keep=None):
        q = self.compute_q(imgs)
        num_transforms_logits = q @ self.num_transforms_embs.t()
        numaug_logps = self.get_numaug_logps(sampled_numaug_indices, num_transforms_logits)

        scale_logits = self.get_scale_logits(augmentation_inds,q)
        self.scale_logits = scale_logits


        scale_logps = self.get_scale_logps(sampled_scales, scale_logits)
        scale_logps[augmentation_mask] = 0.
        logp = numaug_logps + scale_logps.sum(1)
        self.p_num_transforms = self.dist(num_transforms_logits, -1)
        self.augmentation_mask = augmentation_mask
        entropy = self.get_numaug_entropy()+self.get_scale_entropy()
        if self.aug_logits is not None:
            assert keep is not None
            aug_logps = self.get_aug_logps(keep, augmentation_inds, augmentation_mask)
            scale_logps[keep == 0.0] = 0.0
            logp += aug_logps
            entropy += self.get_augprob_entropy()

        return logp, entropy

    def get_scale_logps(self, sampled_scales, scale_logits):
        log_p_scale = self.log_dist(scale_logits, 2)
        flat_log_p_scale = log_p_scale.view(-1,log_p_scale.shape[2])
        log_ps_of_sampled_scales = flat_log_p_scale[torch.arange(len(flat_log_p_scale)),sampled_scales.flatten()].view_as(sampled_scales)
        return log_ps_of_sampled_scales

    def get_numaug_logps(self, sampled_numaug_indices, num_transforms_logits):
        self.log_p_num_transforms = self.log_dist(num_transforms_logits, -1)
        return self.log_p_num_transforms[torch.arange(len(self.log_p_num_transforms)), sampled_numaug_indices]

    def get_aug_logps(self, keep, augmentation_inds, augmentation_mask):
        aug_logits = self.aug_logits
        aug_ps = torch.sigmoid(aug_logits)
        aug_1logps = torch.nn.functional.logsigmoid(aug_logits)
        aug_0logps = torch.log(1. - aug_ps + 10E-6)
        del aug_ps
        augmentation_inds[keep == 0.0] = 0  # index of id
        aug_logps = torch.gather(torch.cat([aug_0logps, aug_1logps]), 0,
                                 augmentation_inds.cuda().flatten() + (keep.int().flatten() * len(aug_0logps))).view_as(
            augmentation_inds)
        aug_logps[augmentation_mask] = 0.
        aug_logps = aug_logps.sum(1)
        return aug_logps

    def get_scale_logits(self, augmentation_inds, q):
        if q is None:
            q = self.compute_q() # images???
        if self.has_input:
            hidden = q
            scale_logits = torch.einsum('bh,bosh->bos',hidden,self.scale_embs[augmentation_inds])
        else:
            hidden = self.op_embs[augmentation_inds]
            if self.q_residual:
                hidden += q.unsqueeze(1).expand(-1, hidden.shape[1], -1)
            scale_logits = hidden @ self.scale_embs.t()

        if self.scale_embs_zero_strength_bias:
            scale_logits[:,:,0] += self.scale_embs_zero_strength_bias
        return scale_logits

    def get_numaug_entropy(self):
        neg_entropy = torch.einsum('bt,bt->b', self.p_num_transforms,
                                   (self.log_p_num_transforms * (self.p_num_transforms != 0.))).mean()
        return - neg_entropy

    def get_augprob_entropy(self):
        d = torch.distributions.Bernoulli(logits=self.aug_logits)
        return d.entropy().sum()

    def get_scale_entropy(self):
        log_p_scale = self.log_dist(self.scale_logits, 2)
        p_scale = self.dist(self.scale_logits, 2)
        gpu_aug_mask_float = self.augmentation_mask.float().to(p_scale.device)
        avg_neg_entropy = (torch.einsum('bos,bos->bo', p_scale,
                                        log_p_scale * (p_scale != 0.)) * (1. - gpu_aug_mask_float)).sum() / (
                                  1. - gpu_aug_mask_float).sum()  # here we could have logp = -inf and p = 0.
        return - avg_neg_entropy

    def get_numaug_distribution(self):
        return self.p_num_transforms.mean(0)

    def get_scale_distribution(self):
        augmentation_inds = torch.arange(len(self.op_embs)).unsqueeze(0).expand(len(self.q), -1)
        scale_logits = self.get_scale_logits(augmentation_inds, self.q)
        p_scale = self.dist(scale_logits, 2).mean(0)
        return p_scale

def update(model: RandAugmentationSampler, opt, all_rewards, ent_alpha):
    epochs = 4
    eps_clip = .2

    data = (
        model.imgs, model.augmentation_inds, model.sampled_numaug_indices, model.augmentation_mask, model.sampled_scales,
        model.keep if hasattr(model, 'keep') else len(model.logps) * [None], model.logps.detach(), all_rewards)
    for _ in range(epochs):
        #data_loader = torch.utils.data.DataLoader(
        #    torch.utils.data.TensorDataset(model.imgs, model.augmentation_inds, model.sampled_numaug_indices, model.augmentation_mask, model.sampled_scales, model.keep, model.logps, all_rewards), batch_size=32,shuffle=True,drop_last=True)
        data_loader = ListDataLoader(*data, bs=32)
        for imgs, augmentation_inds, sampled_numaug_indices, augmentation_mask, sampled_scales, keep, old_logprobs, rewards in data_loader:
            augmentation_inds = torch.stack(augmentation_inds)
            sampled_numaug_indices = torch.stack(sampled_numaug_indices)
            augmentation_mask = torch.stack(augmentation_mask)
            sampled_scales = torch.stack(sampled_scales)
            keep = torch.stack(keep) if keep[0] is not None else None
            old_logprobs = torch.stack(old_logprobs)
            rewards = torch.stack(rewards)
            with torch.autograd.set_detect_anomaly(True):
                # Evaluating old actions and values :
                logprobs, dist_entropy = model.evaluate(imgs,augmentation_inds,sampled_numaug_indices, augmentation_mask, sampled_scales,keep=keep)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs)
                if torch.isinf(ratios).any():
                    continue

                # Finding Surrogate Loss:
                advantages = rewards # no baseline :/
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                loss = -torch.min(surr1, surr2).mean() - ent_alpha * dist_entropy # + 0.5 * mse_loss(state_values, rewards)

                # take gradient step
                opt.zero_grad()
                loss.backward()
                opt.step()


class NonEmbeddingRandAugmentationSampler(nn.Module):
    def __init__(self, hidden_dimension, num_transforms, num_scales, possible_num_sequential_transforms, q_residual, q_zero_init, scale_embs_zero_init, scale_embs_zero_strength_bias,
                 label_smoothing_rate, distribution_functions, use_images, use_labels, aug_probs, scale_logits_trainable, dataset_info):
        super().__init__()
        assert not use_images
        self.dist, self.log_dist = distribution_functions
        if aug_probs:
            self.aug_logits = nn.Parameter(torch.zeros(num_transforms, requires_grad=True))
        else:
            self.aug_logits = None
        self.num_transforms_logits = nn.Parameter(torch.zeros(len(possible_num_sequential_transforms), requires_grad=True))

        self.scale_logits = nn.Parameter(torch.zeros(num_transforms,num_scales))
        if scale_logits_trainable:
            self.scale_logits.requires_grad = True
        else:
            self.scale_logits.requires_grad = False
        self.label_smoothing_rate = label_smoothing_rate

        self.possible_num_sequential_transforms = possible_num_sequential_transforms
        self.scale_embs_zero_strength_bias = scale_embs_zero_strength_bias

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad

    def forward(self, imgs, labels=None):
        num_samples = len(imgs)
        self.p_num_transforms = self.dist(self.num_transforms_logits, -1)
        self.log_p_num_transforms = self.log_dist(self.num_transforms_logits, -1)
        sampled_transform_indices = torch.multinomial(self.p_num_transforms, len(imgs), replacement=True)
        sampled_num_transforms = self.possible_num_sequential_transforms[sampled_transform_indices]
        augmentation_inds = torch.randint(self.scale_logits.shape[0],(num_samples,self.p_num_transforms.shape[0]))#, device=self.q.device)
        augmentation_mask = torch.arange(augmentation_inds.shape[1]).unsqueeze(0).expand(num_samples,augmentation_inds.shape[1]).to(sampled_num_transforms.device) >= sampled_num_transforms.unsqueeze(1)
        self.augmentation_mask = augmentation_mask
        augmentation_inds[augmentation_mask] = 0 # index of identity augmentation

        scale_logits = self.scale_logits
        if self.scale_embs_zero_strength_bias:
            scale_logits[:,0] += self.scale_embs_zero_strength_bias
        scale_log_ps = self.log_dist(self.scale_logits,1)
        scale_ps = self.dist(self.scale_logits,1)
        sampled_scales = torch.multinomial(scale_ps[augmentation_inds.flatten()],1).view_as(augmentation_inds)
        log_ps_of_sampled_scales = scale_log_ps[augmentation_inds.flatten(),sampled_scales.flatten()].view_as(sampled_scales)
        log_ps_of_sampled_scales[augmentation_mask] = 0.
        if self.aug_logits is not None:
            aug_logits = self.aug_logits  # [augmentation_inds.flatten()].view_as(augmentation_inds)
            aug_ps = torch.sigmoid(aug_logits)
            aug_1logps = torch.nn.functional.logsigmoid(aug_logits)
            del aug_logits
            aug_0logps = torch.log(1. - aug_ps + 10E-6)
            keep = torch.bernoulli(aug_ps[augmentation_inds], 1)
            del aug_ps
            augmentation_inds[keep == 0.0] = 0  # index of id
            aug_logps = torch.gather(torch.cat([aug_0logps, aug_1logps]), 0,
                                     augmentation_inds.cuda().flatten() + (keep.int().flatten() * len(aug_0logps))).view_as(
                augmentation_inds)
            aug_logps[augmentation_mask] = 0.
            aug_logps = aug_logps.sum(1)

            del aug_1logps, aug_0logps
            log_ps_of_sampled_scales[keep == 0.0] = 0.0
            self.logps = self.log_p_num_transforms[sampled_transform_indices] + log_ps_of_sampled_scales.sum(1) + aug_logps
        else:
            self.logps = self.log_p_num_transforms[sampled_transform_indices] + log_ps_of_sampled_scales.sum(1)
        return augmentation_inds.cpu(), sampled_scales.cpu()

    def get_numaug_entropy(self):
        neg_entropy = self.p_num_transforms @ (self.log_p_num_transforms * (self.p_num_transforms != 0.))
        return -neg_entropy

    def get_scale_entropy(self):
        log_p_scale = self.log_dist(self.scale_logits, 1)
        p_scale = self.dist(self.scale_logits, 1)
        avg_neg_entropy = (torch.einsum('os,os->o', p_scale,
                                        log_p_scale * (p_scale != 0.))).mean()  # here we could have logp = -inf and p = 0.

        return - avg_neg_entropy

    def get_scale_distribution(self):
        return self.dist(self.scale_logits, 1)

    def get_numaug_distribution(self):
        return self.p_num_transforms

class RNNAugmentationSampler(Sampler):
    """
    For each image this augmentationsampler runs an LSTM for a predefined number of steps (sampled from `possible_num_sequential_transforms` per step so each example in a batch has the same num of possible augs).
    The steps of the LSTM are of two interchanging kinds
        - augmentation prediction
        - strength prediction
    The prediction in each step is sampled for each step  and the regarding embedding is fed back to the RNN.
    """
    def __init__(self, hidden_dimension, num_transforms, num_scales, possible_num_sequential_transforms, q_residual, q_zero_init, scale_embs_zero_init, scale_embs_zero_strength_bias,
                 label_smoothing_rate, distribution_functions, use_images, use_labels, aug_probs, scale_logits_trainable, dataset_info):
        super().__init__()
        self.possible_num_sequential_transforms = possible_num_sequential_transforms



        self.h0 = nn.Parameter(torch.zeros(hidden_dimension))
        self.c0 = nn.Parameter(torch.zeros(hidden_dimension))
        self.i0 = nn.Parameter(torch.zeros(hidden_dimension))
        self.rnn = nn.LSTMCell(hidden_dimension,hidden_dimension)
        self.aug_pred = nn.Linear(hidden_dimension, num_transforms, bias=False)
        self.scale_pred = nn.Linear(hidden_dimension, num_scales, bias=False)

    def rnn_step(self, h_c, augmentation_ind=None, scale_ind=None):
        if augmentation_ind is not None:
            assert self.aug_pred.weight.shape[1] == len(h_c[0][0]), "Just to check orietnation of weight."
            inp = self.aug_pred.weight[augmentation_ind]
        elif scale_ind is not None:
            inp = self.scale_pred.weight[scale_ind]
        else:
            inp = self.i0.unsqueeze(0).repeat(len(h_c[0]),1)
        return self.rnn(inp, h_c)

    def rnn_double_step(self, h_c, last_scale_ind=None):
        h_c1 = self.rnn_step(h_c, scale_ind=last_scale_ind)
        aug_dist = torch.distributions.Categorical(logits=self.aug_pred(h_c1[0]))
        aug_ind = aug_dist.sample()
        h_c2 = self.rnn_step(h_c1, augmentation_ind=aug_ind)
        scale_dist = torch.distributions.Categorical(logits=self.scale_pred(h_c1[0]))
        scale_ind = scale_dist.sample()
        return h_c2, (aug_dist, aug_ind), (scale_dist, scale_ind)

    def initial_h_c(self, bs):
        return (self.h0.unsqueeze(0).repeat(bs,1), self.c0.unsqueeze(0).repeat(bs,1))

    def forward(self, imgs):
        num_samples = len(imgs)
        h_c = self.initial_h_c(num_samples)

        scale_ind = None
        sampled_augs = torch.zeros((num_samples, max(self.possible_num_sequential_transforms)), dtype=torch.int64)
        sampled_scales = torch.zeros((num_samples, max(self.possible_num_sequential_transforms)), dtype=torch.int64)
        self.logps = torch.zeros(num_samples,device=self.h0.device)

        num_augs = random.choice(self.possible_num_sequential_transforms)

        for i in range(num_augs):
            h_c, (aug_dist, aug_ind), (scale_dist, scale_ind) = self.rnn_double_step(h_c, last_scale_ind=scale_ind)
            sampled_augs[:,i] = aug_ind
            sampled_scales[:,i] = scale_ind
            self.logps += aug_dist.log_prob(aug_ind) + scale_dist.log_prob(scale_ind)

        return sampled_augs, sampled_scales

class SimpleAugmentationSampler(Sampler):
    def __init__(self, hidden_dimension, num_transforms, num_scales, possible_num_sequential_transforms, q_residual, q_zero_init, scale_embs_zero_init, scale_embs_zero_strength_bias,
                 label_smoothing_rate, distribution_functions, use_images, use_labels, aug_probs, scale_logits_trainable, dataset_info):
        super().__init__()
        self.possible_num_sequential_transforms = possible_num_sequential_transforms

        self.aug_logits = nn.Parameter(torch.zeros(num_transforms))
        self.scale_logits = nn.Parameter(torch.zeros(num_scales))

    def forward(self, imgs):
        num_samples = len(imgs)
        num_augs = random.choice(self.possible_num_sequential_transforms)
        aug_dist = torch.distributions.Categorical(logits=self.aug_logits)
        scale_dist = torch.distributions.Categorical(logits=self.scale_logits)

        sampled_augs = aug_dist.sample((num_samples,num_augs))
        sampled_scales = scale_dist.sample((num_samples,num_augs))

        self.logps = aug_dist.log_prob(sampled_augs).sum(1) + scale_dist.log_prob(sampled_scales).sum(1)

        return sampled_augs, sampled_scales

    def get_scale_distribution(self):
        return torch.softmax(self.scale_logits,0).unsqueeze(0).repeat(len(self.aug_logits),1)

    def get_aug_distribution(self):
        return torch.softmax(self.aug_logits,0)

