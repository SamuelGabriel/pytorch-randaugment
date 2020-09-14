import torch
from torch import nn
import torch.nn.functional as F

from copy import deepcopy
import time
from RandAugment.common import CheckpointFunctionForSampler

class Sampler(nn.Module):
    def __init__(self, num_dropouts, hidden_dimension, out_bias=False, relu=True):
        super().__init__()
        self.get_keep_logits = nn.Sequential(
            nn.Linear(num_dropouts,hidden_dimension,bias=False),
            nn.ReLU() if relu else nn.Identity(),
            nn.Linear(hidden_dimension,num_dropouts,bias=out_bias),
        )
    def forward(self, state):
        l = self.get_keep_logits(state)
        p = torch.sigmoid(l)
        #with torch.no_grad():
        #    print('p[0,:10]',torch.round(p[0,:10] * 10**2) / (10**2))
        sample = torch.bernoulli(p).to(torch.bool)
        self.true_logps = torch.nn.functional.logsigmoid(l)
        eps = 10E-6
        self.false_logps = torch.log(1.-p+eps) # problem nan gradients, when p = 1.
        #false_logps = -l - torch.log(1.+torch.exp(-l)) # log(1-sig(l))
        self.logps = torch.where(sample,self.true_logps,self.false_logps).sum(1)
        return sample

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad

    def cleanup(self):
        pass

class ConvSampler(nn.Module):
    def __init__(self, conv_params, hidden_dimension, out_bias=False, relu=True):
        planes, kernel_size, stride, padding = conv_params
        super().__init__()
        self.get_keep_logits = nn.Sequential(
            nn.BatchNorm2d(planes, momentum=.1),
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=planes, bias=True),
            nn.ReLU() if relu else nn.Identity(),
            nn.Conv2d(planes, planes, kernel_size=1, bias=out_bias)
        )


    def forward(self, state):
        if True:
            def get_sample_logps_ce(state):
                l = self.get_keep_logits(state)
                p = torch.sigmoid(l)
                sample = torch.bernoulli(p).to(torch.bool)
                eps = 10E-6
                false_logps = torch.log(1. - p + eps)  # problem nan gradients, when p = 1.
                del p
                true_logps = torch.nn.functional.logsigmoid(l)
                del l
                logps = torch.where(sample, true_logps, false_logps).sum(1).sum(1).sum(1)
                ce = (.8 * true_logps + (.2) * false_logps).sum()
                del true_logps, false_logps
                return sample, logps, ce
            sample, self.logps, self.ce = CheckpointFunctionForSampler.apply(get_sample_logps_ce, True, state.requires_grad_())
            sample.detach_()
        else:
            l = self.get_keep_logits(state)
            p = torch.sigmoid(l)
            #with torch.no_grad():
            #    print('p[0,:10]',torch.round(p[0,:10] * 10**2) / (10**2))
            sample = torch.bernoulli(p).to(torch.bool)
            self.true_logps = torch.nn.functional.logsigmoid(l)
            eps = 10E-6
            self.false_logps = torch.log(1.-p+eps) # problem nan gradients, when p = 1.
            #false_logps = -l - torch.log(1.+torch.exp(-l)) # log(1-sig(l))
            self.logps = torch.where(sample,self.true_logps,self.false_logps).sum(1).sum(1).sum(1)
        return sample

    def cleanup(self):
        (self.logps.sum()+self.ce).backward()
        del self.get_keep_logits

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad

class AdaptiveDropouter(nn.Module):
    def __init__(self, dropout_shape, hidden_dimension, optimizer_creator, train_bs, val_bs, cross_entropy_alpha=None, target_p=None, out_bias=False, relu=True, inference_dropout=False, scale_by_p=False, summary_writer=None):
        super().__init__()
        self.target_p = target_p
        self.cross_entropy_alpha = cross_entropy_alpha
        self.normalize_reward = True
        self.scale_by_p = scale_by_p
        self.inference_dropout = inference_dropout
        self.summary_writer = summary_writer
        self.train_bs = train_bs
        self.val_bs = val_bs

        if isinstance(dropout_shape,tuple):
            self.sampler = ConvSampler(dropout_shape, hidden_dimension, out_bias=out_bias, relu=relu)
        else:
            self.sampler = Sampler(dropout_shape,hidden_dimension,out_bias=out_bias,relu=relu)
        self.sampler_copies = []

        self.optimizer = optimizer_creator(self.sampler.parameters())
        self.t = -1

    def forward(self, orig_hiddens):
        # hiddens shall have size None x num_dropouts
        hiddens = orig_hiddens.detach()
        keep_mask = torch.ones_like(hiddens, dtype=torch.bool)
        if self.training:
            self.t += 1
            assert len(hiddens) == self.val_bs + self.train_bs
            sampler = deepcopy(self.sampler)
            self.sampler_copies.append(sampler)

            keep_mask[:self.train_bs] = sampler(hiddens[:self.train_bs])

            self.write_summary(sampler, keep_mask, self.t)
        elif self.inference_dropout:
            assert not torch.is_grad_enabled()
            assert self.val_bs == 0
            keep_mask = self.sampler(hiddens)

        r = keep_mask.detach()*orig_hiddens
        if self.scale_by_p:
            r = r/keep_mask.float().mean()
        del keep_mask

        return r

    def compute_weights(self, rewards):
        if self.normalize_reward:
            rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)
        return rewards

    def step(self, rewards):
        assert len(self.sampler_copies) == 2
        sampler = self.sampler_copies.pop(0)  # pops the oldest state first (queue-style)
        if self.t % 100 == 50 and self.summary_writer is not None and self.training:
            self.summary_writer.add_scalar(f'Alignment/AverageAlignment', rewards.mean(), self.t)
            self.summary_writer.add_scalar(f'Alignment/MaxAlignment', rewards.max(), self.t)
            self.summary_writer.add_scalar(f'Alignment/MinAlignment', rewards.min(), self.t)

        with torch.no_grad():
            rewards = rewards.to(sampler.logps.device)
            weights = self.compute_weights(rewards).detach()

        sampler.zero_grad()
        loss = - weights.detach() @ sampler.logps / float(len(weights))
        if self.cross_entropy_alpha is not None and self.target_p is not None:
            if hasattr(sampler,'ce') and self.target_p != .8:
                raise ValueError("You use conv ada dropout, but that only supports target_p = .8 so far.")
            ce = sampler.ce if hasattr(sampler,'ce') else (self.target_p * sampler.true_logps + (1.-self.target_p) * sampler.false_logps).sum()
            loss -= self.cross_entropy_alpha * ce / float(len(weights))


        loss.backward()
        torch.nn.utils.clip_grad_value_(sampler.parameters(), 5.)
        self.sampler.zero_grad()
        self.sampler.add_grad_of_copy(sampler)
        self.optimizer.step()
        del sampler

    def reset_state(self):
        for c in self.sampler_copies:
            c.cleanup()
        del self.sampler_copies
        self.sampler_copies = []

    def write_summary(self, sampler, keep_mask, step):
        if step % 100 == 0 and self.summary_writer is not None and self.training:
            with torch.no_grad():
                self.summary_writer.add_scalar(f'Dropouter/average_logp', sampler.logps.mean(), step)
                self.summary_writer.add_scalar(f'Dropouter/keep_share', keep_mask.float().mean(), step)

class Modulator(nn.Module):
    def __init__(self, state_size, hidden_dimension, optimizer_creator, out_bias=False, relu=True, summary_writer=None):
        super().__init__()
        self.get_multiplier = nn.Sequential(
            nn.Linear(state_size,hidden_dimension,bias=False),
            nn.ReLU() if relu else nn.Identity(),
            nn.Linear(hidden_dimension,state_size,bias=out_bias),
            nn.Sigmoid()
        )
        self.get_multiplier_copies = []
        self.opt = optimizer_creator(self.get_multiplier.parameters())
        self.summary_writer = summary_writer

        self.t = -1

    def forward(self, orig_hiddens):
        if self.training:
            get_multiplier = deepcopy(self.get_multiplier)
            self.get_multiplier_copies.append(get_multiplier)
        else:
            assert not torch.is_grad_enabled(), "You are in inference mode of nn, but compute gradient. Not supported for this model."
            get_multiplier = self.get_multiplier
        m = get_multiplier(orig_hiddens)
        return m*orig_hiddens

    def step(self, diff_alignment):
        assert len(self.get_multiplier_copies) <= 2
        get_multiplier = self.get_multiplier_copies.pop(0)

        for p in get_multiplier.parameters(): p.grad = None
        grads = torch.autograd.grad(diff_alignment, get_multiplier.parameters()) # takes 1.5s for wres28x10 for bs 128
        for g,p in zip(grads,self.get_multiplier.parameters()):
            p.grad = g
        torch.nn.utils.clip_grad_value_(self.get_multiplier.parameters(), 5.)
        self.opt.step()
        for p in self.get_multiplier.parameters(): p.grad = None
        del get_multiplier, diff_alignment

    def reset_state(self):
        del self.get_multiplier_copies
        self.get_multiplier_copies = []




