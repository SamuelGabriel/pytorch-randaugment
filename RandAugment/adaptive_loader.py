from copy import deepcopy
import random

import torch

from torch.utils.data import DataLoader,Subset
from collections import defaultdict

from RandAugment.common import sigmax, log_sigmax, relabssum, log_relabssum

#torch.utils.data.DataLoader(
#        total_trainset, batch_size=batch, shuffle=True, num_workers=16, pin_memory=True,
#        sampler=valid_sampler, drop_last=False)

class Sampler(torch.nn.Module):
    def __init__(self, num_subsets):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(num_subsets), requires_grad=True)  # keep on cpu, since small

    def forward(self, bs):
        self.p = sigmax(self.logits, 0)
        self.sample = torch.multinomial(self.p, bs, replacement=True)
        self.logps = log_sigmax(self.logits[self.sample], 0)
        return self.sample

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad


class AdaptiveLoaderByLabel():
    def __init__(self, dataset, optimizer_factory, bs, val_bs, summary_writer=None):
        self.ds = dataset
        self.bs = bs
        self.val_bs = val_bs
        if val_bs:
            self.val_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=val_bs, shuffle=True, num_workers=16, pin_memory=False,
                sampler=None, drop_last=False)
        else:
            self.val_loader = []
        self.val_iter = iter(self.val_loader)
        self.epoch = -1

        idxs = defaultdict(list)
        for idx, values in enumerate(dataset):
            label = values[-1]
            idxs[label].append(idx)
        self.subsets = [Subset(dataset,idxs[label]) for label in sorted(idxs.keys())]
        self.avg_batch_alignments = torch.tensor([0.] * len(idxs))
        self.avg_normalized_reward = torch.tensor([0.] * len(idxs))
        self.num_uses_of_partitions = torch.tensor([0] * len(idxs))

        self.sampler = Sampler(len(self.subsets))
        self.sampler_copies = []

        self.optimizer = optimizer_factory(self.sampler.parameters())

        self.summary_writer = summary_writer

    def __next__(self):
        if self.t == len(self):
            raise StopIteration
        self.t += 1
        assert len(self.sampler_copies) < 2
        sampler = deepcopy(self.sampler)
        subset_idxs = sampler(self.bs)
        self.sampler_copies.append(sampler)

        r = self.sample_from_subsets(subset_idxs)

        return r

    def sample_from_subsets(self, subset_idxs):
        b = []
        for i in subset_idxs:
            b.append(self.subsets[i][random.randint(0,len(self.subsets[i])-1)])
        x = torch.stack([e[0] for e in b])
        y = torch.tensor([e[1] for e in b])
        if self.val_loader:
            val_batch = next(self.val_iter)
            val_x, val_y = val_batch[:2]
            x, y = torch.cat([x,val_x]), torch.cat([y,val_y])
        return x, y

    def compute_weights(self, rewards):
        rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)
        return rewards

    def step(self, rewards):
        assert len(self.sampler_copies) <= 2
        sampler = self.sampler_copies.pop(0)
        self.num_uses_of_partitions.scatter_add_(0, sampler.sample.cpu(), torch.ones_like(sampler.sample.cpu()))
        self.avg_batch_alignments.scatter_add_(0, sampler.sample.cpu(), rewards.cpu())
        if self.get_total_t() % 100 == 50 and self.summary_writer is not None:
            for i in range(len(sampler.p)):
                self.summary_writer.add_scalar(f'DatasetDistribution/p{i}', sampler.p[i], self.get_total_t())
            for i, x in enumerate(self.avg_batch_alignments):
                self.summary_writer.add_scalar(f'AverageAlignment/c{i}', x/self.num_uses_of_partitions[i], self.get_total_t())
        with torch.no_grad():
            rewards = rewards.to(sampler.logps.device)
            weights = self.compute_weights(rewards).detach()
        self.avg_normalized_reward.scatter_add_(0, sampler.sample.cpu(), weights.cpu())
        if self.get_total_t() % 100 == 50 and self.summary_writer is not None:
            print(self.avg_normalized_reward/self.num_uses_of_partitions)
            for i, x in enumerate(self.avg_normalized_reward):
                self.summary_writer.add_scalar(f'AverageNormalizedReward/c{i}', x/self.num_uses_of_partitions[i], self.get_total_t())

        sampler.zero_grad()
        loss = - weights.detach() @ sampler.logps / float(len(weights))

        loss.backward()
        torch.nn.utils.clip_grad_value_(sampler.parameters(), 5.)
        self.sampler.zero_grad()
        self.sampler.add_grad_of_copy(sampler)
        self.optimizer.step()

    def reset_state(self):
        del self.sampler_copies
        self.sampler_copies = []

    def get_total_t(self):
        return self.epoch * len(self) + self.t

    def __len__(self):
        return len(self.ds) // (self.bs+self.val_bs)

    def __iter__(self):
        self.t = 0
        self.epoch += 1
        del self.val_iter
        self.val_iter = iter(self.val_loader)
        return self



