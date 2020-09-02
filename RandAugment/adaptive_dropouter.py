import torch
from torch import nn

from copy import deepcopy

class Sampler(nn.Module):
    def __init__(self, num_dropouts, hidden_dimension):
        super().__init__()
        self.get_keep_logits = nn.Sequential(
            nn.Linear(num_dropouts,hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension,num_dropouts),
        )
    def forward(self, state):
        l = self.get_keep_logits(state)
        sample = torch.bernoulli(torch.sigmoid(l)).to(torch.bool)
        true_logps = torch.nn.functional.logsigmoid(l)
        false_logps = -l - torch.log(1.+torch.exp(-l)) # log(1-sig(l))
        self.logps = torch.where(sample,true_logps,false_logps).sum(1)
        return sample

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad

class AdaptiveDropouter(nn.Module):
    def __init__(self, num_dropouts, hidden_dimension, optimizer_creator, summary_writer=None):
        super().__init__()
        print('using learned preprocessor')
        self.normalize_reward = True
        self.summary_writer = summary_writer

        self.sampler = Sampler(num_dropouts,hidden_dimension)
        self.sampler_copies = []

        self.optimizer = optimizer_creator(self.sampler.parameters())
        self.t = -1

    def forward(self, orig_hiddens):
        self.t += 1
        # hiddens shall have size None x num_dropouts
        hiddens = orig_hiddens.detach()
        if self.training:
            sampler = deepcopy(self.sampler)
            self.sampler_copies.append(sampler)

            keep_mask = sampler(hiddens)

            self.write_summary(sampler, keep_mask, self.t)
        else:
            keep_mask = torch.ones_like(hiddens,dtype=torch.bool)
        return keep_mask.detach()*orig_hiddens

    def compute_weights(self, rewards):
        if self.normalize_reward:
            rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)
        return rewards

    def step(self, rewards):
        sampler = self.sampler_copies.pop(0)  # pops the oldest state first (queue-style)
        if self.t % 100 == 50 and self.summary_writer is not None and self.training:
            self.summary_writer.add_scalar(f'Alignment/AverageAlignment', rewards.mean(), self.t)
            self.summary_writer.add_scalar(f'Alignment/MaxAlignment', rewards.max(), self.t)
            self.summary_writer.add_scalar(f'Alignment/MinAlignment', rewards.min(), self.t)

        with torch.no_grad():
            rewards = rewards.to(sampler.logps.device)
            weights = self.compute_weights(rewards).detach()

        sampler.zero_grad()
        loss = -sampler.logps @ weights.detach() / float(len(weights))

        loss.backward()
        torch.nn.utils.clip_grad_value_(sampler.parameters(), 5.)
        self.sampler.zero_grad()
        self.sampler.add_grad_of_copy(sampler)
        self.optimizer.step()
        del sampler

    def write_summary(self, sampler, keep_mask, step):
        if step % 100 == 0 and self.summary_writer is not None and self.training:
            with torch.no_grad():
                self.summary_writer.add_scalar(f'Dropouter/average_logp', sampler.logps.mean(), step)
                self.summary_writer.add_scalar(f'Dropouter/keep_share', keep_mask.float().mean(), step)
