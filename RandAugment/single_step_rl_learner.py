import torch
from torch import nn

from collections import defaultdict, Counter
from typing import NamedTuple, List
from copy import deepcopy
from RandAugment.common import get_sum_along_batch, get_gradients, sigmax, log_sigmax, recursive_backpack_memory_cleanup, to_gradient_copy, Gradients



class SamplingState(NamedTuple):
    logit_network: nn.Module
    sample: torch.Tensor
    logits: torch.Tensor
    flip_inds: torch.Tensor
    ps: torch.Tensor


class SingleStepRLLearner(nn.Module):
    def __init__(self, sampling_network, optimizer_creator, entropy_alpha=0.0, pair_estimate=False, dist='bernoulli'):
        super().__init__()
        assert dist in ('bernoulli', 'categorical', 'categorical_with_sigmax')

        self.logit_network = sampling_network
        self.sampling_states: List[SamplingState] = []
        self.optimizer = optimizer_creator(self.logit_network.parameters())

        self.pair_estimate = pair_estimate
        self.entropy_alpha = entropy_alpha
        self.dist = dist

        self.t = -1

    def forward(self, inputs=None):
        # Returns a sample, and ps, which should show what weight each example has in the distribution compared to the others.
        if not self.training:
            print("WARNING: Sampling during inference.")
        self.t += 1  # starts at -1, so is zero-based from the last reset/init

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach()

        logit_network = deepcopy(self.logit_network)
        # self.logit_network_copies.append(logit_network)

        logits = logit_network(inputs)
        dist = self.get_dist(logits)

        sample = dist.sample()

        ps = torch.ones(len(inputs), device=logits.device) / len(inputs)
        flip_inds = None

        if self.pair_estimate:
            assert (inputs.view(len(sample) // 2, 2, -1)[:, 0] == inputs.view(len(sample) // 2, 2, -1)[:, 1]).all(), \
                'For pair estimate, inputs for pairs of neighbor examples should be the same.'

            # In the following we change ps, sample and optinally flip_inds for data transfer to step method
            if isinstance(dist, torch.distributions.Bernoulli):
                # shape expectation: [bs,..]

                flattened_sample = sample.view(len(sample), -1)

                grouped_sample = flattened_sample.view(-1, 2, flattened_sample.shape[1])
                grouped_sample[:, 1, :] = grouped_sample[:, 0, :]

                flip_inds = torch.randint(high=flattened_sample.shape[1], size=(len(grouped_sample),))

                grouped_sample[:, 1, :][torch.arange(len(grouped_sample)), flip_inds] = 1
                grouped_sample[:, 0, :][torch.arange(len(grouped_sample)), flip_inds] = 0

                flattened_ps = dist.probs.view(len(sample), -1)
                flip_ps = flattened_ps.view(-1, 2, flattened_ps.shape[1])[:, 0, :][
                    torch.arange(len(grouped_sample)), flip_inds]
                neg_flip_ps = 1. - flip_ps

                ps = torch.stack([neg_flip_ps.unsqueeze(1), flip_ps.unsqueeze(1)], 1).flatten().detach() / len(
                    neg_flip_ps)

            else:
                # logits shape expectation: [bs,num canonical choices]
                assert len(logits.shape) == 2

                dist_probs = dist.probs.view(-1, 2, dist.probs.shape[1])[:, 0]  # (bs//2, num_choices)
                # take only every second, since we assume neighbors to be equal

                # create permutation for each batch-example
                permutations = torch.rand(*dist_probs.shape).argsort(dim=1)  # (bs//2, num_choices)

                # write distribution over pairs
                permuted_probs = dist_probs.gather(1, permutations)
                paired_probs = permuted_probs.view(len(permuted_probs), -1, 2).sum(-1)
                pair_dist = self.get_dist(probs=paired_probs)

                permutation_pairs = permutations.view(len(dist_probs), -1, 2)

                # sample pairs
                pair_sample_inds = pair_dist.sample()  # (bs//2,)
                pair_sample = permutation_pairs[torch.arange(len(permutation_pairs)), pair_sample_inds]
                # shape: (bs//2, 2)

                # create prob for each pair member
                _sample_probs = dist_probs.gather(1, pair_sample)  # shape: (bs//2,2)
                sample_probs = _sample_probs / _sample_probs.sum(1, keepdim=True)
                # print('paired_probs', paired_probs, 'permuted_probs', permuted_probs, 'permutations', permutations, 'dist_prob', dist_probs)

                # write to outer-scope variables
                # Important: need to detach ps!
                ps = sample_probs.flatten().detach() / len(dist_probs)
                flip_inds = permutations
                sample = pair_sample.flatten()

                # For debugging in two class case:
                # sample = torch.tensor([0,1]).unsqueeze(0).repeat(logits.shape[0]//2,1).flatten()
                # ps = dist.probs[torch.arange(len(sample)),sample] / (len(dist.probs) // 2)
                # print(ps.sum())
                # print('sample,ps',sample,ps)

        recursive_backpack_memory_cleanup(logit_network)
        self.logit_network.load_state_dict(
            logit_network.state_dict())  # this is done because the running averages of bn's are now updated here.

        self.sampling_states.append(SamplingState(logit_network, sample, logits, flip_inds, ps))

        return sample, ps

    def step(self, rewards, curr_lr_factor=None):
        # this function does not normalize rewards
        assert len(self.sampling_states) <= 2
        logit_network, sample, logits, flip_inds, ps = self.sampling_states.pop(
            0)  # pops the oldest state first (queue-style)

        dist = self.get_dist(logits)

        rewards = rewards.to(logits.device)

        logit_network.zero_grad()

        logps = dist.log_prob(sample)

        if self.pair_estimate:
            if isinstance(dist, torch.distributions.Bernoulli):
                true_logps = dist.log_prob(torch.ones_like(dist.probs))
                false_logps = dist.log_prob(torch.zeros_like(dist.probs))
                even_true_logps = true_logps.view(len(true_logps) // 2, 2, -1)[:, 0, :]
                even_false_logps = false_logps.view(len(true_logps) // 2, 2, -1)[:, 0, :]
                flip_true_logps = even_true_logps[torch.arange(len(even_true_logps)), flip_inds]
                flip_false_logps = even_false_logps[torch.arange(len(even_false_logps)), flip_inds]

                logps = torch.stack([flip_false_logps.unsqueeze(1), flip_true_logps.unsqueeze(1)], 1).flatten()
            else:
                if True:
                    permutations = flip_inds  # (bs//2, num_choices)
                    pair_sample = sample.view(permutations.shape[0], 2)

                    # one can use the logits straight instead of the probs: better numerical stability
                    dist_logits = dist.logits.view(-1, 2, dist.probs.shape[1])[:, 0]  # (bs//2, num_choices)

                    # permuted_probs = dist_probs.gather(1,permutations)
                    # in_pair_probs = permuted_probs.view(len(permuted_probs),-1,2)
                    in_pair_logits = dist_logits.gather(1, pair_sample)
                    in_pair_dist = self.get_dist(logits=in_pair_logits)

                    logps = in_pair_dist.logits.flatten()  # these logits are normalized, unlike the logits from before

                    # even_logps = in_pair_dist.log_prob(torch.zeros(len(in_pair_logits),dtype=torch.long))
                    # odd_logps = in_pair_dist.log_prob(torch.ones(len(in_pair_logits),dtype=torch.long))

                    # logps = torch.stack([even_logps.unsqueeze(1),odd_logps.unsqueeze(1)],1).flatten()

        loss = - (rewards.detach() * ps.detach()) @ logps.view(len(rewards), -1).sum(1)
        if self.entropy_alpha:
            loss -= (self.entropy_alpha / float(len(rewards))) * dist.entropy().sum()

        loss.backward()

        torch.nn.utils.clip_grad_value_(logit_network.parameters(), 5.)
        self.logit_network.zero_grad()
        self.add_grad_of_copy(logit_network)

        self.opt_step(curr_lr_factor)
        for p in self.logit_network.parameters():
            p.grad = None

    def opt_step(self, curr_lr_factor):
        # Includes both the actual step, and lr scaling, if curr_lr_factor is not None
        assert len(self.optimizer.param_groups) == 1
        base_lr = self.optimizer.param_groups[0]['lr']
        assert isinstance(base_lr, float), type(base_lr)
        if curr_lr_factor is not None:
            self.optimizer.param_groups[0]['lr'] *= curr_lr_factor

        self.optimizer.step()
        self.optimizer.param_groups[0]['lr'] = base_lr

    def get_dist(self, logits=None, probs=None):
        assert (logits is None) != (probs is None), 'Either provide probs or logits.'
        if self.dist == 'bernoulli':
            # We have to sample from only one logit.
            # In this case we use Sigmoid.
            return torch.distributions.Bernoulli(logits=logits, probs=probs)
        elif self.dist == 'categorical':
            # We have multiple logits, thus we use
            # softmax.
            return torch.distributions.Categorical(logits=logits, probs=probs)
        elif self.dist == 'categorical_with_sigmax':
            def p_func(x, d):
                s = torch.sigmoid(x)
                return s / (torch.sum(s, d, keepdim=True) + .0001)

            return torch.distributions.Categorical(probs=probs if logits is None else p_func(logits, -1))
        else:
            raise NotImplementedError

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.logit_network.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad

    def reset_state(self):
        del self.sampling_states
        self.sampling_states = []
        self.t = -1





