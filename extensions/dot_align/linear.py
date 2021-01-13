import math

import torch
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase
from extensions.dot_align.utils import StatehandlingMeta



def get_adam_factor(opt, parameter):
    assert len(opt.param_groups) == 1
    group = opt.param_groups[0]
    beta1, beta2 = group['betas']
    state = opt.state[parameter]
    exp_avg_sq = state['exp_avg_sq']

    bias_correction2 = 1 - beta2 ** state['step']

    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
    return 1. / denom

class DotAlignLinear(BatchGradBase, StatehandlingMeta):
    def __init__(self,align_func,align_func_vec,state,align_with_next=False, opt=None):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])
        self.align_func = align_func
        self.align_func_vec = align_func_vec
        self.state = state
        self.align_with_next = align_with_next
        self.opt = opt

    def bias(self, ext, module, g_inp, g_out, bpquantities):
        comp_grad_batch = super().bias(ext, module, g_inp, g_out, bpquantities)
        grad_batch,comp_grad_batch = self.handle_state(self.state,'bias_state',module,comp_grad_batch)
        if grad_batch is None:
            return None
        if self.opt is not None:
            if isinstance(self.opt,torch.optim.Adam):
                comp_grad_batch *= get_adam_factor(self.opt, module.bias)
            elif isinstance(self.opt,torch.optim.SGD):
                pass
            else:
                raise ValueError(f'Used Optimizer type {type(self.opt)} not supported.')

        return self.align_func(grad_batch,comp_grad_batch)

    def weight(self, ext, module, g_inp, g_out, bpquantities):
        comp_mat = g_out[0].unsqueeze(-1)
        comp_d_weight = module.input0.unsqueeze(-1)
        d_weight, mat, comp_d_weight, comp_mat = self.handle_state(self.state,'weight_state',module,comp_d_weight,comp_mat)
        if d_weight is None:
            return None
        optimizer_factor = None
        if self.opt is not None:
            if isinstance(self.opt,torch.optim.Adam):
                optimizer_factor = get_adam_factor(self.opt, module.weight)
            elif not isinstance(self.opt,torch.optim.SGD):
                raise ValueError(f'Used Optimizer type {type(self.opt)} not supported.')
        ga = self.align_func_vec(d_weight,mat,comp_d_weight,comp_mat, optimizer_factor=optimizer_factor)
        return ga
