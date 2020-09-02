import torch
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase
from extensions.dot_align.utils import StatehandlingMeta

class DotAlignLinear(BatchGradBase, StatehandlingMeta):
    def __init__(self,align_func,align_func_vec,state,align_with_next=False):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])
        self.align_func = align_func
        self.align_func_vec = align_func_vec
        self.state = state
        self.align_with_next = align_with_next

    def bias(self, ext, module, g_inp, g_out, bpquantities):
        comp_grad_batch = super().bias(ext, module, g_inp, g_out, bpquantities)
        grad_batch,comp_grad_batch = self.handle_state(self.state,'bias_state',module,comp_grad_batch)
        if grad_batch is None:
            return None
        return self.align_func(grad_batch,comp_grad_batch)

    def weight(self, ext, module, g_inp, g_out, bpquantities):
        comp_mat = g_out[0].unsqueeze(-1)
        comp_d_weight = module.input0.unsqueeze(-1)
        d_weight, mat, comp_d_weight, comp_mat = self.handle_state(self.state,'weight_state',module,comp_d_weight,comp_mat)
        if d_weight is None:
            return None
        ga = self.align_func_vec(d_weight,mat,comp_d_weight,comp_mat)
        return ga
