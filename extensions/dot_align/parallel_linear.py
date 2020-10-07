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
        comparison_grad_batch = super().bias(ext, module, g_inp, g_out, bpquantities)
        grad_batch,comparison_grad_batch = self.handle_state(self.state, 'bias_state', module, comparison_grad_batch)
        if grad_batch is None:
            return None
        module.grad_batch, module.comparison_grad_batch = grad_batch, comparison_grad_batch
        def hook():
            module.bias.grad_alignments = self.align_func(module.grad_batch, module.comparison_grad_batch, curr_grad=module.bias.grad)
            del module.grad_batch, module.comparison_grad_batch
        self.add_callback(self.state, hook)

        ga = 'overwrite me'
        #ga =self.align_func(grad_batch, comparison_grad_batch)
        return ga


    def weight(self, ext, module, g_inp, g_out, bpquantities):
        comp_mat = g_out[0].unsqueeze(-1)
        comp_d_weight = module.input0.unsqueeze(-1)
        d_weight, mat, comp_d_weight, comp_mat = self.handle_state(self.state,'weight_state',module,comp_d_weight,comp_mat)
        if d_weight is None:
            return None
        module.d_weight, module.mat, module.comp_d_weight, module.comp_mat = d_weight, mat, comp_d_weight, comp_mat
        def hook():
            module.weight.grad_alignments = self.align_func_vec(module.d_weight, module.mat, module.comp_d_weight, module.comp_mat, curr_grad=module.weight.grad)
            del module.d_weight, module.mat, module.comp_d_weight, module.comp_mat
        self.add_callback(self.state, hook)

        ga = 'overwrite me'
        #ga = self.align_func_vec(d_weight,mat,comp_d_weight,comp_mat)
        return ga
