import torch
from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase
from backpack.utils.ein import eingroup
from backpack.utils import conv as convUtils
from extensions.dot_align.utils import StatehandlingMeta


class DotAlignConv2d(BatchGradBase, StatehandlingMeta):
    def __init__(self,align_func,align_func_vec,state,align_with_next=False):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])
        self.align_func = align_func
        self.align_func_vec = align_func_vec
        self.align_func_conv = None
        self.state = state
        self.align_with_next = align_with_next

    def bias(self, ext, module, g_inp, g_out, bpquantities):
        comparison_grad_batch = super().bias(ext, module, g_inp, g_out, bpquantities)
        grad_batch,comparison_grad_batch = self.handle_state(self.state, 'bias_state', module, comparison_grad_batch)
        if grad_batch is None:
            return None
        return self.align_func(grad_batch, comparison_grad_batch)

    def weight(self, ext, module, g_inp, g_out, bpquantities):
        o_input0, o_g_out0, c_input0, c_g_out0 = self.handle_state(self.state,'weight_state', module, module.input0, g_out[0])
        if o_input0 is None:
            return None
        if self.align_func_conv:
            ga = self.align_func_conv(o_input0, o_g_out0, c_input0, c_g_out0, module)
        else:
            comp_X, comp_dE_dY = convUtils.get_weight_gradient_factors(
                c_input0, c_g_out0, module
            )
            X, dE_dY = convUtils.get_weight_gradient_factors(
                o_input0, o_g_out0, module
            )

            ga = self.align_func_vec(X,dE_dY,comp_X,comp_dE_dY)
        return ga
