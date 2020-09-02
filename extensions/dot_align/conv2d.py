import torch
from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase
from backpack.utils.ein import eingroup
from backpack.utils import conv as convUtils
from extensions.dot_align.utils import StatehandlingMeta


class DotAlignConv2d(BatchGradBase, StatehandlingMeta):
    def __init__(self,align_func,align_func_vec,align_func_conv,state,align_with_next=False):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])
        self.align_func = align_func
        self.align_func_vec = align_func_vec
        self.align_func_conv = align_func_conv
        self.state = state
        self.align_with_next = align_with_next

    def bias(self, ext, module, g_inp, g_out, bpquantities):
        comparison_grad_batch = super().bias(ext, module, g_inp, g_out, bpquantities)
        grad_batch,comparison_grad_batch = self.handle_state(self.state, 'bias_state', module, comparison_grad_batch)
        if grad_batch is None:
            return None
        return self.align_func(grad_batch, comparison_grad_batch)

    def weight(self, ext, module, g_inp, g_out, bpquantities):
        o_input0, o_g_out, c_input0, c_g_out = self.handle_state(self.state,'weight_state', module, module.input0, g_out)
        if o_input0 is None:
            return None
        if self.align_func_conv:
            module.o_input0, module.o_g_out, module.c_input0, module.c_g_out = o_input0.clone().detach(), tuple(t.clone().detach() for t in o_g_out), c_input0.clone().detach(), tuple(t.clone().detach() for t in c_g_out)
            if not hasattr(module, 'backward_weight_hook'):
                def hook(grad):
                    module.weight.grad_alignments = self.align_func_conv(module.o_input0, module.o_g_out, module.c_input0, module.c_g_out,
                                                                         module, curr_grad=grad)
                    del module.o_input0, module.o_g_out, module.c_input0, module.c_g_out
                module.backward_weight_hook = module.weight.register_hook(hook)

            #ga = self.align_func_conv(o_input0, o_g_out, c_input0, c_g_out, module)
            ga = None
        else:
            comp_X, comp_dE_dY = convUtils.get_weight_gradient_factors(
                c_input0, c_g_out[0], module
            )
            X, dE_dY = convUtils.get_weight_gradient_factors(
                o_input0, o_g_out[0], module
            )

            ga = self.align_func_vec(X, dE_dY, comp_X, comp_dE_dY)
        if False:
            print('comp cinput0',c_input0.flatten())
            comp_X, comp_dE_dY = convUtils.get_weight_gradient_factors(
                c_input0, c_g_out[0], module
            )
            X, dE_dY = convUtils.get_weight_gradient_factors(
                o_input0, o_g_out[0], module
            )

            comp_ga = self.align_func_vec(X, dE_dY, comp_X, comp_dE_dY)
            print('abs avg diff', (comp_ga-ga).abs().mean())
            print('rel avg diff', ((comp_ga-ga)/comp_ga).abs().mean())

        return ga
