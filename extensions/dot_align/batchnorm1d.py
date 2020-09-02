from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class DotAlignBatchNorm1d(BatchGradBase):
    def __init__(self,align_func):
        super().__init__(
            derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"]
        )
        self.align_func = align_func

    def bias(self, ext, module, g_inp, g_out, bpquantities):
        grad_batch = super().bias(ext, module, g_inp, g_out, bpquantities)
        return self.align_func(grad_batch)

    def weight(self, ext, module, g_inp, g_out, bpquantities):
        grad_batch = super().weight(ext, module, g_inp, g_out, bpquantities)
        return self.align_func(grad_batch)
