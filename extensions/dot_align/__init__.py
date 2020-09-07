import torch
from torch.nn import BatchNorm1d, Conv2d, Linear
from torch.nn import functional as F

from backpack.extensions.backprop_extension import BackpropExtension
from .conv2d_derivatives import Conv2DDerivatives

from . import batchnorm1d, conv2d, linear, utils

from opt_einsum import contract


class ComputeAlignment():
    def __init__(self,bs,val_bs,remove_me_summation,normalized_summation,cossim,compare_to_difference):
        self.bs = bs
        self.val_bs = val_bs
        self.normalized_summation = normalized_summation
        self.compare_to_difference = compare_to_difference
        if remove_me_summation:
            self.grad_alignment = lambda g0,g0_matrix,g_matrix: contract('bs,bs->b',g_matrix, g0-g_matrix)
        elif cossim:
            self.grad_alignment = lambda g0,g0_matrix,g_matrix: torch.cosine_similarity(g_matrix,g0.unsqueeze(0),dim=1)
        else:
            self.grad_alignment = lambda g0,g0_matrix,g_matrix: g_matrix @ g0
    def __call__(self, batch_grads, comparison_batch_grads):
        batch_grad_vecs = batch_grads.flatten(start_dim=1)
        comparison_grad_vecs = comparison_batch_grads.flatten(start_dim=1)
        if self.val_bs:
            comparison_grad_vecs = comparison_grad_vecs[self.bs-self.val_bs:]
            comparison_this_batch_grad_vecs = batch_grad_vecs[self.bs-self.val_bs:]
            batch_grad_vecs = batch_grad_vecs[:self.bs-self.val_bs]
        else:
            comparison_this_batch_grad_vecs = batch_grad_vecs
        if self.normalized_summation:
            comparison_grad_vec = torch.sum(comparison_grad_vecs/torch.norm(comparison_grad_vecs,dim=1).unsqueeze_(1),dim=0)
        else:
            comparison_grad_vec = comparison_grad_vecs.sum(dim=0)
        if self.compare_to_difference:
            if self.normalized_summation:
                c = torch.sum(comparison_this_batch_grad_vecs/torch.norm(comparison_this_batch_grad_vecs,dim=1).unsqueeze_(1),dim=0)
            else:
                c = comparison_this_batch_grad_vecs.sum(dim=0)
            comparison_grad_vec = comparison_grad_vec - c
        ga = self.grad_alignment(comparison_grad_vec,comparison_grad_vecs,batch_grad_vecs)
        return ga

class ComputeAlignmentVec():
    """This is a vectorized implementation of the above. Instead of individual gradients it expects
    the building matrices of these vectors: X, dE_dY of shape BS x w1 x p and BS x w2 x p, where the weight has w1 x w2,
    and p is an extra dimension used to vectorize convolutions into this form, as is done by BACKPACK.
    """
    def __init__(self,bs,val_bs,remove_me_summation,normalized_summation,cossim,compare_to_difference):
        self.bs = bs
        self.val_bs = val_bs
        self.normalized_summation = normalized_summation
        self.remove_me_summation = remove_me_summation
        self.cossim = cossim
        self.compare_to_difference = compare_to_difference
    
    def grad_alignment(self,comparison_grad, X, dE_dY):
        sums = contract("nai,oa->noi", X, comparison_grad)
        dot_prod = contract("noi,noi->n",dE_dY,sums)
        if self.remove_me_summation or self.cossim:
            squared_grad_norms = contract("nml,nkl,nmi,nki->n", dE_dY, X, dE_dY, X)
            if self.remove_me_summation:
                return dot_prod - squared_grad_norms
            else:
                squared_grad_norm = (comparison_grad**2).sum()
                return dot_prod/(squared_grad_norm*squared_grad_norms).sqrt()
        return dot_prod

    def __call__(self, X, dE_dY, comparison_X, comparison_dE_dY):
        if self.val_bs:
            comparison_X = comparison_X[self.bs-self.val_bs:]
            comparison_dE_dY = comparison_dE_dY[self.bs-self.val_bs:]
            X = X[:self.bs-self.val_bs]
            dE_dY = dE_dY[:self.bs-self.val_bs]
            comparison_this_X = X[self.bs-self.val_bs:]
            comparison_this_dE_dY = dE_dY[self.bs-self.val_bs:]
        else:
            comparison_this_X = X
            comparison_this_dE_dY = dE_dY

        if self.normalized_summation:
            squared_grad_norms = contract("nml,nkl,nmi,nki->n", comparison_dE_dY, comparison_X, comparison_dE_dY, comparison_X)
            comparison_X = comparison_X/squared_grad_norms.sqrt().unsqueeze_(1).unsqueeze_(1)
        comparison_grad = contract("noi,nai->oa", comparison_dE_dY, comparison_X)
        if self.compare_to_difference:
            if self.normalized_summation:
                squared_grad_norms = contract("nml,nkl,nmi,nki->n", comparison_this_dE_dY, comparison_this_X, comparison_this_dE_dY, comparison_this_X)
                comparison_this_X = comparison_this_X/squared_grad_norms.sqrt().unsqueeze_(1).unsqueeze_(1)
            comparison_grad = comparison_grad - contract("noi,nai->oa", comparison_this_dE_dY, comparison_this_X)
        #print('comp comp grad', comparison_grad.numel(), comparison_grad.flatten())

        ga = self.grad_alignment(comparison_grad, X, dE_dY)
        return ga

class ComputeAlignmentConv():
    """This is a vectorized implementation of the above. Instead of individual gradients it expects
    the building matrices of these vectors: X, dE_dY of shape BS x w1 x p and BS x w2 x p, where the weight has w1 x w2,
    and p is an extra dimension used to vectorize convolutions into this form, as is done by BACKPACK.
    """

    def __init__(self, bs, val_bs, remove_me_summation, normalized_summation, cossim, compare_to_difference):
        self.bs = bs
        self.val_bs = val_bs
        self.normalized_summation = normalized_summation
        self.remove_me_summation = remove_me_summation
        self.cossim = cossim
        self.compare_to_difference = compare_to_difference
        assert not remove_me_summation and not normalized_summation and not cossim, "This requires a little implementation effort, still."

    def grad_alignment(self, comparison_grad, X, dE_dY, conv):
        dot_prod = contract('bcwh,bcwh->b',conv(X,comparison_grad),dE_dY)
        return dot_prod

    def __call__(self, X, dE_dY, comparison_X, comparison_dE_dY, module, curr_grad=None):
        # It should work for other cases, but rather first test it.
        assert len(comparison_dE_dY) == 1 and len(dE_dY) == 1
        dE_dY = dE_dY[0]
        comparison_dE_dY = comparison_dE_dY[0]
        if module.dilation != (1,1):
            print(f'Dilation {module.dilation} not supported, I think.')
            exit
        if module.groups != 1:
            print(f'Number groups {module.groups} not supported, I think.')
            exit
        conv = lambda x,w: torch.nn.functional.conv2d(x,w,padding=module.padding,stride=module.stride)
        if self.val_bs:
            comparison_X = comparison_X[self.bs - self.val_bs:]
            comparison_dE_dY = comparison_dE_dY[self.bs - self.val_bs:]
            X = X[:self.bs - self.val_bs]
            dE_dY = dE_dY[:self.bs - self.val_bs]
            comparison_this_X = X[self.bs - self.val_bs:]
            comparison_this_dE_dY = dE_dY[self.bs - self.val_bs:]
        else:
            comparison_this_X = X
            comparison_this_dE_dY = dE_dY

        #print('start hand-compute gradient')
        #comp_ga = Conv2DDerivatives().weight_jac_t_mat_prod(module,comparison_X,(comparison_dE_dY,),comparison_dE_dY,sum_batch=True)
        #print('end hand-compute gradient')
        #comparison_grad = torch.nn.grad.conv2d_weight(comparison_X,module.weight.shape,comparison_dE_dY)
        def get_grad(X,dE_dY):
            with torch.enable_grad():
                return torch.autograd.grad(conv(X, module.weight), module.weight, dE_dY)[0]
        if curr_grad is not None and not self.val_bs:
            comparison_grad = curr_grad
        else:
            comparison_grad = get_grad(comparison_X,comparison_dE_dY)
        if self.compare_to_difference:
            comparison_grad = comparison_grad - get_grad(comparison_this_X,comparison_this_dE_dY)
        #print('comp grad', comparison_grad.numel(), comparison_grad.flatten())
        ga = self.grad_alignment(comparison_grad, X, dE_dY, conv)
        return ga


class DotAlignment(BackpropExtension):
    """Alignment of individual gradients for each sample in a minibatch.

    Stores the output in ``grad_batch`` as a ``[N x ...]`` tensor,
    where ``N`` batch size and ``...`` is the shape of the gradient.

    Note: beware of scaling issue
        The `individual gradients` depend on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``BatchGrad`` will return

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.

    The concept of individual gradients is only meaningful if the
    objective is a sum of independent functions (no batchnorm).

    """

    @utils.run_once
    def warn(self,val_bs):
        print("Warn: BatchNorm bias/weight are not compared.")
        if val_bs:
            print("Oooh! Validation batch is not tested!")

    def __init__(self,bs,val_bs,state,remove_me_summation,normalized_summation,cossim,align_with='1',use_slow_version=False):
        assert align_with in ('1','2','2-1')
        self.warn(val_bs)
        alignment_function = ComputeAlignment(bs,val_bs,remove_me_summation,normalized_summation,cossim,align_with=='2-1')
        alignment_function_vec = ComputeAlignmentVec(bs,val_bs,remove_me_summation,normalized_summation,cossim,align_with=='2-1')
        alignment_function_conv = ComputeAlignmentConv(bs,val_bs,remove_me_summation,normalized_summation,cossim,align_with=='2-1')
        super().__init__(
            savefield="grad_alignments",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.DotAlignLinear(alignment_function,alignment_function_vec,state,align_with_next='2' in align_with),
                Conv2d: conv2d.DotAlignConv2d(alignment_function,alignment_function_vec,None if use_slow_version else alignment_function_conv,state,align_with_next='2' in align_with),
                #BatchNorm1d: batchnorm1d.DotAlignBatchNorm1d(alignment_function),
            },
        )
