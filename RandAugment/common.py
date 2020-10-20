import logging
import warnings
import random
from copy import copy
from typing import Union

import torch
from backpack import memory_cleanup
from torch.utils.checkpoint import check_backward_validity, detach_variable, get_device_states, set_device_states
from torchvision.datasets import VisionDataset, CIFAR10, CIFAR100, ImageFolder

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_filehandler(logger, filepath):
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def get_sum_along_batch(model, attribute):
    grad_list = []
    for param in model.parameters():
        ga = getattr(param, attribute, None)
        if ga is not None:
            if isinstance(ga, str):
                print(param.shape)
            grad_list.append(ga)
    return torch.stack(grad_list).sum(0)

def get_gradients(model):
    grad_list = []
    for param in model.parameters():
        grad_list.append(param.grad)
    return grad_list

def recursive_backpack_memory_cleanup(module: torch.nn.Module):
    """Remove I/O stored by backpack during the forward pass.

    Deletes the attributes created by `hook_store_io` and `hook_store_shapes`.
    """

    memory_cleanup(module)
    for m in module.modules():
        memory_cleanup(m)

def replace_parameters(optimizer, old_ps, new_ps):
    new_ps, old_ps = list(new_ps), list(old_ps)
    assert len(new_ps) == len(old_ps)
    assert len(optimizer.param_groups) == 1
    for o_p, n_p in zip(old_ps,new_ps):
        optimizer.state[n_p] = optimizer.state[o_p]
        del optimizer.state[o_p]
    optimizer.param_groups[0]['params'] = list(new_ps)


class CheckpointFunctionForSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        if True:
            ctx.preserve_rng_state = preserve_rng_state
            if preserve_rng_state:
                ctx.fwd_cpu_state = torch.get_rng_state()
                # Don't eagerly initialize the cuda context by accident.
                # (If the user intends that the context is initialized later, within their
                # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
                # we have no way to anticipate this will happen before we run the function.)
                ctx.had_cuda_in_fwd = False
                if torch.cuda._initialized:
                    ctx.had_cuda_in_fwd = True
                    ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        sample, logps, ce = outputs
        ctx.mark_non_differentiable(sample)

        return sample, logps.requires_grad_(), ce.requires_grad_()

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if True:
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                detached_inputs = detach_variable(inputs)
                with torch.enable_grad():
                    outputs = ctx.run_function(*detached_inputs)
        else:
            with torch.enable_grad():
                outputs = ctx.run_function(*inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs[1:], args[1:])
        del inputs
        del outputs
        del args
        del ctx.run_function
        return (None, None) + (None,)

def log_sigmax(logits, d):
    return torch.nn.functional.logsigmoid(logits) - torch.log(torch.sigmoid(logits).sum(d, keepdim=True))

def sigmax(logits, d):
    sigs = torch.sigmoid(logits)
    return sigs / sigs.sum(d, keepdim=True)

def relabssum(logits, d):
    abslogits = logits.abs() + .00001
    return abslogits / abslogits.sum(d, keepdim=True)

def log_relabssum(logits, d):
    return torch.log(relabssum(logits,d))

alpha = .1

def exploresoftmax(logits, d):
    return torch.softmax(logits,d) * (1.-alpha) + torch.ones_like(logits) * alpha / logits.shape[d]

class LogExpSoftmaxImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, d):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        if not isinstance(d, int):
            d = d.cpu().item()
        ctx.d = d
        ctx.save_for_backward(logits)
        return torch.log(exploresoftmax(logits,d))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        logits, = ctx.saved_tensors
        d = ctx.d
        logits = logits.clone().requires_grad_(True)
        with torch.enable_grad():
            log_sm = torch.log_softmax(logits,d)
        log_sm_grad, = torch.autograd.grad(log_sm,logits,grad_output/ \
               (1. + (alpha / (torch.softmax(logits,d)*(1.-alpha)*logits.shape[d]) )))
        del ctx.d
        return log_sm_grad, None

def log_exploresoftmax(logits, d):
    return LogExpSoftmaxImpl.apply(logits, d) # was tested against AD

def ListDataLoader(*lists, bs=None):
    num_elements = len(lists[0])
    num_shown = 0
    assert all(len(l) == num_elements for l in lists)
    assert bs is not None
    zipped_lists = list(zip(*lists))
    while True:
        if num_shown > num_elements:
            raise StopIteration
        batch = random.choices(zipped_lists,k=bs)
        num_shown += bs
        yield list(zip(*batch))

class RoundRobinDataLoader:
    def __init__(self,*dataloaders):
        '''
        All dataloaders are just restarted when ending, but the first one. If the first provided DataLoader ends we end.
        WARNING: Only supports one iterator per dataloader at any given time.
        '''
        self.loaders = dataloaders

    def __iter__(self):
        self.iterators = [iter(l) for l in self.loaders]
        self.steps = 0
        return self

    def __len__(self):
        return len(self.loaders[0])*len(self.loaders)

    def __next__(self):
        iterator_idx = self.steps % len(self.iterators)
        try:
            b = next(self.iterators[iterator_idx])
        except StopIteration:
            if iterator_idx == 0:
                raise StopIteration
            else:
                self.iterators[iterator_idx] = iter(self.loaders[iterator_idx])
                b = next(self.iterators[iterator_idx])
        self.steps += 1
        return b

def copy_and_replace_transform(ds: Union[CIFAR10, ImageFolder], transform):
    assert ds.transform is not None # make sure still uses old style transform
    new_ds = copy(ds)
    new_ds.transform = transform
    return new_ds



