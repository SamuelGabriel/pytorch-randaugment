import logging
import warnings
import torch
from backpack import memory_cleanup

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
    assert len(optimizer.param_groups) == 1
    for o_p, n_p in zip(old_ps,new_ps):
        optimizer.state[n_p] = optimizer.state[o_p]
        del optimizer.state[o_p]
    optimizer.param_groups[0]['params'] = list(new_ps)
