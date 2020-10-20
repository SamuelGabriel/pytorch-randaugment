import itertools
import json
import logging
import math
import os
from collections import OrderedDict
import gc
import resource
import tempfile
import random



import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser
from backpack import backpack, extend, memory_cleanup
from backpack.extensions import BatchL2Grad
from extensions.dot_align import DotAlignment, DistributedDotAlignment

from RandAugment.common import get_logger, get_sum_along_batch, replace_parameters
from RandAugment.data import get_dataloaders
from RandAugment.lr_scheduler import adjust_learning_rate_resnet
from RandAugment.metrics import accuracy, Accumulator
from RandAugment.networks import get_model, num_class
from RandAugment.preprocessors import LearnedPreprocessorRandaugmentSpace, StandardCIFARPreprocessor, LearnedRandAugmentPreprocessor, LearnedPreprocessorEnsemble
from RandAugment.differentiable_preprocessor import DifferentiableLearnedPreprocessor
from warmup_scheduler import GradualWarmupScheduler
from RandAugment import google_augmentations

from RandAugment.common import add_filehandler, recursive_backpack_memory_cleanup

logger = get_logger('RandAugment')
logger.setLevel(logging.DEBUG)

def compute_preprocessor_gradients(model: nn.Module, preprocessor: nn.Module, old_model_parameters, loss_fn, generated_data, old_detached_generated_data, label):
    # preprocessor should hold gradients from train step and should not receive gradients in val step
    # model should hold recent gradients from val step

    current_model_parameters = model.state_dict()
    model.load_state_dict(old_model_parameters)
    eps = 0.01/torch.sqrt(sum([p.grad.square().sum() for p in model.parameters()]))

    for p in model.parameters():
        with torch.no_grad():
            p -= p.grad * eps

    preds = model(old_detached_generated_data)
    loss = loss_fn(preds, label)
    old_detached_generated_data.grad -= torch.autograd.grad(loss,old_detached_generated_data)[0]# negative for finite difference

    torch.autograd.backward(generated_data,-old_detached_generated_data.grad/eps) # maybe multiply with learning rate?

    model.load_state_dict(current_model_parameters)

    # Now the preprocessor gradients hold the estimated meta gradient and the weights are left as is
    # The model holds the weights from before and the gradients are left as is

def run_epoch(rank, worldsize, model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None, preprocessor=None,sec_optimizer=None):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        logging_loader = tqdm(loader, disable=tqdm_disable)
        logging_loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))
    else:
        logging_loader = loader

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    backpack_state = {}

    def call_attr_on_meta_modules(fun_name, *args, **kwargs):
        if hasattr(preprocessor, 'step'):
            getattr(preprocessor, fun_name)(*args, **kwargs)
        if hasattr(loader, 'step'):
            getattr(loader, fun_name)(*args, **kwargs)
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
        if hasattr(actual_model, 'adaptive_dropouters'):
            for ada_drop in actual_model.adaptive_dropouters:
                if hasattr(ada_drop, 'step'):
                    getattr(ada_drop, fun_name)(*args, **kwargs)

    if optimizer:
        call_attr_on_meta_modules('reset_state')
        call_attr_on_meta_modules('train')
    else:
        call_attr_on_meta_modules('eval')
    gc.collect()
    torch.cuda.empty_cache()
    print('mem usage', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    has_val_steps = C.get().get('alignment_loss', {}).get('has_val_steps', False)
    finite_difference_loss = C.get().get('finite_difference_loss', {})
    if finite_difference_loss: has_val_steps = True
    for batch in logging_loader: # logging loader might be a loader or a loader wrapped into tqdm
        data, label = batch[:2]
        steps += 1

        if preprocessor:
            data = [torchvision.transforms.ToPILImage()(ti) for ti in data]
            data = preprocessor(data,int((epoch - 1) * total_steps) + steps, validation_step=(has_val_steps and steps % 2 == 0))
        if worldsize > 1:
            data, label = data.to(rank), label.to(rank)
        else:
            data, label = data.cuda(), label.cuda()

        if optimizer and finite_difference_loss:
            if steps % 2 == 1:
                # train step
                old_parameters = model.state_dict()
                old_generated_data = data
                data = old_generated_data.detach().requires_grad_()
                old_detached_generated_data = data
                old_label = label

        if optimizer:
            optimizer.zero_grad()

        recursive_backpack_memory_cleanup(model)
        preds = model(data)
        if 'test' in desc_default:
            recursive_backpack_memory_cleanup(model)
        loss = loss_fn(preds, label)
        #print('mem usage before backward', torch.cuda.memory_allocated()/1000//1000, "MB")
        val_batch = C.get().get('val_batch',0)
        if optimizer:
            if 'alignment_loss' in C.get():
                alignment_loss_flags = C.get()['alignment_loss']
                if worldsize > 1:
                    with backpack(DistributedDotAlignment(len(data),0,backpack_state, 'cossim' == alignment_loss_flags['alignment_type'], align_with=alignment_loss_flags['align_with'])):
                        loss.backward()
                        if 'callbacks' in backpack_state:
                            for c in backpack_state['callbacks']:
                                c()
                            backpack_state['callbacks'] = []
                else:
                    with backpack(DotAlignment(len(data), val_batch, backpack_state, 'remove_me' == alignment_loss_flags['summation'],
                                               'normalized' == alignment_loss_flags['summation'],
                                               'cossim' == alignment_loss_flags['alignment_type'],
                                               align_with=alignment_loss_flags['align_with'], use_slow_version=alignment_loss_flags.get('use_slow_version',False))):
                        loss.backward()
                if ('2' not in alignment_loss_flags['align_with'] or steps > 1) and (not has_val_steps or steps % 2 == 0):
                    ga = get_sum_along_batch(model, 'grad_alignments')
                else:
                    ga = None
            elif 'unrolled_loop_loss' in C.get():
                gradients = torch.autograd.grad(loss,[p for n,p in model.named_parameters() if 'adaptive_dropouter' not in n],create_graph=True)
                #if hasattr(model, 'module'):
                #    actual_model = model.module
                #else:
                #    actual_model = model
                #if hasattr(actual_model, 'adaptive_dropouter') and hasattr(actual_model.adaptive_dropouter, 'step'):
                #    for p in actual_model.adaptive_dropouter.parameters(): p.grad = None
                if steps == 1:
                    ga = None
                    last_grads = gradients
                else:
                    curr_grads = gradients
                    ga = sum(c_g.flatten() @ l_g.flatten() for c_g,l_g in zip(curr_grads,last_grads) if c_g is not None or l_g is not None)
                    # now we can use autograd.grad with ga towards the weights we want to optimize with alignment
                    last_grads = curr_grads

                # this part is to replace parameters in both model and parameter, s.t. we do not update tensors that are part of graph
                new_parameters = []
                old_parameters = []
                for (n, p),g in zip(model.named_parameters(),gradients):
                    if 'adaptive_dropouter' in n:
                        continue
                    old_parameters.append(p)
                    new_p = torch.nn.Parameter(p.clone().detach(),requires_grad=True)
                    new_p.grad = g.detach().clone() if g is not None else None
                    new_parameters.append(new_p)
                    names = n.split('.')
                    sub_module = model
                    for n in names[:-1]:
                        sub_module = getattr(sub_module, n)
                    sub_module.register_parameter(names[-1], new_p)
                replace_parameters(optimizer,old_parameters,new_parameters)
            else:
                loss.backward()
                ga = None

            if finite_difference_loss and steps % 2 == 0: # eval step
                compute_preprocessor_gradients(model, preprocessor, old_parameters, loss_fn, old_generated_data,
                                               old_detached_generated_data, old_label)
                call_attr_on_meta_modules('step', ga)
                print('step preprocessor')
            if ga is not None:
                call_attr_on_meta_modules('step',ga)
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            if (steps-1) % C.get().get('step_optimizer_every', 1) == 0:
                print('take optimizer step')
                optimizer.step()
                if sec_optimizer is not None:
                    sec_optimizer.step()
            del ga

        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        if steps % 2 == 0:
            metrics.add('eval_top1', top1.item() * len(data) * 2) # times 2 since it is only recorded every sec step
            print('add eval top1')
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            logging_loader.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    if tqdm_disable:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'], metrics / cnt, optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(rank, worldsize, tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False):
    if not reporter:
        reporter = lambda **kwargs: 0

    if not tag or rank > 0:
        from RandAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided or rank > 0 -> no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs3/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test', 'testtrain']]

    def get_meta_optimizer_factory():
        meta_flags = C.get().get('meta_opt',{})
        if 'meta_optimizer' in meta_flags:
            mo_flags = meta_flags['meta_optimizer']
            if mo_flags['type'] == 'adam':
                def get_meta_optimizer(es_optimized_variables):
                    return torch.optim.Adam(es_optimized_variables, lr=mo_flags['lr'], betas=(mo_flags['beta1'],mo_flags['beta2']))
            elif mo_flags['type'] == 'sgd':
                def get_meta_optimizer(es_optimized_variables):
                    return torch.optim.SGD(es_optimized_variables, lr=mo_flags['lr'], momentum=mo_flags['beta1'])
            elif mo_flags['type'] == 'same_as_main':
                def get_meta_optimizer(es_optimized_variables):
                    return optim.SGD(
                        es_optimized_variables,
                        lr=C.get()['lr'],
                        momentum=C.get()['optimizer'].get('momentum', 0.9),
                        weight_decay=C.get()['optimizer']['decay'],
                        nesterov=C.get()['optimizer']['nesterov']
                    )
            else:
                raise ValueError()
        else:
            raise ValueError()
        return get_meta_optimizer
    google_augmentations.set_search_space(C.get().get('augmentation_search_space','standard'))
    max_epoch = C.get()['epoch']
    val_bs = C.get().get('val_batch',0)
    trainsampler, trainloader, validloader, testloader_, testtrainloader_, dataset_info = get_dataloaders(C.get()['dataset'], C.get()['batch']+val_bs, dataroot, test_ratio, split_idx=cv_fold, get_meta_optimizer_factory=get_meta_optimizer_factory, distributed=worldsize>1, summary_writer=writers[0])

    # create a model & an optimizer
    model = get_model(C.get()['model'], C.get()['batch'], val_bs, get_meta_optimizer_factory, num_class(C.get()['dataset']), writer=writers[0])
    if worldsize > 1:
        model = DDP(model.to(rank), device_ids=[rank])
    else:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    else:
        raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

    if 'sec_optimizer' in C.get() and C.get()['sec_optimizer']['type'] == 'adam':
        sec_optimizer = optim.Adam(model.parameters())
    else:
        sec_optimizer = None

    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    elif lr_scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.)
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    if 'preprocessor' in C.get():
        preprocessor_flags = C.get()['preprocessor']
        preprocessor_type = C.get()['preprocessor']['type']
        if preprocessor_type in ('learned_randaugmentspace','learned_random_randaugmentspace','learned_random_randaugmentspace_ensemble','learned_1x1conv'):
            importance_sampling = False
            assert not preprocessor_flags.get('online_tests_on_model')
            extra_kwargs = {}
            if 'possible_num_sequential_transforms' in preprocessor_flags:
                extra_kwargs['possible_num_sequential_transforms'] = preprocessor_flags['possible_num_sequential_transforms']

            preprocessor_classes = {'learned_randaugmentspace': LearnedPreprocessorRandaugmentSpace,
                                    'learned_random_randaugmentspace': LearnedRandAugmentPreprocessor,
                                    'learned_random_randaugmentspace_ensemble': LearnedPreprocessorEnsemble}

            if preprocessor_type in preprocessor_classes:
                image_preprocessor = preprocessor_classes[preprocessor_type](dataset_info,
                                                                              preprocessor_flags['hidden_dim'],
                                                                              get_meta_optimizer_factory(),
                                                                              C.get()['batch'],val_bs,
                                                                              entropy_alpha=preprocessor_flags['entropy_alpha'],
                                                                              scale_entropy_alpha=preprocessor_flags['scale_entropy_alpha'],
                                                                              cutout=C.get().get('cutout', 0),
                                                                              importance_sampling=importance_sampling,
                                                                              normalize_reward=preprocessor_flags.get('normalize_reward',True),
                                                                              model_for_online_tests=None,
                                                                              q_zero_init=preprocessor_flags['q_zero_init'],
                                                                              scale_embs_zero_init=preprocessor_flags.get('scale_embs_zero_init',False),
                                                                              scale_embs_zero_strength_bias=preprocessor_flags.get('scale_embs_zero_strength_bias',0.),
                                                                              q_residual=preprocessor_flags.get('q_residual',False),
                                                                              label_smoothing_rate=preprocessor_flags.get('label_smoothing_rate',0.),
                                                                              device=torch.device(rank if worldsize > 1 else 'cuda:0'), sigmax_dist=preprocessor_flags['sigmax_dist'], exploresoftmax_dist=preprocessor_flags['exploresoftmax_dist'],
                                                                              use_images_for_sampler=preprocessor_flags['use_images_for_sampler'],
                                                                              summary_writer=writers[0],
                                                                              uniaug_val=preprocessor_flags['uniaug_val'],
                                                                              old_preprocessor_val=preprocessor_flags['oldpreprocessor_val'],
                                                                              current_preprocessor_val=preprocessor_flags.get('currpreprocessor_val', False),
                                                                              aug_probs=preprocessor_flags.get('aug_probs', False),
                                                                              use_non_embedding_sampler=preprocessor_flags.get('non_embedding_sampler',False),
                                                                              ppo=preprocessor_flags.get('ppo',False),
                                                                              **extra_kwargs)
            else:
                image_preprocessor = DifferentiableLearnedPreprocessor(dataset_info,hidden_dimension=preprocessor_flags['hidden_dim'],
                                                                       optimizer_creator=get_meta_optimizer_factory(),
                                                                       cutout=C.get().get('cutout', 0),
                                                                       uniaug_val=preprocessor_flags['uniaug_val'],
                                                                       old_preprocessor_val=preprocessor_flags['oldpreprocessor_val'])

        elif preprocessor_type == 'standard_cifar':
            image_preprocessor = StandardCIFARPreprocessor(dataset_info, C.get().get('cutout', 0))
        else:
            raise NotImplementedError()
    else:
        image_preprocessor = None



    result = OrderedDict()
    epoch_start = 1
    if save_path and os.path.exists(save_path):
        logger.info('%s file found. loading...' % save_path)
        data = torch.load(save_path)
        if 'model' in data or 'state_dict' in data:
            key = 'model' if 'model' in data else 'state_dict'
            logger.info('checkpoint epoch@%d' % data['epoch'])
            if not isinstance(model, DataParallel):
                model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
            else:
                model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
            optimizer.load_state_dict(data['optimizer'])
            if data['epoch'] < C.get()['epoch']:
                epoch_start = data['epoch']
            else:
                only_eval = True
        else:
            model.load_state_dict({k: v for k, v in data.items()})
        del data
    else:
        logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
        if only_eval:
            logger.warning('model checkpoint not found. only-evaluation mode is off.')
        only_eval = False

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        if image_preprocessor: image_preprocessor.eval()
        rs = dict()
        with torch.no_grad():
            rs['train'] = run_epoch(rank, worldsize, model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0])
            #rs['valid'] = run_epoch(rank, worldsize, model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1])
            rs['test'] = run_epoch(rank, worldsize, model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        if worldsize > 1:
            trainsampler.set_epoch(epoch)

        model.train()
        if image_preprocessor: image_preprocessor.train()
        rs = dict()
        rs['train'] = run_epoch(rank, worldsize,extend(model) if 'meta_opt' in C.get() else model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=True, scheduler=scheduler, preprocessor=image_preprocessor, sec_optimizer=sec_optimizer)
        model.eval()
        if image_preprocessor: image_preprocessor.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 5 == 0 or epoch == max_epoch:
            with torch.no_grad():
                rs['testtrain'] = run_epoch(rank, worldsize, model, testtrainloader_, criterion, None, desc_default='testtrain', epoch=epoch, writer=writers[3], verbose=True, preprocessor=image_preprocessor)
                rs['test'] = run_epoch(rank, worldsize, model, testloader_, criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=True, preprocessor=image_preprocessor)


            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test', 'testtrain']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                #writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['test']['loss'], top1_valid=rs['test']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

                # save checkpoint
                if save_path:
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path)
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path.replace('.pth', '_e%d_top1_%.3f_%.3f' % (epoch, rs['train']['top1'], rs['test']['top1']) + '.pth'))

    del model

    result['top1_test'] = best_top1
    return result

def setup(rank, world_size, port_suffix):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'123{port_suffix}'

    # initialize the process group
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def spawn_process(rank, worldsize, port_suffix):
    if worldsize:
        setup(rank, worldsize, port_suffix)
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--only-eval', action='store_true')
    args = parser.parse_args()

    if worldsize:
        assert worldsize == C.get()['gpus'], "Did not specify the number of GPUs in Config with which it was started."

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    if args.save:
        add_filehandler(logger, args.save.replace('.pth', '.log'))

    #logger.info(json.dumps(C.get().conf, indent=4))

    import time
    t = time.time()
    result = train_and_eval(rank, worldsize, args.tag, args.dataroot, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.save, only_eval=args.only_eval, metric='test')
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)
    if worldsize:
        cleanup()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    port_suffix = str(random.randint(10,99))
    if world_size > 1:
        result = mp.spawn(spawn_process,
                          args=(world_size,port_suffix),
                          nprocs=world_size,
                          join=True)
    else:
        spawn_process(0, 0, None)
