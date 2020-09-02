import itertools
import json
import logging
import math
import os
from collections import OrderedDict

import torch
from torch import nn, optim
import torchvision
from torch.nn.parallel.data_parallel import DataParallel

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser
from backpack import backpack, extend, memory_cleanup
from backpack.extensions import BatchL2Grad
from extensions.dot_align import DotAlignment

from RandAugment.common import get_logger, get_sum_along_batch
from RandAugment.data import get_dataloaders
from RandAugment.lr_scheduler import adjust_learning_rate_resnet
from RandAugment.metrics import accuracy, Accumulator
from RandAugment.networks import get_model, num_class
from RandAugment.preprocessors import LearnedPreprocessorRandaugmentSpace, StandardCIFARPreprocessor
from warmup_scheduler import GradualWarmupScheduler

from RandAugment.common import add_filehandler

logger = get_logger('RandAugment')
logger.setLevel(logging.INFO)


def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None, preprocessor=None):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        loader = tqdm(loader, disable=tqdm_disable)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    backpack_state = {}
    for data, label in loader:
        steps += 1
        if preprocessor:
            data = [torchvision.transforms.ToPILImage()(ti) for ti in data]
            data = preprocessor(data,epoch - 1 + float(steps) / total_steps)
        data, label = data.cuda(), label.cuda()

        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)

        if optimizer:
            if 'alignment_loss' in C.get():
                alignment_loss_flags = C.get()['alignment_loss']
                with backpack(DotAlignment(len(data), 0, backpack_state, 'remove_me' == alignment_loss_flags['summation'],
                                           'normalized' == alignment_loss_flags['summation'],
                                           'cossim' == alignment_loss_flags['alignment_type'], align_with=alignment_loss_flags['align_with'])):
                    loss.backward()
                if '2' not in alignment_loss_flags['align_with'] or steps > 1:
                    ga = get_sum_along_batch(model, 'grad_alignments')
                else:
                    ga = None
            else:
                loss.backward()
            if hasattr(preprocessor, 'step') and ga is not None and optimizer:
                preprocessor.step(ga)
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

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


def train_and_eval(tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False):
    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, testloader_, dataset_info = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold)

    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset']))

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

    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    if not tag:
        from RandAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test']]
    def get_meta_optimizer_factory():
        meta_flags = C.get()['meta_opt']
        if 'meta_optimizer' in meta_flags:
            mo_flags = meta_flags['meta_optimizer']
            if mo_flags['type'] == 'adam':
                def get_meta_optimizer(es_optimized_variables):
                    return torch.optim.Adam(es_optimized_variables, lr=mo_flags['lr'], betas=(mo_flags['beta1'],mo_flags['beta2']))
            elif mo_flags['type'] == 'sgd':
                def get_meta_optimizer(es_optimized_variables):
                    return torch.optim.SGD(es_optimized_variables, lr=mo_flags['lr'], momentum=mo_flags['beta1'])
            else:
                raise ValueError()
        else:
            raise ValueError()
        return get_meta_optimizer
    if 'preprocessor' in C.get():
        preprocessor_flags = C.get()['preprocessor']
        preprocessor_type = C.get()['preprocessor']['type']
        if preprocessor_type == 'learned_randaugmentspace':
            importance_sampling = False
            assert not preprocessor_flags.get('online_tests_on_model')
            image_preprocessor = LearnedPreprocessorRandaugmentSpace(dataset_info,
                                                                     preprocessor_flags['hidden_dim'],
                                                                     get_meta_optimizer_factory(),
                                                                     preprocessor_flags['entropy_alpha'],
                                                                     scale_entropy_alpha=preprocessor_flags['scale_entropy_alpha'],
                                                                     cutout=C.get().get('cutout', 0),
                                                                     importance_sampling=importance_sampling,
                                                                     normalize_reward=preprocessor_flags.get('normalize_reward',True),
                                                                     model_for_online_tests=None,
                                                                     q_zero_init=preprocessor_flags['q_zero_init'],
                                                                     scale_embs_zero_init=preprocessor_flags.get('scale_embs_zero_init',False),
                                                                     q_residual=preprocessor_flags.get('q_residual',False),
                                                                     label_smoothing_rate=preprocessor_flags.get('label_smoothing_rate',0.),
                                                                     device=torch.device('cuda:0'),
                                                                     summary_writer=writers[0])
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
    if 'meta_opt' in C.get():
        model = extend(model)

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        if image_preprocessor: image_preprocessor.eval()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0])
        rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1])
        rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        model.train()
        if image_preprocessor: image_preprocessor.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=True, scheduler=scheduler, preprocessor=image_preprocessor)
        model.eval()
        if image_preprocessor: image_preprocessor.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 5 == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=True, preprocessor=image_preprocessor)
            rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=True, preprocessor=image_preprocessor)

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

                # save checkpoint
                if save_path:
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path)
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path.replace('.pth', '_e%d_top1_%.3f_%.3f' % (epoch, rs['train']['top1'], rs['test']['top1']) + '.pth'))

    del model

    result['top1_test'] = best_top1
    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--only-eval', action='store_true')
    args = parser.parse_args()

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    if args.save:
        add_filehandler(logger, args.save.replace('.pth', '.log'))

    logger.info(json.dumps(C.get().conf, indent=4))

    import time
    t = time.time()
    result = train_and_eval(args.tag, args.dataroot, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.save, only_eval=args.only_eval, metric='test')
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)
