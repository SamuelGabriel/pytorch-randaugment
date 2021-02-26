import torch

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
# from torchvision import models

from RandAugment.networks.resnet import ResNet
from RandAugment.networks.pyramidnet import PyramidNet
from RandAugment.networks.shakeshake.shake_resnet import ShakeResNet
from RandAugment.networks.wideresnet import WideResNet
from RandAugment.networks.shakeshake.shake_resnext import ShakeResNeXt
from RandAugment.networks.convnet import SeqConvNet
from RandAugment.networks.mlp import MLP
from RandAugment.common import apply_weightnorm

from RandAugment.meta_state_adaption import AdaptiveDropouter, Modulator


# example usage get_model(
def get_model(conf, bs, val_bs, optimizer_creator_factory, num_class=10, writer=None):
    name = conf['type']
    assert not ('adaptive_dropouter' in conf and 'adaptive_modulator' in conf)
    if 'adaptive_dropouter' in conf:
        assert name in ('wresnet28_10','wresnet40_2','miniconvnet','mlp')
        ad_conf = conf['adaptive_dropouter']

        if ad_conf['simple_dropout']:
            ad_creators = (lambda w: torch.nn.Dropout(p=1.-ad_conf['target_p']), lambda w: torch.nn.Dropout(p=1.-ad_conf['target_p']))
        else:
            ad_creators = (lambda w: AdaptiveDropouter(w, ad_conf['hidden_size'], optimizer_creator_factory(), bs, val_bs, cross_entropy_alpha=ad_conf['cross_entropy_alpha'], target_p=ad_conf['target_p'], out_bias=ad_conf['out_bias'], relu=ad_conf['relu'], inference_dropout=ad_conf.get('inference_dropout', False), scale_by_p=ad_conf.get('scale_by_p', False), batch_norm=ad_conf['batch_norm'], ppo=ad_conf['ppo'], tanh=ad_conf['tanh'], scale_lr_with_outer_opt=ad_conf.get('scale_lr_with_outer_opt', False), train_only_bias=ad_conf.get('train_only_bias', False), share_prob=ad_conf.get('share_prob', True), summary_writer=writer).cuda(),
                           lambda planes, kernel_size, stride, padding: AdaptiveDropouter((planes, kernel_size, stride, padding), ad_conf['hidden_size'], optimizer_creator_factory(), bs,
                                                       val_bs, cross_entropy_alpha=ad_conf['cross_entropy_alpha'],
                                                       target_p=ad_conf['target_p'], out_bias=ad_conf['out_bias'],
                                                       relu=ad_conf['relu'],
                                                       inference_dropout=ad_conf.get('inference_dropout', False),
                                                       scale_by_p=ad_conf.get('scale_by_p', False),
                                                       batch_norm=ad_conf['batch_norm'],
                                                       scale_lr_with_outer_opt=ad_conf.get('scale_lr_with_outer_opt', False),
                                                       summary_writer=writer).cuda())
        if not ad_conf.get('conv_dropout', False):
            ad_creators = (ad_creators[0], None)
        else:
            assert not ad_conf['ppo']
            assert not ad_conf['tanh']
    elif 'adaptive_modulator' in conf:
        assert name in ('wresnet28_10',)
        ad_conf = conf['adaptive_modulator']
        assert val_bs == 0
        ad_creators = (lambda w: Modulator(w, ad_conf['hidden_size'], optimizer_creator_factory(), out_bias=ad_conf['out_bias'], relu=ad_conf['relu'], summary_writer=writer),None)
    else:
        ad_creators = (None,None)


    if name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, num_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, num_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=conf.get('dropout',0.0), num_classes=num_class, adaptive_dropouter_creator=ad_creators[0],adaptive_conv_dropouter_creator=ad_creators[1], groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False), virtual_bn=conf.get('virtual_bn', False))
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=conf.get('dropout',0.0), num_classes=num_class, adaptive_dropouter_creator=ad_creators[0],adaptive_conv_dropouter_creator=ad_creators[1], groupnorm=conf.get('groupnorm',False), examplewise_bn=conf.get('examplewise_bn', False), virtual_bn=conf.get('virtual_bn', False))
    elif name == 'miniconvnet':
        model = SeqConvNet(num_class,adaptive_dropout_creator=ad_creators[0],batch_norm=False)
    elif name == 'mlp':
        model = MLP(num_class, (3,32,32), adaptive_dropouter_creator=ad_creators[0])
    elif name == 'shakeshake26_2x32d':
        model = ShakeResNet(26, 32, num_class)
    elif name == 'shakeshake26_2x64d':
        model = ShakeResNet(26, 64, num_class)
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_class)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_class)

    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_class)

    elif name == 'pyramid':
        model = PyramidNet('cifar10', depth=conf['depth'], alpha=conf['alpha'], num_classes=num_class, bottleneck=conf['bottleneck'])
    else:
        raise NameError('no model named, %s' % name)

    if conf.get('weight_norm', False):
        print('Using weight norm.')
        apply_weightnorm(model)

    #model = model.cuda()
    #model = DataParallel(model)
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'noised_cifar10': 10,
        'targetnoised_cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'fiftyexample_cifar100': 100,
        'tenclass_cifar100': 10,
        'svhn': 10,
        'svhncore': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
