import logging
logging.basicConfig(level=logging.WARNING)

import argparse
import pickle
import time
import os

from typing import Dict, Any

import torch

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from RandAugment import train


def bool_hp(name):
    return CSH.CategoricalHyperparameter(name, choices=[True,False])


class MyWorker():


    def compute(self, num_run, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        print('started run')

        augmentations = sorted(self.get_configspace().get_hyperparameter_names())

        aug = augmentations[num_run % len(augmentations)]

        result = train.run_from_py('data', self.csconfig_to_realconfig({aug: True}, 200))
        loss = 1.-result
        print(f'finished with loss {loss}')

        return({
                    'loss': loss,  # this is the a mandatory field to run hyperband
                    'info': {'acc': result, 'aug': aug}  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def csconfig_to_realconfig(cs_config: Dict[str, Any], epochs):
        config = \
        {'model': {'type': 'wresnet28_10'},
         'dataset': 'svhncore', #'cifar10',
         'aug': 'randaugment',
         'randaug': {'N': 1, 'M': 0},
         'augmentation_search_space': 'fix_custom',
         'custom_search_space_augs': [aug for aug, use_this_aug in cs_config.items() if use_this_aug],
         'cutout': 16,
         'batch': 128,
         'gpus': 1,
         'epoch': epochs,
         'lr': .005,#0.05,
         'seed': 2020,
         'lr_schedule': {'type': 'cosine', },#'warmup': {'multiplier': 2, 'epoch': 5}},
         'optimizer': {'type': 'sgd', 'nesterov': True, 'decay': .005}, #0.0005},
         #'finite_difference_loss': True,
         'throwaway_share_of_ds': {'throwaway_share': .2,'use_throwaway_as_val': True}
        }
        print(config)
        return config


    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace(seed=1)
        cs.add_hyperparameter(bool_hp('auto_contrast'))
        cs.add_hyperparameter(bool_hp('equalize'))
        cs.add_hyperparameter(bool_hp('rotate'))
        cs.add_hyperparameter(bool_hp('solarize'))
        cs.add_hyperparameter(bool_hp('color'))
        cs.add_hyperparameter(bool_hp('posterize'))
        cs.add_hyperparameter(bool_hp('contrast'))
        cs.add_hyperparameter(bool_hp('brightness'))
        cs.add_hyperparameter(bool_hp('sharpness'))
        cs.add_hyperparameter(bool_hp('shear_x'))
        cs.add_hyperparameter(bool_hp('shear_y'))
        cs.add_hyperparameter(bool_hp('translate_x'))
        cs.add_hyperparameter(bool_hp('translate_y'))
        cs.add_hyperparameter(bool_hp('blur'))
        cs.add_hyperparameter(bool_hp('invert'))
        cs.add_hyperparameter(bool_hp('flip_lr'))
        cs.add_hyperparameter(bool_hp('flip_ud'))
        cs.add_hyperparameter(bool_hp('cutout'))
        cs.add_hyperparameter(bool_hp('crop_bilinear'))
        cs.add_hyperparameter(bool_hp('contour'))
        cs.add_hyperparameter(bool_hp('detail'))
        cs.add_hyperparameter(bool_hp('edge_enhance'))
        cs.add_hyperparameter(bool_hp('sharpen'))
        cs.add_hyperparameter(bool_hp('max_'))
        cs.add_hyperparameter(bool_hp('min_'))
        cs.add_hyperparameter(bool_hp('median'))
        cs.add_hyperparameter(bool_hp('gaussian'))

        return cs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--worker_id', type=int, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.')


    args=parser.parse_args()



    print("Num GPUs", torch.cuda.device_count())

    w = MyWorker()
    res = w.compute(args.worker_id)


    with open(os.path.join(args.shared_directory, f"runresult_{res['info']['aug']}.csv"), 'a') as fh:
        fh.write(f"{res['info']['aug']}, {res['info']['acc']}\n")


    with open(os.path.join(args.shared_directory, 'runresults.csv'), 'a') as fh:
        fh.write(f"{res['info']['aug']}, {res['info']['acc']}\n")



