import submitit
from RandAugment import train
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import random
import argparse
from copy import deepcopy
import yaml
import pickle
import numpy as np
import time


from dehb import DEHB


from collections import defaultdict
from typing import Dict,Any

def acc_max(l):
    return max(l, key=lambda x: x[1])


def get_searchspace(full_search_space=True):
    def bool_hp(name):
        return CSH.CategoricalHyperparameter(name, choices=[True, False])

    cs = CS.ConfigurationSpace(seed=1)

    cs.add_hyperparameter(bool_hp('identity'))
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
    if full_search_space:
        cs.add_hyperparameter(bool_hp('flip_lr'))
        cs.add_hyperparameter(bool_hp('invert'))
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



def run_conf(conf, args):
    epochs = args.epochs
    save_path = args.load_save_path + '____' + ''.join('_' + c for c, use in conf.get_dictionary().items() if use) + '.model'
    return conf, train.run_from_py('data', csconfig_to_realconfig(conf.get_dictionary(), epochs=epochs), save_path)

def csconfig_to_realconfig(cs_config: Dict[str, bool], epochs):
    config = deepcopy(base_config)
    config['custom_search_space_augs'] = [aug for aug, use_this_aug in cs_config.items() if use_this_aug] * 1000
    config['epoch'] = epochs
    return config




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_config')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--reduced_search_space', action='store_true')
    parser.add_argument('--log_file', default=None)
    parser.add_argument('--load_save_path', default='')

    args = parser.parse_args()

    with open(args.base_config, 'r') as f:
        base_config = yaml.load(f,Loader=yaml.FullLoader)




    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder="log_test")
    # set timeout in min, and partition for running the job
    executor.update_parameters(timeout_min=60*24, slurm_partition="alldlc_gpu-rtx2080", slurm_gres='gpu:1',
                              slurm_setup=['export MKL_THREADING_LAYER=GNU'], slurm_exclude='dlcgpu02,dlcgpu18')

    cs = get_searchspace(full_search_space=not args.reduced_search_space)

    for aug in cs.get_hyperparameter_names():
        config = CS.Configuration(cs,{**{k:False for k in cs.get_hyperparameter_names()}, aug:True})
        new_member = executor.submit(
            run_conf, config, args
        )




    def target_function(conf, budget):
        assert budget == args.epochs
        new_member = executor.submit(
            run_conf, conf, args
        )
        acc = new_member.result()[1]
        return 1.-acc




