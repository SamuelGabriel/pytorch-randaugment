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



from collections import defaultdict
from typing import Dict,Any

def acc_max(l):
    return max(l, key=lambda x: x[1])



def run_conf(conf, args):
    epochs = args.epochs
    #save_path = args.load_save_path + '____' + ''.join('_' + c for c, use in conf.items()) + '.model'
    return conf, train.run_from_py('data', csconfig_to_realconfig(conf, epochs=epochs), '')

def csconfig_to_realconfig(cs_config: Dict[str, bool], epochs):
    config = deepcopy(base_config)
    config['randaug'] = cs_config # contains N and M as keys
    config['epoch'] = epochs
    return config


def wait_for_runs(population):
    new_population = []
    for member in population:
        if isinstance(member,tuple):
            new_population.append(member)
        else:
            new_population.append(member.result())
    return new_population


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_config')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_file', default=None)

    args = parser.parse_args()

    with open(args.base_config, 'r') as f:
        base_config = yaml.load(f,Loader=yaml.FullLoader)




    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder="log_test")
    # set timeout in min, and partition for running the job
    executor.update_parameters(timeout_min=60*24, slurm_partition="alldlc_gpu-rtx2080", slurm_gres='gpu:1',
                              slurm_setup=['export MKL_THREADING_LAYER=GNU'], slurm_exclude='dlcgpu02,dlcgpu18')


    runs = []
    for N in range(1,4):
        for M in range(1,31):
            config = {'N': N, 'M': M}
            runs.append(executor.submit(run_conf, config, args))


    results = {}
    for run in runs:
        try:
            result = run.result()
            results[(result[0]['N'], result[0]['M'])] = result[1]
        except:
            print("Some exception happened")

    print(results)
    if args.log_file is not None:
        with open(args.log_file + '.pickle', 'wb') as f:
            pickle.dump(results, f)
    print("Acc max", acc_max(results.items()))





