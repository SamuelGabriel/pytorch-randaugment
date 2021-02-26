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



from collections import defaultdict
from typing import Dict,Any

def acc_max(l):
    return max(l, key=lambda x: x[1])


def get_searchspace(full_search_space=True):
    def bool_hp(name):
        return CSH.CategoricalHyperparameter(name, choices=[True, False])

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

def get_fixed_search_space(cs):
    fixed_search_space_config_dict = {'auto_contrast': True,
                                      'blur': True,
                                      'brightness': True,
                                      'color': True,
                                      'contour': False,
                                      'contrast': True,
                                      'crop_bilinear': False,
                                      'cutout': False,
                                      'detail': False,
                                      'edge_enhance': False,
                                      'equalize': True,
                                      'flip_lr': False,
                                      'flip_ud': False,
                                      'gaussian': False,
                                      'invert': False,
                                      'max_': False,
                                      'median': False,
                                      'min_': False,
                                      'posterize': True,
                                      'rotate': True,
                                      'sharpen': False,
                                      'sharpness': True,
                                      'shear_x': True,
                                      'shear_y': True,
                                      'solarize': True,
                                      'translate_x': True,
                                      'translate_y': True}
    fixed_search_space_config_dict = {k: v for k,v in fixed_search_space_config_dict.items() if k in cs.get_hyperparameter_names() or v}
    fixed_search_space_config = CS.Configuration(cs, values=fixed_search_space_config_dict)
    return fixed_search_space_config

def mutate(config,random_replacement_prob):
    config_dict = config.get_dictionary()
    new_config_dict = {}
    is_there_a_difference = False
    while not is_there_a_difference:
        for config_name, value in config_dict.items():
            random_bool = bool(random.randint(0,1))
            new_value = random_bool if random.random() < random_replacement_prob else value
            new_config_dict[config_name] = new_value
            if value != new_value:
                is_there_a_difference = True
    return CS.Configuration(cs, values=new_config_dict)


def run_conf(conf, args):
    epochs = args.epochs
    save_path = args.load_save_path
    return conf, train.run_from_py('data', csconfig_to_realconfig(conf.get_dictionary(), epochs=epochs), save_path)

def wait_for_runs(population):
    new_population = []
    for member in population:
        if isinstance(member,tuple):
            new_population.append(member)
        else:
            new_population.append(member.result())
    return new_population


def initialize_evolution(executor, cs, fixed_search_space_config, args):
    pop = []
    #for i in range(args.pop_size - 1):
    #    pop.append(executor.submit(run_conf, cs.sample_configuration(), args.epochs))
    biased_portion = int(args.initial_biased_population_share * args.pop_size)
    unbiased_portion = args.pop_size - biased_portion
    configs = []
    for i in range(unbiased_portion-1):
        config = cs.sample_configuration()
        configs.append(config)
        pop.append(executor.submit(run_conf, config, args))
    pop.append(executor.submit(run_conf, CS.Configuration(cs, vector=np.array([False]*len(fixed_search_space_config.get_dictionary()))), args))
    if biased_portion:
        for i in range(biased_portion - 1):
            new_conf = mutate(fixed_search_space_config, args.random_replacement_prob)
            pop.append(executor.submit(run_conf, new_conf, args))
        pop.append(executor.submit(run_conf, fixed_search_space_config, args))
    return pop


def evolution_step(executor, cs, pop, died_pop, args):
    assert len(pop) == args.pop_size
    pop_size = len(pop)
    added_pop = []
    for contest_idx in range(args.number_of_parallel_runs):
        contestants = random.sample(pop, args.number_of_contestants_per_contest)
        best_contestant = acc_max(contestants)
        new_conf = mutate(best_contestant[0], args.random_replacement_prob)
        new_member = executor.submit(
            run_conf, new_conf, args
        )
        added_pop.append(new_member)
    pop.extend(wait_for_runs(added_pop))
    died_pop.extend(pop[:-pop_size])
    del pop[:-pop_size]
    return pop, died_pop



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_config')
    parser.add_argument('--number_of_parallel_runs', type=int, default=10)
    parser.add_argument('--number_of_contestants_per_contest', type=int, default=20)
    parser.add_argument('--pop_size', type=int, default=100)
    parser.add_argument('--random_replacement_prob', type=float, default=.05)
    parser.add_argument('--number_of_evolution_steps', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--initial_biased_population_share', type=float, default=.5)
    parser.add_argument('--load_pop_and_dead_pop', action='store_true')
    parser.add_argument('--reduced_search_space', action='store_true')
    parser.add_argument('--log_file', default=None)
    parser.add_argument('--load_save_path', default='')

    args = parser.parse_args()

    with open(args.base_config, 'r') as f:
        base_config = yaml.load(f,Loader=yaml.FullLoader)


    def csconfig_to_realconfig(cs_config: Dict[str, bool], epochs):
        config = deepcopy(base_config)
        config['custom_search_space_augs'] = [aug for aug, use_this_aug in cs_config.items() if use_this_aug]
        config['epoch'] = epochs
        return config

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder="log_test")
    # set timeout in min, and partition for running the job
    executor.update_parameters(timeout_min=60*24, slurm_partition="alldlc_gpu-rtx2080", slurm_gres='gpu:1',
                              slurm_setup=['export MKL_THREADING_LAYER=GNU'], slurm_exclude='dlcgpu02,dlcgpu18')

    cs = get_searchspace(full_search_space=not args.reduced_search_space)
    fixed_search_space_config = get_fixed_search_space(cs)

    if args.load_pop_and_dead_pop:
        with open(args.log_file + '.pickle', 'rb') as f:
            loaded_dict = pickle.load(f)
            pop = loaded_dict['pop']
            died_pop = loaded_dict['died_pop']
    else:
        pop = wait_for_runs(initialize_evolution(executor,cs,fixed_search_space_config,args))
        died_pop = []


    for step in range(args.number_of_evolution_steps):
        print('youngest', pop[-1])
        print('strongest', acc_max(pop))
        if died_pop: print('dead strongest', acc_max(died_pop))

        if args.log_file is not None:
            with open(args.log_file+'.pickle', 'wb') as f:
                pickle.dump({'pop': pop, 'died_pop': died_pop},f)

        pop,died_pop = evolution_step(executor,cs,pop,died_pop,args)


