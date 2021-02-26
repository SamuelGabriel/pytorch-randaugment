import logging
logging.basicConfig(level=logging.DEBUG)

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

"""
model:
  type: wresnet28_10
dataset: cifar100
aug: default
augmentation_search_space: long_wide
cutout: 16
batch: 64
gpus: 4
epoch: 200
lr: 0.05
lr_schedule:
  type: 'cosine'
  warmup:
    multiplier: 2
    epoch: 5
optimizer:
  type: sgd
  nesterov: True
  decay: 0.0005

preprocessor:
  type: learned_random_randaugmentspace_ensemble
  entropy_alpha: 0.
  hidden_dim: 1
  scale_entropy_alpha: .0
  online_tests_on_model: false
  q_zero_init: true
  q_residual: false
  scale_embs_zero_init: true
  normalize_reward: true
  importance_sampling: false
  label_smoothing_rate: .0
  possible_num_sequential_transforms: [1,2,3,4]
  sigmax_dist: false
  use_images_for_sampler: false
  uniaug_val: false
  oldpreprocessor_val: false
  exploresoftmax_dist: false
  aug_probs: true
  non_embedding_sampler: true
  update_every_k_steps: 100


alignment_loss:
  summation: standard
  align_with: '2'
  val_share: .0
  alignment_type: dot
  has_val_steps: true

meta_opt:
  meta_optimizer:
    type: adam
    lr: .1
    beta1: .9
    beta2: .999

"""

def bool_hp(name):
    return CSH.CategoricalHyperparameter(name, choices=[True,False])


class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def compute(self, config, budget, **kwargs):
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
        result = train.run_from_py('data', self.csconfig_to_realconfig(config, round(budget)))
        loss = 1.-result
        print(f'finished with loss {loss}')

        return({
                    'loss': loss,  # this is the a mandatory field to run hyperband
                    'info': result  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def csconfig_to_realconfig(cs_config: Dict[str, Any], epochs):
        cs_config = {(k[2:] if k.startswith('1-') else k) : (1-v if k.startswith('1-') else v) for k,v in cs_config.items()}
        config = \
        {'model': {'type': 'wresnet28_10'},
         'dataset': 'cifar100',
         'aug': 'default',
         'augmentation_search_space': 'long_wide',
         'cutout': 16,
         'batch': 64,
         'gpus': 4,
         'epoch': epochs,
         'lr': 0.05,
         'seed': 2020,
         'lr_schedule': {'type': 'cosine', 'warmup': {'multiplier': 2, 'epoch': 5}},
         'optimizer': {'type': 'sgd', 'nesterov': True, 'decay': 0.0005},
         #'finite_difference_loss': True,
         'preprocessor': {
                          **{
                              'type': 'learned_random_randaugmentspace_ensemble',
                              'sigmax_dist': False,
                              'hidden_dim': 1,
                              'online_tests_on_model': False,
                              'q_zero_init': True,
                              'q_residual': False,
                              'scale_embs_zero_init': True,
                              'normalize_reward': True,
                              'importance_sampling': False,
                              'use_labels_for_sampler': False,
                              'label_smoothing_rate': 0.0,
                              'aug_probs': True,
                              'non_embedding_sampler': True,
                              'possible_num_sequential_transforms': [1, 2, 3, 4],
                              'use_images_for_sampler': False,
                              'uniaug_val': False,
                              'oldpreprocessor_val': False,
                          },
                          **{key[len('prpr.'):]: value for key, value in cs_config.items() if key.startswith('prpr.')}},

         'meta_opt': {'meta_optimizer': {'type': 'adam',**{key[len('meop.meop.'):]: value for key, value in cs_config.items() if key.startswith('meop.meop.')}}},
         'throwaway_share_of_ds': {'throwaway_share': .2,'use_throwaway_as_val': True}
        }
        if 'reward_type' in cs_config:
            if cs_config['reward_type'] == 'alignment_loss':
                config['alignment_loss'] =  {'summation': 'standard',
                'align_with': cs_config['allo.align_with'],
                'val_share': 0.0,
                'alignment_type': 'dot',
                'has_val_steps': True}
            elif cs_config['reward_type'] == 'next_step_loss':
                config['next_step_loss'] = True
                config['preprocessor']['normalize_reward'] = False
            else:
                raise ValueError()
        else:
            config['next_step_loss'] = True
            config['preprocessor']['normalize_reward'] = False
            """
            config['alignment_loss'] = {'summation': 'standard',
                                        'align_with': '2',
                                        'val_share': 0.0,
                                        'alignment_type': 'dot',
                                            'has_val_steps': True}
            """
        print(config)
        return config


    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace(seed=1)
        #cs.add_hyperparameter(CSH.CategoricalHyperparameter('prpr.type',
        #                                                    choices=['learned_random_randaugmentspace_ensemble',
        #                                                             'learned_random_randaugmentspace']))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('prpr.entropy_alpha', lower=0.0000001, upper=.1, log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('prpr.scale_entropy_alpha', lower=0.0000001, upper=.1, log=True))
        #hidden_dim = CSH.UniformIntegerHyperparameter('prpr.hidden_dim', lower=1, upper=100)
        #cs.add_hyperparameter(hidden_dim)
        #non_embedding_sampler = bool_hp('prpr.non_embedding_sampler')
        #cs.add_hyperparameter(non_embedding_sampler)
        #cs.add_condition(CS.EqualsCondition(hidden_dim, non_embedding_sampler, False))
        #scale_embs_zero_init = bool_hp('prpr.scale_embs_zero_init')
        #cs.add_hyperparameter(scale_embs_zero_init)
        #cs.add_condition(CS.EqualsCondition(scale_embs_zero_init, non_embedding_sampler, False))
        cs.add_hyperparameter(bool_hp('prpr.exploresoftmax_dist'))
        #cs.add_hyperparameter(bool_hp('prpr.aug_probs'))
        #use_labels_for_sampler = bool_hp('prpr.use_labels_for_sampler')
        #cs.add_hyperparameter(use_labels_for_sampler)
        #cs.add_condition(CS.EqualsCondition(use_labels_for_sampler, non_embedding_sampler, False))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('prpr.update_every_k_steps', lower=1, upper=300))

        #reward_type = CS.CategoricalHyperparameter('reward_type', ['alignment_loss', 'next_step_loss'])
        #align_with = CS.CategoricalHyperparameter('allo.align_with', ['1', '2'])
        #cs.add_hyperparameters([reward_type, align_with])
        #cs.add_condition(CS.EqualsCondition(align_with, reward_type, 'alignment_loss'))

        #opt_type = CS.CategoricalHyperparameter('meop.meop.type', ['adam', 'sgd', 'adabelief'])
        #cs.add_hyperparameter(opt_type)
        lr = CS.UniformFloatHyperparameter('meop.meop.lr', lower=.0001, upper=1., log=True)
        beta1 = CS.UniformFloatHyperparameter('1-meop.meop.beta1', lower=.0001, upper=.1, log=True)
        beta2 = CS.UniformFloatHyperparameter('1-meop.meop.beta2', lower=.0001, upper=.5, log=True)
        cs.add_hyperparameters([lr, beta1, beta2])
        #cs.add_condition(CS.NotEqualsCondition(beta2, opt_type, 'sgd'))
        return cs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=66)
    parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=200)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=30)
    parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=2)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='eth0')
    parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.')


    args=parser.parse_args()

    print("Num GPUs", torch.cuda.device_count())

    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)



    if args.worker:
        time.sleep(5)   # short artificial delay to make sure the nameserver is already running
        w = MyWorker(run_id=args.run_id, host=host)
        w.load_nameserver_credentials(working_directory=args.shared_directory)
        w.run(background=False)
        exit(0)

    # Start a nameserver:
    # We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
    ns_host, ns_port = NS.start()

    # Most optimizers are so computationally inexpensive that we can affort to run a
    # worker in parallel to it. Note that this one has to run in the background to
    # not plock!
    w = MyWorker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    # Run an optimizer
    # We now have to specify the host, and the nameserver information
    result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=False)
    bohb = BOHB(  configspace = MyWorker.get_configspace(),
                          run_id = args.run_id,
                          host=host,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          result_logger=result_logger,
                          min_budget=args.min_budget, max_budget=args.max_budget,
                   )
    res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)


    # In a cluster environment, you usually want to store the results for later analysis.
    # One option is to simply pickle the Result object
    with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)


    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()