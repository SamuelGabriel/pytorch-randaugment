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
from hpbandster.optimizers import BOHB, RandomSearch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from RandAugment import train


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
        config = \
        {'model': {'type': 'wresnet28_10'},
         'dataset': 'cifar100',
         'aug': 'randaugment',
         'randaug': {'N': 0, 'M': 0, 'weights': [0.0,1.0]},
         'augmentation_search_space': 'fix_custom',
         'custom_search_space_augs': [aug for aug, use_this_aug in cs_config.items() if use_this_aug],
         'cutout': 16,
         'batch': 128,
         'gpus': 1,
         'epoch': epochs,
         'lr': 0.05,
         'seed': 2020,
         'lr_schedule': {'type': 'cosine', 'warmup': {'multiplier': 2, 'epoch': 5}},
         'optimizer': {'type': 'sgd', 'nesterov': True, 'decay': 0.0005},
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
    parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=66)
    parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=200)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=162)
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
    bohb = RandomSearch(  configspace = MyWorker.get_configspace(),
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