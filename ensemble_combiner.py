import submitit
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import random
import argparse
from copy import deepcopy
import yaml
import pickle
import numpy as np
import time
import yaml

from ensemblemember_training import get_searchspace, acc_max
import tempfile


from collections import defaultdict
from typing import Dict,Any



def run_conf(conf, args):
    epochs = args.epochs
    from theconf import Config as C, ConfigArgumentParser
    from RandAugment.metrics import SummaryWriterDummy
    from RandAugment.train import get_dataloaders, get_model, num_class

    save_path = args.load_save_path + '____' + ''.join('_' + c for c, use in conf.get_dictionary().items() if use) + '.model'
    conf_dict = csconfig_to_realconfig(conf.get_dictionary(), epochs=epochs)
    with tempfile.NamedTemporaryFile(mode='w+') as f:
        yaml.dump(conf_dict, f)

        C(f.name)
        C.get()['started_with_spawn'] = False
        model = get_model(C.get()['model'], C.get()['batch'], 0, None,
                          num_class(C.get()['dataset']), writer=SummaryWriterDummy('hi'))

        data = torch.load(save_path, map_location='cpu')
        key = 'model' if 'model' in data else 'state_dict'
        print('checkpoint epoch@%d' % data['epoch'])
        model.load_state_dict(data[key])
        # if not isinstance(model, DataParallel):
        # model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
        # else:
        # model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
        assert data['epoch'] == C.get()['epoch']
        trainsampler, trainloader, validloader, testloader_, testtrainloader_, dataset_info = get_dataloaders(
            C.get()['dataset'], C.get()['batch'], 'data', 0.0, split_idx=0,
            get_meta_optimizer_factory=None, distributed=False,
            started_with_spawn=C.get()['started_with_spawn'], summary_writer=SummaryWriterDummy('hi'))
        model = model.cuda()
        model.eval()

        all_logits = torch.tensor([])
        all_labels = torch.tensor([], dtype=torch.int64)
        with torch.no_grad():
            for x,y in testloader_:
                logits = model(x.cuda())
                all_logits = torch.cat([all_logits,logits.cpu()])
                all_labels = torch.cat([all_labels,y])

    return all_logits, all_labels





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_config')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--reduced_search_space', action='store_true')
    parser.add_argument('--log_file', default=None)
    parser.add_argument('--load_save_path', default='')
    parser.add_argument('--partition', default="alldlc_gpu-rtx2080")
    parser.add_argument('--eval_on_test', action='store_true')

    args = parser.parse_args()

    with open(args.base_config, 'r') as f:
        base_config = yaml.load(f,Loader=yaml.FullLoader)
    assert 'throwaway_share_of_ds' in base_config and base_config['throwaway_share_of_ds']['use_throwaway_as_val'] == True
    if args.eval_on_test:
        print("EVAL ON TEST")
        del base_config['throwaway_share_of_ds']
        assert 'throwaway_share_of_ds' not in base_config


    def csconfig_to_realconfig(cs_config: Dict[str, bool], epochs):
        config = deepcopy(base_config)
        config['custom_search_space_augs'] = [aug for aug, use_this_aug in cs_config.items() if use_this_aug] * 1000
        config['epoch'] = epochs
        return config


    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder="log_test")
    # set timeout in min, and partition for running the job
    executor.update_parameters(timeout_min=59, slurm_partition=args.partition, slurm_gres='gpu:1',
                              slurm_setup=['export MKL_THREADING_LAYER=GNU'], slurm_exclude='dlcgpu02,dlcgpu18')

    cs = get_searchspace(full_search_space=not args.reduced_search_space)


    runs = {}
    for aug in cs.get_hyperparameter_names():
        config = CS.Configuration(cs,{**{k:False for k in cs.get_hyperparameter_names()}, aug:True})
        new_member = executor.submit(
            run_conf, config, args
        )
        runs[aug] = new_member

    results = {}
    for aug in cs.get_hyperparameter_names():
        l_of_aug, all_labels = runs[aug].result()
        results[aug] = {'logits': l_of_aug, 'labels': all_labels}

    torch.save(results, args.log_file)#  'all_logits.pt_pickle')









