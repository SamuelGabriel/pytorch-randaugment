from tensorboard.backend.event_processing import event_accumulator
from os import listdir
from os.path import isfile, join
import re
import argparse
import numpy as np, scipy.stats as st
def get_last_metric(path, metric):
    onlyfiles = sorted([join(path,f) for f in listdir(path) if isfile(join(path, f))])
    last_point = 0
    v = None
    for f in onlyfiles:
        ea = event_accumulator.EventAccumulator(f)
        # top1 not found
        ea.Reload()
        try:
            e = ea.Scalars(metric)[-1]
            if e.step >= last_point:
                if last_point > 0:
                    print("Warning: Multiple runs with one name:", f, "other result:", v, 'at', last_point)
                last_point = e.step
                v = e.value
        except Exception as e:
            print(e)
    return v, last_point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Top1 Results.')
    parser.add_argument('path',
                   help='Prefix of the paths/conf files for the runs to evaluate.')
    parser.add_argument('--logdir', default='logs4')
    parser.add_argument('--split', default='test')
    parser.add_argument('--metric', default='top1', help='Can be e.g. top1, top5, loss, eval_top1')
    args = parser.parse_args()
    mypath = args.path.split('/')[-1] #'wresnet28x10_cifar100_4xb64_valsteps_maxlr.1_learnedrandprepreprocessorensemble_optwidelongsesp_50epochs_2_.0ent_no0augs_.001mlr_exploresm'
    suffix = '.yaml'
    if mypath.endswith(suffix):
        mypath = mypath[:-len(suffix)]
    logdir = args.logdir
    paths = [path for path in listdir(logdir) if re.search(f'{mypath}(_.try|).yaml',path)]
    print([path[len(mypath):] for path in paths])

    paths = [join(join(logdir,path), args.split) for path in paths]
    results = [get_last_metric(path, args.metric) for path in paths]
    assert all([r[1] == results[0][1] for r in results]), results
    step = results[0][1]
    results = [r[0] for r in results]
    assert all(results), results
    print(f"The results are the following {len(results)} at step {step}: {results}")

    results = np.array(results)
    n = len(results)
    m, se = np.mean(results), st.sem(results)
    confidence = .95
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    print(f"Mean: {round(np.mean(results),4)}, Std: {round(np.std(results),4)}, +/-: {round(h,4)}")