from tensorboard.backend.event_processing import event_accumulator
from tensorboardX import SummaryWriter
from os import listdir, makedirs, rmdir
from os.path import isfile, join, isdir
import shutil
import re
import argparse
from collections import defaultdict
import numpy as np, scipy.stats as st
from tqdm import tqdm
def get_last_curve(path, tag='top1'):
    onlyfiles = sorted([join(path,f) for f in listdir(path) if isfile(join(path, f))])
    last_point = 0
    v = None
    for f in onlyfiles:
        ea = event_accumulator.EventAccumulator(f)
        # top1 not found
        ea.Reload()
        try:
            e = ea.Scalars(tag)[-1]
            if e.step >= last_point:
                if last_point > 0:
                    print("Warning: Multiple runs with one name:", f, "other result:", v, 'at', last_point)
                last_point = e.step
                v = ea.Scalars(tag)
        except Exception as e:
            print(e)
    return v, last_point

def get_exp_name(path: str):
    path = path.split('/')[-1]
    return re.sub(r"(_[0-9]+try)?.yaml","", path)

def curves_to_matrix(curves):
    matrix = np.zeros((len(curves), min(len(c) for c in curves)))
    for t,events in enumerate(zip(*curves)):
        assert all(e.step == events[0].step for e in events), [e.step for e in events]
        matrix[:,t] = np.array([e.value for e in events])
    return matrix




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Top1 Results.')
    parser.add_argument('--logdir', default='logs3')
    parser.add_argument('--out-logdir', default='/tmp/logs3av')
    args = parser.parse_args()

    if isdir(args.out_logdir):
        if not input(f"I will now delete {args.out_logdir}, ok?\n"):
            raise ValueError(f"Need to delete {args.out_logdir} before using it again.")
        shutil.rmtree(args.out_logdir)

    else:
        makedirs(args.out_logdir)


    directories = [join(args.logdir,d) for d in listdir(args.logdir)]
    directory_map = defaultdict(list)
    for d in directories:
        directory_map[get_exp_name(d)].append(d)
    for exp_name, exp_instance_directory in tqdm(directory_map.items()):
        for split in ['test', 'testtrain']:
            ds = [join(d,split) for d in exp_instance_directory]
            ds = [d for d in ds if isdir(d)]
            if not ds:
                continue
            for tag in ['top1',]:
                curves = [get_last_curve(d, tag)[0] for d in ds]
                curves = [c for c in curves if c is not None]
                if not curves:
                    continue
                matrix = curves_to_matrix(curves)
                means = matrix.mean(axis=0)
                out_writer = SummaryWriter(log_dir=join(args.out_logdir,exp_name+f'_average_of_{len(curves)}'))
                for step, mean in zip([e.step for e in curves[0]], means):
                    out_writer.add_scalar(tag+'_'+split, mean, step)








