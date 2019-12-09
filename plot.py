'''
from baselines.common import plot_util as pu
import matplotlib.pyplot as plt

LOG_DIRS = 'logs/halfcheetahr/'

results = pu.load_results(LOG_DIRS)

f, arr = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=True)

plt.savefig("img/randomsplit_halfcheetah.png")

'''

import matplotlib.pyplot as plt
import numpy as np
import os

result_dir = 'experiment_results/'
envs = ['HalfCheetah-v2', 'Hopper-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2', 'Ant-v2']

env = envs[0]

path = result_dir + env + '/'
is_gail = False
lowerf = []
upperf = []
splitf = []
rsplitf = []

def read_stats(f):
    return np.load(path+f)

for f in os.listdir(path):
    if is_gail:
        if 'gail' not in f:
            continue
    else:
        if 'gail' in f:
            continue

    if 'baseline' in f and 'dimh2' in f:
        lowerf.append(path+f)
    elif 'baseline' in f and 'dimh32' in f:
        upperf.append(path+f)
    elif 'split' in f and 'rsplit' not in f:
        splitf.append(path+f)
    elif 'rsplit' in f:
        rsplitf.append(path+f)

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']

exps = [lowerf, upperf, splitf, rsplitf]
titles = ['dim2', 'dim32', 'split', 'randsplit']

for i, (legd, exp) in enumerate(zip(titles, exps)):
    x = []; r = []
    for e in exp:
        dic = np.load(e, allow_pickle=True)
        x.append(list(dic.item().get('steps')))
        r.append(list(dic.item().get('mean reward')))
    x = np.array(x) # [n_seeds, :]
    x = x[0]
    r = np.array(r) # [n_seeds, :]
    r_mu = np.mean(r, 0)
    r_std = np.std(r, 0)

    plt.plot(x, r_mu, c=COLORS[i], label=legd)
    plt.fill_between(x, r_mu-r_std, r_mu+r_std)

plt.savefig(result_dir+env+'.png', dpi=300)
