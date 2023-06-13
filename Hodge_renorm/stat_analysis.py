import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import powerlaw as pwl
import scipy
from Functions import plotting

plt.rcParams["text.usetex"] = True
palette = np.array(
    [
        [0.3647, 0.2824, 0.1059],
        [0.8549, 0.6314, 0.3294],
        [0.4745, 0.5843, 0.5373],
        [0.4745, 0.3843, 0.7373],
        [107.0 / 255, 42.0 / 255, 2.0 / 255],
    ]
)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# palette = np.array([[0, 18, 25], [10, 147, 150], [233, 216, 166] ,[238, 155, 0], [174, 32, 18]])/255


N = int(sys.argv[1])
d = int(sys.argv[2])
n_tau = int(sys.argv[3])
rep = int(sys.argv[4])
METHOD = sys.argv[5]  # {"representative","closest"}
SPARSIFY = sys.argv[6] == "True"
TRUE_CONNECTIONS = sys.argv[7] == "True"

s = 1
beta = 0.1
ls = True  # logscale

suff = ""
pref = f"d{d}s{s}" + suff

path = f"Tests/Experiments_{METHOD}_{SPARSIFY}_{TRUE_CONNECTIONS}/{pref}"


with open(path + "/deg_dist.pkl", "rb") as f:
    deg_dist = pickle.load(f)


plotting.plot_deg_dist(deg_dist)
