import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw as pwl
import scipy
from Functions import plotting, scomplex
from scipy.stats import t
import seaborn as sns
import pandas as pd

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


N = int(sys.argv[1])  # 2000
d = int(sys.argv[2])  # 1
n_tau = int(sys.argv[3])  # 100
rep = int(sys.argv[4])  # 10
METHOD = sys.argv[5]  # {"representative","closest"}
SPARSIFY = sys.argv[6] == "True"  # False
TRUE_CONNECTIONS = sys.argv[7] == "True"  # True

s = 1
beta = 0.1
ls = True  # logscale

suff = ""
pref = f"d{d}s{s}" + suff
path = f"Tests/Experiments_{METHOD}_{SPARSIFY}_{TRUE_CONNECTIONS}/{pref}"


with open(path + "/deg_dist.pkl", "rb") as f:
    deg_dist = pickle.load(f)

Ns = np.zeros((rep, d + 1, n_tau), dtype=int)
for r in range(rep):
    for i in range(d + 1):
        for t in range(n_tau):
            Ns[r, i, t] = len(deg_dist[r][t][i][0])

# Ns = sp.io.loadmat(path + '/Ns.mat')['Ns']
# deg_dist = sp.io.loadmat(fpath pref + '/deg_dist.mat')['deg_dist']

names = ["Node", "Link", "Face", "Tetrahedron", "4_Simplex"]


deg_distance = np.zeros((rep, d + 1, d, n_tau))
for r in range(rep):
    for norml in range(d + 1):
        for degg in range(d):
            deg1 = deg_dist[r][0][0][degg]
            for tau in range(n_tau):
                deg2 = deg_dist[r][tau][norml][degg]
                if len(deg2) == 0:
                    deg2 = [0]
                # Powerlaw
                fit_function = pwl.Fit(deg2, xmin=1, discrete=True)
                deg_distance[r, norml, degg, tau] = fit_function.power_law.D


# fig = plt.figure(figsize=(10, 6))

# gs = fig.add_gridspec(d, 2)
# ax1 = fig.add_subplot(gs[:, 0])
# axv = []
# for i in range(d):
#     ax = fig.add_subplot(gs[i, 1])
#     axv = axv + [ax]
# fig.tight_layout(pad=3)

# sc = scomplex.NGF(d, 100, s, beta)


# plotting.plot_complex(sc, ax1)
# ax1.set_title(r"\textbf{NGF d = " + str(d) + ", s = " + str(s) + "}", fontsize=14)


fig, axv = plt.subplots(1, d, figsize=(5 * d, 4))

for i in range(d):
    if d == 1:
        ax = axv
    else:
        ax = axv[i]
    ax.set_xlabel("Coarse graining rate")
    ax.set_ylabel("KS distance")

    for j in range(d + 1):
        for r in range(rep):
            if r == 0:
                lab = "$L_" + str(j) + "$"
            else:
                lab = ""
            id = np.argwhere(Ns[r, j, :] != 1)
            ax.plot(
                1 - Ns[r, j, id] / N,
                deg_distance[r, j, i, id],
                color=palette[j, :],
                alpha=0.3,
                linewidth=0.8,
            )
            ax.plot(
                1 - Ns[r, j, id] / N,
                deg_distance[r, j, i, id],
                "o",
                alpha=0.8,
                color=palette[j, :],
                ms=4,
                label=lab,
            )
            ax.legend()
    ax.set_title(r"\textbf{" + names[i] + "-" + names[d] + "}", fontsize=14)


# plt.show()
plt.savefig(path + "/powerlaw.pdf", format="pdf")  # , bbox_inches="tight")
