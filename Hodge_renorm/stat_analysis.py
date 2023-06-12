import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import powerlaw as pwl
import scipy
from Functions import plotting, scomplex
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
                # KS Distance
                test = scipy.stats.kstest(deg1, deg2)
                deg_distance[r, norml, degg, tau] = test.statistic


# fig = plt.figure(figsize=(10, 6))

# gs = fig.add_gridspec(d, 2)
# ax1 = fig.add_subplot(gs[:, 0])
# axv = []
# for i in range(d):
#     ax = fig.add_subplot(gs[i, 1])
#     axv = axv + [ax]
# fig.tight_layout(pad=3)

# sc = scomplex.NGF(d, 100, s, beta)


# plotting.plot_complex(sc, ax1, palette[d-1,:])
# ax1.set_title(r"\textbf{NGF d = " + str(d) + ", s = " + str(s) + "}", fontsize=14)


# bin = np.linspace(0, 1, num=30)

fig, axv = plt.subplots(1, d, figsize=(5 * d, 4))
plt.locator_params(axis='y', nbins=6)
plt.locator_params(axis='x', nbins=5)


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

# for i in range(d):
#     if d == 1:
#         ax = axv
#     else:
#         ax = axv[i]
#     data = {
#         "Compression rate": bin[
#             np.digitize(1 - (Ns[:, :, :].flatten()) / N, bin, right=True) - 1
#         ],
#         "KS distance": deg_distance[:, :, i, :].flatten(),
#         "type": np.array(
#             [
#                 [["$L_" + str(j) + "$" for _ in range(n_tau)] for j in range(d + 1)]
#                 for _ in range(rep)
#             ]
#         ).flatten(),
#     }
#     # Creates pandas DataFrame.
#     df = pd.DataFrame(data)
#     if ls:
#         ax.set_yscale("log")
#     sns.lineplot(
#         x="Compression rate",
#         y="KS distance",
#         hue="type",
#         data=df,
#         ax=ax,
#         palette=palette,
#         legend="brief"
#     )

#     ax.legend(loc="upper left")
#     ax.set_title(r"\textbf{" + names[i] + "-" + names[d] + "}", fontsize=14)
#     ax.set_xlim([0, 0.5])

# plt.show()
plt.savefig(path + "/deg_errors.pdf", format="pdf")  # , bbox_inches="tight")
