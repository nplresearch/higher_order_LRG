import pickle
import sys

sys.path.append(sys.path[0]+'/..')  # Adds higher directory to python modules path.

import matplotlib.pyplot as plt
import numpy as np
import scipy

plt.rcParams["text.usetex"] = True
palette = np.array(
    [
        [0.3647, 0.2824, 0.1059],
        [0.8549, 0.6314, 0.3294],
        [0.4745, 0.5843, 0.5373],
        [200.0 / 255, 202.0 / 255, 180.0 / 255],
        [107.0 / 255, 42.0 / 255, 2.0 / 255],
    ]
)
# palette = np.array([[0, 18, 25], [10, 147, 150], [233, 216, 166] ,[238, 155, 0], [174, 32, 18]])/255


N = 500
d = 2
n_tau = 100
rep = 1
METHOD = "representative"
SPARSIFY = False
TRUE_CONNECTIONS = False

s = 1
beta = 0.1
ls = True  # logscale

suff = "updown"
pref = f"d{d}s{s}" + suff

path = f"Tests/Experiments_{METHOD}_{SPARSIFY}_{TRUE_CONNECTIONS}/{pref}"


with open(path + "/deg_dist.pkl", "rb") as f:
    deg_dist = pickle.load(f)

Ns = np.zeros((rep, 3, n_tau), dtype=int)
for r in range(rep):
    for i in range(d + 1):
        for t in range(n_tau):
            Ns[r, i, t] = len(deg_dist[r][t][i][0])

# Ns = sp.io.loadmat(path + '/Ns.mat')['Ns']
# deg_dist = sp.io.loadmat(fpath pref + '/deg_dist.mat')['deg_dist']

names = ["Node", "Link", "Face", "Tetrahedron", "4_Simplex"]


deg_distance = np.zeros((rep, 3, d, n_tau))
for r in range(rep):
    for norml in range(3):
        for degg in range(d):
            deg1 = deg_dist[r][0][0][degg]
            for tau in range(n_tau):
                deg2 = deg_dist[r][tau][norml][degg]
                if len(deg2) == 0:
                    deg2 = [0]
                # KS Distance
                test = scipy.stats.kstest(deg1, deg2)
                deg_distance[r, norml, degg, tau] = test.statistic


fig, axv = plt.subplots(1, d, figsize=(5 * d, 4))

for i in range(d):
    if d == 1:
        ax = axv
    else:
        ax = axv[i]
    for j in range(3):
        for r in range(rep):
            if r == 0:
                if j == 0:
                    lab = r"$L_1$"
                elif j == 1:
                    lab = r"$L_1^{\uparrow}$"
                elif j == 2:
                    lab = r"$L_1^{\downarrow}$"

            else:
                lab = ""
            id = np.argwhere(Ns[r, j, :] != 1)
            ax.plot(
                1 - Ns[r, j, id] / N,
                deg_distance[r, j, i, id],
                "-o",
                color=palette[j, :],
                alpha=0.7,
                linewidth=0.8,
                markersize=4,
                label=lab,
            )
    ax.legend()
    ax.set_title(r"\textbf{" + names[i] + "-" + names[d] + "}", fontsize=14)

plt.savefig(path + "/deg_errors.pdf", format="pdf")  # , bbox_inches="tight")
