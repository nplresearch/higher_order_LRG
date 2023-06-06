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

pref = f"d{d}s{s}"
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

names = ["Node", "Link", "Face", "Tetrahedron"]


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
                # Powerlaw
                # fit_function = pwl.Fit(deg2)
                # deg_distance[r, norml, degg, tau] = fit_function.power_law.D


fig = plt.figure(figsize=(10,6))

gs = fig.add_gridspec(d, 2)
ax1 = fig.add_subplot(gs[:, 0])
axv = []
for i in range(d):
    ax = fig.add_subplot(gs[i, 1])
    axv = axv + [ax]
fig.tight_layout(pad = 3)

sc = scomplex.NGF(d, 100, s, beta)


plotting.plot_complex(sc, ax1)
ax1.set_title(r"\textbf{NGF d = " + str(d) + ", s = " + str(s) + "}", fontsize=14)


bin = np.linspace(0,1,num = 10)
for i in range(d):
    ax = axv[i]
    data = {"Compression rate": bin[np.digitize(1- (Ns[:,:,:].flatten())/N,bin,right = True)-1],
            "KS distance": deg_distance[:,:,i,:].flatten(),
            'type': np.array([[["$L_" + str(j)+"$" for _ in range(n_tau)] for j in range(d+1)] for _ in range(rep)]).flatten()}
    # Creates pandas DataFrame.
    df = pd.DataFrame(data)
    if ls:
        ax.set_yscale("log")
    sns.lineplot(x="Compression rate",y="KS distance", hue = 'type', data=df, ax = ax, palette=palette, legend = 'brief')

    ax.legend(loc = 'upper left')
    ax.set_title(r"\textbf{" + names[i] + "-" + names[d] + "}", fontsize=14)
    ax.set_xlim([0, 0.5])

#plt.show()
plt.savefig(path + "/deg_errors.pdf", format="pdf")#, bbox_inches="tight")
