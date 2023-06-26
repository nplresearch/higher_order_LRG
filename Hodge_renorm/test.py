import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw as pwl
from scipy.sparse.linalg import eigsh
from Functions import plotting, scomplex, renormalize, support
from scipy.stats import t

palette = np.array(
    [
        [0.3647, 0.2824, 0.1059],
        [0.8549, 0.6314, 0.3294],
        [0.4745, 0.5843, 0.5373],
        [0.4745, 0.3843, 0.7373],
        [107.0 / 255, 42.0 / 255, 2.0 / 255],
    ]
)

N = 300
d = 2
s = 1
sc = scomplex.NGF(d, N, s, 0.1)

B1, B2, B3, B4, edge_dict, face_dict, tet_dict = scomplex.boundary_matrices_3(sc)

# Laplacians
L0 = B1 @ (B1.T)

L0 = L0.asfptype()
Na = sc["n0"] - 1
D0, U0 = eigsh(L0, k=Na, sigma=0, which="LM")
D0 = np.concatenate((D0, 10000 * np.ones(sc["n0"] - Na)))
U0 = np.concatenate((U0, np.zeros((sc["n0"], sc["n0"] - Na))), axis=1)

L1 = (B1.T) @ B1 + B2 @ (B2.T)

L1 = L1.asfptype()
Na = sc["n1"] - 1
D1, U1 = eigsh(L1, k=Na, sigma=0, which="LM")
D1 = np.concatenate((D1, 10000 * np.ones(sc["n1"] - Na)))
U1 = np.concatenate((U1, np.zeros((sc["n1"], sc["n1"] - Na))), axis=1)

Ds = [D0, D1]
Us = [U0, U1]
orders = [0, 1]
taus = [1, 1]
new_sc, mapnodes = renormalize.renormalize_simplicial_Dirac(
    sc,
    orders,
    Us,
    Ds,
    taus,
)

f, axs = plt.subplots(1, 2)
plotting.plot_complex(sc, axs[0], palette[0, :])
plotting.plot_complex(new_sc, axs[1], palette[1, :])
plt.show()
exit()


N = 500
d = 2
n_tau = 50
rep = 5
METHOD = "representative"
SPARSIFY = False
TRUE_CONNECTIONS = False

s = 1
beta = 0.1
ls = True  # logscale

suff = "simple"
pref = f"d{d}s{s}" + suff
path = f"hodge_renormalization/Hodge_renorm/Tests/Experiments_{METHOD}_{SPARSIFY}_{TRUE_CONNECTIONS}/{pref}"


with open(path + "/deg_dist.pkl", "rb") as f:
    deg_dist = pickle.load(f)


Ns = np.zeros((rep, d + 1, n_tau), dtype=int)
for r in range(rep):
    for i in range(d + 1):
        for t in range(n_tau):
            Ns[r, i, t] = len(deg_dist[r][t][i][0])


fig, axs = plt.subplots(d, d + 1, figsize=(2 * (d + 1), 2 * (d)))

compression = 0.4
tau_correct = np.zeros(d + 1, dtype=int)
chr = 0

for j in range(d + 1):
    print(Ns[0, j, :])

for j in range(d + 1):
    tau_correct[j] = np.argwhere(np.squeeze(Ns[chr, j, :]) <= (1 - compression) * N)[0]

for i in range(d):
    for j in range(d + 1):
        ax = axs[i, j]
        ddist = deg_dist[chr][tau_correct[j]][j][i]
        pwl.plot_pdf(deg_dist[chr][0][0][i], color="black", linewidth=2, ax=ax)
        pwl.plot_pdf(ddist, color="red", linewidth=2, ax=ax)


plt.show()
