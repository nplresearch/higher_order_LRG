import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw as pwl
from scipy.sparse.linalg import eigsh
from Functions import plotting, scomplex, renormalize
from scipy.stats import t
from scipy.signal import find_peaks




N = 500
d = 2
n_tau = 50
rep = 5
METHOD = 'representative'
SPARSIFY = False
TRUE_CONNECTIONS = False

s = 1
beta = 0.1
ls = True  # logscale

suff = "simple"
pref = f"d{d}s{s}"+suff
path = f"hodge_renormalization/Hodge_renorm/Tests/Experiments_{METHOD}_{SPARSIFY}_{TRUE_CONNECTIONS}/{pref}"


with open(path + "/deg_dist.pkl", "rb") as f:
    deg_dist = pickle.load(f)


Ns = np.zeros((rep, d + 1, n_tau), dtype=int)
for r in range(rep):
    for i in range(d + 1):
        for t in range(n_tau):
            Ns[r, i, t] = len(deg_dist[r][t][i][0])
            

fig,axs = plt.subplots(d,d+1,figsize = (2*(d+1),2*(d)))

compression = 0.4
tau_correct = np.zeros(d+1,dtype=int)
chr = 0

for j in range(d+1):
    print(Ns[0,j,:])

for j in range(d+1):
    tau_correct[j] = np.argwhere(np.squeeze(Ns[chr,j,:])<=(1-compression)*N)[0]

for i in range(d):
    for j in range(d+1):
        ax = axs[i,j]
        ddist = deg_dist[chr][tau_correct[j]][j][i]
        pwl.plot_pdf(deg_dist[chr][0][0][i], color='black', linewidth=2, ax = ax)
        pwl.plot_pdf(ddist, color='red', linewidth=2, ax = ax)
       
       
plt.show()