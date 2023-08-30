import os
import pickle
import sys

sys.path.append(sys.path[0] + "/..")  # Adds higher directory to python modules path.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Functions import renormalize, scomplex, plotting
from scipy.io import savemat
from scipy.signal import find_peaks
from scipy.sparse.linalg import eigsh
import scipy

# Pass arguments from command line:
# python stat_scan.py N d n_tau rep METHOD SPARSIFY TRUE_CONNECTIONS

N = int(sys.argv[1])
d = int(sys.argv[2])
n_tau = int(sys.argv[3])
rep = int(sys.argv[4])
METHOD = sys.argv[5]  # {"representative","closest"}
SPARSIFY = sys.argv[6] == "True"
TRUE_CONNECTIONS = sys.argv[7] == "True"

threshold = 1
s = 1
beta = 0.1

suff = "_multiorder"
pref = f"d{d}s{s}" + suff

deg_dist = []

for r in range(rep):
    rowr = []
    sc = scomplex.NGF(d, N, s, beta)

    # Laplacians
    L1 = scomplex.laplacian_of_order(sc, 1)
    L1 = L1.asfptype()
    Na = sc["n0"] - 1
    D1, U1 = eigsh(L1, k=Na, sigma=0, which="LM")
    D1 = np.concatenate((D1, 10000 * np.ones(sc["n0"] - Na)))
    U1 = np.concatenate((U1, np.zeros((sc["n0"], sc["n0"] - Na))), axis=1)
    [specific_heat, tau_space] = renormalize.compute_heat(D1, -2, 1.5, 200)
    id, __ = find_peaks(specific_heat)
    tau_max1 = tau_space[id[0]]
    tau_space1 = np.linspace(0, tau_max1, n_tau)

    if sc["n2"] != 0:
        L2 = scomplex.laplacian_of_order(sc, 2)
        L2 = L2.asfptype()
        Na = sc["n0"] - 1
        # D0,U0 = scipy.linalg.eig(L0.todense())
        D2, U2 = eigsh(L2, k=Na, sigma=0, which="LM")
        D2 = np.concatenate((D2, 10000 * np.ones(sc["n0"] - Na)))
        U2 = np.concatenate((U2, np.zeros((sc["n0"], sc["n0"] - Na))), axis=1)
        [specific_heat, tau_space] = renormalize.compute_heat(D2, -2, 1.5, 200)
        id, __ = find_peaks(specific_heat)
        tau_max2 = tau_space[id[0]]
        tau_space2 = np.linspace(0, tau_max2, n_tau)

    if sc["n3"] != 0:
        L3 = scomplex.laplacian_of_order(sc, 3)
        L3 = L3.asfptype()
        Na = sc["n0"] - 1
        # D0,U0 = scipy.linalg.eig(L0.todense())
        D3, U3 = eigsh(L3, k=Na, sigma=0, which="LM")
        D3 = np.concatenate((D3, 10000 * np.ones(sc["n0"] - Na)))
        U3 = np.concatenate((U3, np.zeros((sc["n0"], sc["n0"] - Na))), axis=1)
        [specific_heat, tau_space] = renormalize.compute_heat(D3, -2, 1.5, 200)
        id, __ = find_peaks(specific_heat)
        tau_max3 = tau_space[id[0]]
        tau_space3 = np.linspace(0, tau_max3, n_tau)

    for t in range(n_tau):
        rowt = []
        print(
            "rep: "
            + str(r + 1)
            + "/"
            + str(rep)
            + ", t: "
            + str(t + 1)
            + "/"
            + str(n_tau)
        )
        for i in range(d):
            rowi = []
            order = i + 1
            if order == 1:
                tau_space = tau_space1
                U = U1
                D = D1
                L = L1
            elif order == 2:
                tau_space = tau_space2
                U = U2
                D = D2
                L = L2
            elif order == 3:
                tau_space = tau_space3
                U = U3
                D = D3
                L = L3

            new_sc, mapnodes, comp, __ = renormalize.renormalize_simplicial_VARIANTS(
                sc,
                0,
                L,
                U,
                D,
                tau_space[t],
                METHOD,
                SPARSIFY,
                TRUE_CONNECTIONS,
                threshold=1,
                simple=True,
            )
            print(new_sc["n0"])
            new_edge_dict, new_face_dict, new_tet_dict = scomplex.make_dict(new_sc)

            new_deg = scomplex.generalized_degree(
                new_sc, new_edge_dict, new_face_dict, new_tet_dict, d
            )
            # repetitions x Lk x deg type x tau
            for j in range(d):
                rowi.append(new_deg[j])
            rowt.append(rowi)
        rowr.append(rowt)
    deg_dist.append(rowr)


path = f"Tests/Experiments_{METHOD}_{SPARSIFY}_{TRUE_CONNECTIONS}/{pref}"

if not os.path.exists(path):
    os.makedirs(path)

with open(path + "/deg_dist.pkl", "wb") as f:
    pickle.dump(deg_dist, f)
