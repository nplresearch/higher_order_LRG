import os
import pickle
import sys

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

N = 500
d = 2
n_tau = 100
rep = 1
METHOD = "representative"
SPARSIFY = False
TRUE_CONNECTIONS = False

threshold = 0.9
s = 1
beta = 0.1

suff = "single"
pref = f"d{d}s{s}" + suff

deg_dist = []

for r in range(rep):
    rowr = []
    sc = scomplex.NGF(d, N, s, beta)

    B1, B2, B3, B4, edge_dict, face_dict, tet_dict = scomplex.boundary_matrices_3(sc)

    # Laplacians
    L0 = B1 @ (B1.T)

    L0 = L0.asfptype()
    Na = sc["n0"] - 1
    # D0,U0 = scipy.linalg.eig(L0.todense())
    D0, U0 = eigsh(L0, k=Na, sigma=0, which="LM")
    D0 = np.concatenate((D0, 10000 * np.ones(sc["n0"] - Na)))
    U0 = np.concatenate((U0, np.zeros((sc["n0"], sc["n0"] - Na))), axis=1)

    # tau_space0 = np.flip(1/D0)[0::sc["n0"]//n_tau]
    [specific_heat, tau_space] = renormalize.compute_heat(D0, -2, 1.5, 200)
    id, __ = find_peaks(specific_heat)
    tau_max0 = tau_space[id[0]]
    tau_space0 = np.linspace(0, tau_max0, n_tau)

    if sc["n1"] != 0:
        L1 = (B1.T) @ B1 + B2 @ (B2.T)
        L1 = L1.asfptype()
        Na = sc["n1"] - 1
        D1, U1 = eigsh(L1, k=Na, which="SM")
        D1 = np.concatenate((D1, 10000 * np.ones(sc["n1"] - Na)))
        U1 = np.concatenate((U1, np.zeros((sc["n1"], sc["n1"] - Na))), axis=1)
        # tau_space1 = np.flip(1/D1)[0::sc["n1"]//n_tau]

        [specific_heat, tau_space] = renormalize.compute_heat(D1, -2, 1.5, 200)
        id, __ = find_peaks(specific_heat)
        tau_max1 = tau_space[id[0]]
        tau_space1 = np.linspace(0, tau_max1, n_tau)

    if sc["n2"] != 0:
        L2 = (B2.T) @ B2 + B3 @ (B3.T)
        L2 = L2.asfptype()
        Na = sc["n2"] - 1
        D2, U2 = eigsh(L2, k=Na, sigma=0.0, which="LM")
        D2 = np.concatenate((D2, 10000 * np.ones(sc["n2"] - Na)))
        U2 = np.concatenate((U2, np.zeros((sc["n2"], sc["n2"] - Na))), axis=1)

        # tau_space2 = np.flip(1/D2)[0::sc["n2"]//n_tau]
        [specific_heat, tau_space] = renormalize.compute_heat(D2, -2, 1.5, 200)
        id, __ = find_peaks(specific_heat)
        tau_max2 = tau_space[id[0]]
        tau_space2 = np.linspace(0, tau_max2, n_tau)

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
        for i in range(d + 1):
            rowi = []
            order = i
            if order == 0:
                tau_space = tau_space0
                U = U0
                D = D0
                L = L0
            elif order == 1:
                tau_space = tau_space1
                U = U1
                D = D1
                L = L1
            elif order == 2:
                tau_space = tau_space2
                U = U2
                D = D2
                L = L2

            new_sc, mapnodes, comp, __ = renormalize.renormalize_simplicial_VARIANTS(
                sc,
                order,
                L,
                U,
                D,
                tau_space[t],
                METHOD,
                SPARSIFY,
                TRUE_CONNECTIONS,
                threshold,
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
with open(path + "/deg_dist.pkl", "wb") as f:
    pickle.dump(deg_dist, f)
