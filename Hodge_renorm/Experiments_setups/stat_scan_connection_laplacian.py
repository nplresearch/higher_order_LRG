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
from scipy.linalg import eigh
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

threshold = 0.8
s = 1
beta = 0.1

suff = ""
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
    [specific_heat, tau_space] = renormalize.compute_heat(D0, -2, 2, 200)
    id, __ = find_peaks(specific_heat)
    tau_max0 = tau_space[id[0]]
    tau_space0 = np.linspace(0, tau_max0, n_tau)

    if sc["n1"] != 0:
        L1 = (B1.T) @ B1 + B2 @ (B2.T)
        L1 = np.array(np.abs(L1 - np.diag(np.diag(L1.todense()))))
        L1 = np.diag(np.sum(L1, axis = 0,keepdims = 0)) - L1
        D1, U1 = eigh(L1)

        [specific_heat, tau_space] = renormalize.compute_heat(D1, -2, 2, 200)
        id, __ = find_peaks(specific_heat)
        tau_max1 = tau_space[id[0]]
        tau_space1 = np.linspace(0, tau_max1, n_tau)

    if sc["n2"] != 0:
        L2 = (B2.T) @ B2 + B3 @ (B3.T)
        L2 = np.array(np.abs(L2 - np.diag(np.diag(L2.todense()))))
        L2 = np.diag(np.sum(L2, axis = 0,keepdims = 0)) - L2
        D2, U2 = eigh(L2)
       
        [specific_heat, tau_space] = renormalize.compute_heat(D2, -2, 2, 200)
        id, __ = find_peaks(specific_heat)
        tau_max2 = tau_space[id[0]]
        tau_space2 = np.linspace(0, tau_max2, n_tau)

    if sc["n3"] != 0:
        L3 = (B3.T) @ B3 + B4 @ (B4.T)
        L3 = np.array(np.abs(L3 - np.diag(np.diag(L3.todense()))))
        L3 = np.diag(np.sum(L3, axis = 0,keepdims = 0)) - L3
        D3, U3 = eigh(L3)

        # tau_space3 = np.flip(1/D3)[0::sc["n3"]//n_tau]
        [specific_heat, tau_space] = renormalize.compute_heat(D3, -2, 2, 200)
        id, __ = find_peaks(specific_heat)
        tau_max3 = tau_space[id[0]]
        tau_space3 = np.linspace(0, tau_max3, n_tau)

    if sc["n4"] != 0:
        L4 = (B4.T) @ B4
        L4 = L4.asfptype()
        Na = sc["n4"] - 1
        D4, U4 = eigsh(L4, k=Na, which="SM")
        D4 = np.concatenate((D4, 10000 * np.ones(sc["n4"] - Na)))
        U4 = np.concatenate((U4, np.zeros((sc["n4"], sc["n4"] - Na))), axis=1)

        # tau_space4 = np.flip(1/D4)[0::sc["n4"]//n_tau]
        [specific_heat, tau_space] = renormalize.compute_heat(D4, -2, 1.5, 200)
        id, __ = find_peaks(specific_heat)
        tau_max4 = tau_space[id[0]]
        tau_space4 = np.linspace(0, tau_max4, n_tau)

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
            elif order == 3:
                tau_space = tau_space3
                U = U3
                D = D3
                L = L3
            elif order == 4:
                tau_space = tau_space4
                U = U4
                D = D4
                L = L4

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
