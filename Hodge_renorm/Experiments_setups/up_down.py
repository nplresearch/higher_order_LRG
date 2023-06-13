import os
import pickle
import sys

sys.path.append(sys.path[0]+'/..')  # Adds higher directory to python modules path.

import numpy as np
from Functions import renormalize, scomplex
from scipy.signal import find_peaks
from scipy.sparse.linalg import eigsh

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

suff = "updown"
pref = f"d{d}s{s}" + suff

deg_dist = []

for r in range(rep):
    rowr = []
    sc = scomplex.NGF(d, N, s, beta)

    B1, B2, B3, B4, edge_dict, face_dict, tet_dict = scomplex.boundary_matrices_3(sc)

    L1 = (B1.T) @ B1 + B2 @ (B2.T)
    L1 = L1.asfptype()
    Na = sc["n1"] - 1
    D1, U1 = eigsh(L1, k=Na, which="SM")
    D1 = np.concatenate((D1, 10000 * np.ones(sc["n1"] - Na)))
    U1 = np.concatenate((U1, np.zeros((sc["n1"], sc["n1"] - Na))), axis=1)
    # tau_space1 = np.flip(1/D1)[0::sc["n1"]//n_tau]

    [specific_heat, tau_space] = renormalize.compute_heat(D1, -2, 1.5, 200)
    id, __ = find_peaks(specific_heat)
    tau_max1 = 3  # tau_space[id[0]]
    tau_space1 = np.linspace(0, tau_max1, n_tau)

    L1u = B2 @ (B2.T)
    L1u = L1u.asfptype()
    D1u, U1u = eigsh(L1u, k=Na, which="SM")
    D1u = np.concatenate((D1u, 10000 * np.ones(sc["n1"] - Na)))
    U1u = np.concatenate((U1u, np.zeros((sc["n1"], sc["n1"] - Na))), axis=1)

    [specific_heat, tau_space] = renormalize.compute_heat(D1u, -2, 1.5, 200)
    id, __ = find_peaks(specific_heat)
    tau_max1u = 10  # tau_space[id[0]]
    tau_space1u = np.linspace(0, tau_max1u, n_tau)

    L1d = (B1.T) @ B1
    L1d = L1d.asfptype()
    D1d, U1d = eigsh(L1d, k=Na, which="SM")
    D1d = np.concatenate((D1d, 10000 * np.ones(sc["n1"] - Na)))
    U1d = np.concatenate((U1d, np.zeros((sc["n1"], sc["n1"] - Na))), axis=1)

    [specific_heat, tau_space] = renormalize.compute_heat(D1d, -2, 1.5, 200)
    id, __ = find_peaks(specific_heat)
    tau_max1d = 10  # tau_space[id[0]]
    tau_space1d = np.linspace(0, tau_max1d, n_tau)

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
        for i in range(3):
            rowi = []
            if i == 0:
                tau_space = tau_space1
                U = U1
                D = D1
                L = L1
            elif i == 1:
                tau_space = tau_space1u
                U = U1u
                D = D1u
                L = L1u
            elif i == 2:
                tau_space = tau_space1d
                U = U1d
                D = D1d
                L = L1d

            new_sc, mapnodes, comp, __ = renormalize.renormalize_simplicial_VARIANTS(
                sc,
                1,
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
