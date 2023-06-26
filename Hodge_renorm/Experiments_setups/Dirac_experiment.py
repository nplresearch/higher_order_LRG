import os
import pickle
import sys

sys.path.append(sys.path[0] + "/..")  # Adds higher directory to python modules path.

import numpy as np
from Functions import renormalize, scomplex, plotting
from scipy.io import savemat
from scipy.signal import find_peaks
from scipy.sparse.linalg import eigsh

# Pass arguments from command line:
# python stat_scan.py N d n_tau rep METHOD SPARSIFY TRUE_CONNECTIONS

N = 1000
d = 2
n_tau = 50
rep = 2

orders = [0, 2]

threshold = 0.8
s = 1
beta = 0.1

name = "Dirac"
deg_dist = []

for r in range(rep):
    rowr = []
    sc = scomplex.NGF(d, N, s, beta)

    B1, B2, B3, B4, edge_dict, face_dict, tet_dict = scomplex.boundary_matrices_3(sc)

    Ds = []
    Us = []
    tau_spaces = []

    for o in range(2):
        order = orders[o]
        if order == 0:
            L = B1 @ (B1.T)
            L = L.asfptype()
            Na = sc["n0"] - 1
            D, U = eigsh(L, k=Na, sigma=0, which="LM")
            D = np.concatenate((D, 10000 * np.ones(sc["n0"] - Na)))
            U = np.concatenate((U, np.zeros((sc["n0"], sc["n0"] - Na))), axis=1)

        elif order == 1:
            L = (B1.T) @ B1 + B2 @ (B2.T)
            L = L.asfptype()
            Na = sc["n1"] - 1
            D, U = eigsh(L, k=Na, which="SM")
            D = np.concatenate((D, 10000 * np.ones(sc["n1"] - Na)))
            U = np.concatenate((U, np.zeros((sc["n1"], sc["n1"] - Na))), axis=1)

        elif order == 2:
            L = (B2.T) @ B2 + B3 @ (B3.T)
            L = L.asfptype()
            Na = sc["n2"] - 1
            D, U = eigsh(L, k=Na, sigma=0.0, which="LM")
            D = np.concatenate((D, 10000 * np.ones(sc["n2"] - Na)))
            U = np.concatenate((U, np.zeros((sc["n2"], sc["n2"] - Na))), axis=1)

        elif order == 3:
            L = (B3.T) @ B3 + B4 @ (B4.T)
            L = L.asfptype()
            Na = sc["n3"] - 1
            D, U = eigsh(L, k=Na, which="SM")
            D = np.concatenate((D, 10000 * np.ones(sc["n3"] - Na)))
            U = np.concatenate((U, np.zeros((sc["n3"], sc["n3"] - Na))), axis=1)

        if order == 4:
            L = (B4.T) @ B4
            L = L.asfptype()
            Na = sc["n4"] - 1
            D, U = eigsh(L, k=Na, which="SM")
            D = np.concatenate((D, 10000 * np.ones(sc["n4"] - Na)))
            U = np.concatenate((U, np.zeros((sc["n4"], sc["n4"] - Na))), axis=1)

        [specific_heat, tau_space] = renormalize.compute_heat(D, -2, 1.5, 200)
        id, __ = find_peaks(specific_heat)
        tau_max = tau_space[id[0]]
        tau_spaces.append(np.linspace(0, tau_max, n_tau))
        Ds.append(D)
        Us.append(U)

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
        taus = [tau_spaces[0][t], tau_spaces[1][t]]

        rowi = []
        new_sc, mapnodes = renormalize.renormalize_simplicial_Dirac(
            sc,
            orders,
            Us,
            Ds,
            taus,
        )
        print(new_sc["n0"])
        new_edge_dict, new_face_dict, new_tet_dict = scomplex.make_dict(new_sc)

        new_deg = scomplex.generalized_degree(
            new_sc, new_edge_dict, new_face_dict, new_tet_dict, d
        )

        # repetitions x renorml x deg type x tau

        for j in range(d):
            rowi.append(new_deg[j])

        rowt.append(rowi)
        rowr.append(rowt)
    deg_dist.append(rowr)


plotting.plot_deg_dist(deg_dist)
exit()
path = "Tests/Experiment" + name


if not os.path.exists(path):
    os.makedirs(path)

with open(path + "/deg_dist.pkl", "wb") as f:
    pickle.dump(deg_dist, f)
