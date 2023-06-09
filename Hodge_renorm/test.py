import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw as pwl
from scipy.sparse.linalg import eigsh
from Functions import plotting, scomplex, renormalize
from scipy.stats import t


N = 100
d = 4
s = -1

beta = 0.1

sc = scomplex.NGF(d, N, s, beta)
B1, B2, B3, B4, edge_dict, face_dict, tet_dict = scomplex.boundary_matrices_3(sc)

L0 = (B4.T) @ B4
# print(B1.todense())
# exit()
L0 = L0.asfptype()
Na = sc["n4"] - 1
D0, U0 = eigsh(L0, k=Na, sigma=0, which="LM")
D0 = np.concatenate((D0, 10000 * np.ones(sc["n4"] - Na)))
U0 = np.concatenate((U0, np.zeros((sc["n4"], sc["n4"] - Na))), axis=1)
tau = 0.4
order = 4
L = L0
U = U0
D = D0

new_sc, mapnodes, comp, __ = renormalize.renormalize_simplicial_VARIANTS(
    sc,
    order,
    L,
    U,
    D,
    tau,
    "representative",
    False,
    False,
    1,
)

f, ax = plt.subplots()
plotting.plot_complex(new_sc, ax)
plt.show()
