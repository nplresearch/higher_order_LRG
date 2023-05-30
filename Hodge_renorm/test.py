import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Functions import plotting, renormalize, scomplex, support
from scipy.sparse.linalg import eigsh

N = 1000
order = 1
tau = 3
METHOD = "representative"
SPARSIFY = False
TRUE_CONNECTIONS = True
start = time.time()
sc = scomplex.NGF_d2(N, 1, 0.1)
end = time.time()
print(end - start)

start = time.time()
B1, B2, B3, __, __ = scomplex.boundary_matrices_3(sc)
end = time.time()
print(end - start)

if order == 0:
    nk = sc["n0"]
    L = B1 @ (B1.T)
elif order == 1:
    nk = sc["n1"]
    L = (B1.T) @ B1 + B2 @ (B2.T)
elif order == 2:
    nk = sc["n2"]
    L = (B2.T) @ B2 + B3 @ (B3.T)
elif order == 3:
    nk = sc["n3"]
    L = (B3.T) @ B3

L = L.asfptype()
Na = nk // 2
D, U = eigsh(L, k=Na, which="SM")
D = np.concatenate((D, 10000 * np.ones(nk - Na)))
U = np.concatenate((U, np.zeros((nk, nk - Na))), axis=1)


ax = plt.subplot(1, 2, 1)
plotting.plot_complex(sc, ax)
start = time.time()
new_sc, _, _, _ = renormalize.renormalize_simplicial_VARIANTS(
    sc, order, L, U, D, tau, METHOD, SPARSIFY, TRUE_CONNECTIONS
)
end = time.time()
print(end - start)
ax2 = plt.subplot(1, 2, 2)
plotting.plot_complex(new_sc, ax2)
plt.show()
