import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw as pwl
from scipy.sparse.linalg import eigsh
from Functions import plotting, scomplex, renormalize
from scipy.stats import t


N = 2000
d = 4
s = -1

beta = 0.1

sc = scomplex.NGF(d, N, s, beta)
# f,ax = plt.subplots()
# plotting.plot_complex(sc,ax)
# plt.show()
