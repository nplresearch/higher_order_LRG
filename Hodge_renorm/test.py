import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw as pwl
import scipy
from Functions import plotting, scomplex
from scipy.stats import t

def coarsest_cover(U, V):
    subsets_U = generate_subsets(U)
    subsets_V = generate_subsets(V)

    # Find the coarsest cover
    coarsest_cover = []
    for subset_U in subsets_U:
        for subset_V in subsets_V:
            if is_cover_finer(subset_U, U) and is_cover_finer(subset_V, V):
                # Check if the current subsets form a coarser cover
                if not exists_coarser_cover(subset_U, subset_V, coarsest_cover):
                    coarsest_cover = subset_U + subset_V

    return coarsest_cover


def generate_subsets(cover):
    subsets = [[]]
    for subset_cover in cover:
        subsets += [subset + [element] for subset in subsets for element in subset_cover]
    return subsets


def is_cover_finer(subset, cover):
    return all(any(set(subset_element).issubset(set(element)) for element in cover_subset) for subset_element in subset for cover_subset in cover)


def exists_coarser_cover(subset_U, subset_V, coarsest_cover):
    return any(set(subset_U).issubset(set(existing_subset_U)) and set(subset_V).issubset(set(existing_subset_V)) for existing_subset_U, existing_subset_V in coarsest_cover)


U = [[1, 2, 3, 4], [4,5,6,1]]
V = [[6,1,2,3],[3,4,5,6]]

coarsest = coarsest_cover(U, V)
print(coarsest)






