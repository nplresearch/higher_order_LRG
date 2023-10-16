import numpy as np


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n

    def find(self, p):
        root = p
        while root != self.parent[root]:
            root = self.parent[root]
        while p != root:
            next_p = self.parent[p]
            self.parent[p] = root
            p = next_p
        return root

    def union(self, p, q):
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p == root_q:
            return
        if self.size[root_p] < self.size[root_q]:
            self.parent[root_p] = root_q
            self.size[root_q] += self.size[root_p]
        else:
            self.parent[root_q] = root_p
            self.size[root_p] += self.size[root_q]
        self.count -= 1

    def connected_components(self):
        return self.count

    def add_edges(self, edges):
        for u, v in edges:
            self.union(u, v)


def meet(P, Q):
    PmQ = set()
    for p in P:
        if len(p) == 1:
            PmQ.add(p)
        else:
            for q in Q:
                inter = p.intersection(q)
                if len(inter) != 0:
                    PmQ.add(p.intersection(q))
    return PmQ


def map2partition(mapnodes, nc):
    P = set()
    N = len(mapnodes)
    for n in range(nc):
        ag = np.reshape(np.argwhere(mapnodes == n), (-1,))
        P.add(frozenset(list(ag)))
    return P


def partition2map(P, N):
    nc = len(P)
    mapnodes = np.zeros(N)
    n = 0
    for p in P:
        for node in list(p):
            mapnodes[node] = n
        n += 1
    return mapnodes, nc


def list_dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + list_dim(a[0])
