from itertools import combinations

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix, spdiags


def make_dict(sc):
    """
    Create dictionaries which associate the simplices to indices.

    Parameters
    ----------
    sc: dict
      Simplicial complex object

    Returns
    ----------
    edge_dict: dict
      Dictionary with keys given by the edges
    face_dict: dict
      Dictionary with keys given by the triangles
    tet_dict: dict
      Dictionary with keys given by the tetrahedra
    """

    edge_dict = {}
    for i, edge in enumerate(sc["edges"]):
        # Create a tuple to represent the edge (order doesn't matter)
        edge_key = tuple(sorted(edge))
        # Store the edge index in the dictionary
        edge_dict[edge_key] = i

    face_dict = {}
    for i, face in enumerate(sc["faces"]):
        # Create a tuple to represent the face (order doesn't matter)
        face_key = tuple(sorted(face))
        # Store the edge index in the dictionary
        face_dict[face_key] = i

    tet_dict = {}
    for i, tetrahedron in enumerate(sc["tetrahedra"]):
        # Create a tuple to represent the tetrahedron (order doesn't matter)
        tet_key = tuple(sorted(tetrahedron))
        # Store the edge index in the dictionary
        tet_dict[tet_key] = i

    return edge_dict, face_dict, tet_dict


def convert_graph_to_sc(G, dim=2):
    """
    Converts a graph G to its clique complex of a given dimension.

    Parameters
    ----------
    G: networkx graph object
    dim: int
      Dimension of the output simplicial complex

    Returns
    ----------
    sc: dict
      Simplicial complex object
    """

    G = nx.convert_node_labels_to_integers(G)
    N = len(G.nodes)
    sc = {
        "nodes": np.reshape(np.array(G.nodes), (-1, 1)),
        "n0": N,
        "edges": np.sort(np.array(G.edges), 1),
    }
    cliques = list(nx.enumerate_all_cliques(nx.from_edgelist(sc["edges"])))
    sc["n1"] = len(sc["edges"])

    if dim >= 2:
        sc["faces"] = np.array([x for x in cliques if len(x) == 3])
    else:
        sc["faces"] = np.zeros((0, 3))
    if dim >= 3:
        sc["tetrahedra"] = np.array([x for x in cliques if len(x) == 4])
    else:
        sc["tetrahedra"] = np.zeros((0, 4))

    sc["4-simplices"] = np.zeros((0, 5))
    sc["n2"] = sc["faces"].shape[0]
    if sc["n2"] > 0:
        sc["faces"] = np.sort(sc["faces"], 1)
    sc["n3"] = sc["tetrahedra"].shape[0]
    if sc["n3"] > 0:
        sc["tetrahedra"] = np.sort(sc["tetrahedra"], 1)
    sc["n4"] = 0
    return sc


def import_network_data(f, d):
    """
    Import a network from file and return its clique complex.

    Parameters
    ----------
    f: file
      Network file in edge list format
    d: int
      Maximal dimension of the cliques

    Returns
    ----------
    sc: dict
      Clique complex
    """

    i = 0
    edges = []
    for line in f:
        if i != 0:
            words = line.split()
            edges.append((words[0], words[1]))
        else:
            i += 1
    f.close()
    G = nx.from_edgelist(edges)
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    sc = convert_graph_to_sc(G, dim=d)

    return sc


def adjacency_of_order(sc, k, l, sparse=False):
    """
    Computes the (k,l)-adjacency matrix of a simplicial complex.

    Parameters
    ----------
    sc: dict
      Simplicial complex object
    k: int
      Order of the diffusing simplices
    l: int
      Order of the interaction simplices
    sparse: bool
      If True, return a sparse matrix

    Returns
    ----------
    adj: numpy array
      (k,l)-adjacency matrix
    """

    keys = ["nodes", "edges", "faces", "tetrahedra", "4-simplices"]
    nk = sc[f"n{k}"]
    if sparse:
        adj = lil_matrix((nk, nk), dtype=int)
    else:
        adj = np.zeros((nk, nk), dtype=int)

    assert (l != k), "The interaction order should be different from the order of the diffusing simplices"
    assert l >= 0, "The interaction order should be greater or equal than 0"
    assert (k >= 0), "The order of the diffusing simplices should be greater or equal than 0"
    assert (l <= 4) and (k <= 4), "Simplices of order greater than 4 are not supported"

    if l < k:
        diff_units = sc[keys[k]]
        for i in range(nk):
            for j in range(i + 1, nk):
                intersection = set(diff_units[i, :]) & set(diff_units[j, :])
                if len(intersection) == l + 1:
                    adj[i, j] += 1

    elif l > k:
        edge_dict, face_dict, tet_dict = make_dict(sc)
        dicts = [{(i,): i for i in range(sc["n0"])}, edge_dict, face_dict, tet_dict]
        int_simplices = sc[keys[l]]

        for i in range(sc[f"n{l}"]):
            simp = int_simplices[i, :]
            combs = list(combinations(simp, k + 1))
            ncombs = len(combs)
            combs_ids = np.zeros(ncombs, dtype=int)
            for n in range(ncombs):
                combs_ids[n] = dicts[k][combs[n]]
            combs_ids = np.sort(combs_ids)
            for n in range(ncombs):
                for m in range(n + 1, ncombs):
                    adj[combs_ids[n], combs_ids[m]] += 1

    if sparse:
        adj = csr_matrix(adj)

    return adj + adj.T


def adjacency_of_order_hg(sc, k, l):
    """
    Computes the (k,l)-adjacency matrix of a hypergraph.

    Parameters
    ----------
    sc: dict
      Simplicial complex object
    k: int
      Order of the diffusing simplices
    l: int
      Order of the interaction simplices

    Returns
    ----------
    adj: numpy array
      (k,l)-adjacency matrix
    """

    keys = ["nodes", "edges", "faces", "tetrahedra", "4-simplices"]
    nk = sc[f"n{k}"]
    adj = np.zeros((nk, nk), dtype=int)

    assert (l != k), "The interaction order should be different from the order of the diffusing simplices"
    assert l >= 0, "The interaction order should be greater or equal than 0"
    assert (k >= 0), "The order of the diffusing simplices should be greater or equal than 0"
    assert (l <= 4) and (k <= 4), "Simplices of order greater than 4 are not supported"

    if l < k:

        diff_units = sc[keys[k]]

        if l == 0:
            for i in range(nk):
                for j in range(i + 1, nk):
                    intersection = set(diff_units[i, :]) & set(diff_units[j, :])
                    adj[i, j] = 2 * len(intersection)

        else:
            edge_dict, face_dict, tet_dict = make_dict(sc)
            dicts = [{(i,): i for i in range(sc["n0"])}, edge_dict, face_dict, tet_dict]
            for i in range(nk):
                for j in range(i + 1, nk):
                    intersection = set(diff_units[i, :]) & set(diff_units[j, :])
                    combs = list(combinations(intersection, l + 1))
                    for c in combs:
                        if c in dicts[l]:
                            adj[i, j] += 2

    elif l > k:
        edge_dict, face_dict, tet_dict = make_dict(sc)
        dicts = [{(i,): i for i in range(sc["n0"])}, edge_dict, face_dict, tet_dict]
        for i, simp in enumerate(sc[keys[l]]):
            combs = list(combinations(simp, k + 1))
            ncombs = len(combs)
            combs_present = []
            for c in range(ncombs):
                if combs[c] in dicts[k]:
                    combs_present.append(dicts[k][combs[c]])

            for c1 in combs_present:
                for c2 in combs_present:
                    if c2 != c1:
                        adj[c1, c2] += 1

    return (adj + adj.T) // 2


def XO_laplacian(sc, k, l, sparse=False):
    """
    Computes the (k,l)-cross-order Laplacian matrix of a simplicial complex.

    Parameters
    ----------
    sc: dict
      Simplicial complex object
    k: int
      Order of the diffusing simplices
    l: int
      Order of the interaction simplices
    sparse: bool
      If True, return a sparse matrix

    Returns
    ----------
    L: numpy array
      (k,l)-cross-order Laplacian matrix
    """

    A = adjacency_of_order(sc, k, l, sparse)
    K = np.sum(A, 0)
    if sparse:
        lenK = K.shape[1]
        L = spdiags(K, 0, lenK, lenK) - A
    else:
        L = np.diag(K) - A
    return L


def XO_laplacian_hg(sc, k, l):
    """
    Computes the (k,l)-cross-order Laplacian matrix of a hypergraph.

    Parameters
    ----------
    sc: dict
      Simplicial complex object
    k: int
      Order of the diffusing simplices
    l: int
      Order of the interaction simplices

    Returns
    ----------
    L: numpy array
      (k,l)-cross-order Laplacian matrix
    """

    A = adjacency_of_order_hg(sc, k, l)
    K = np.sum(A, 0)
    L = np.diag(K) - A
    return L


def pseudofractal_d2(steps):
    """
    Generates a 2-dimensional pseudofractal simplicial complex.

    Parameters
    ----------
    steps: int
      Number of construction iterations

    Returns
    ----------
    sc: dict
      Simplicial complex object
    """

    edges = [(0, 1), (1, 2), (0, 2)]
    n = 3
    for s in range(steps):
        boundary = edges.copy()
        for ed in boundary:
            edges.append((ed[0], n))
            edges.append((ed[1], n))
            n += 1

    G = nx.from_edgelist(edges)
    sc = convert_graph_to_sc(G, dim=2)
    return sc


def pseudofractal_d3(steps):
    """
    Generates a 3-dimensional pseudofractal simplicial complex.

    Parameters
    ----------
    steps: int
      Number of construction iterations

    Returns
    ----------
    sc: dict
      Simplicial complex object
    """

    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    n = 4
    for s in range(steps):
        boundary = faces.copy()
        for fa in boundary:
            edges.append((fa[0], n))
            edges.append((fa[1], n))
            edges.append((fa[2], n))
            faces.append((fa[0], fa[1], n))
            faces.append((fa[0], fa[2], n))
            faces.append((fa[1], fa[2], n))
            n += 1

    G = nx.from_edgelist(edges)
    sc = convert_graph_to_sc(G, dim=3)

    return sc


def apollonian_d2(steps):
    """
    Generates a 2-dimensional apollonian simplicial complex

    Parameters
    ----------
    steps: int
      Number of construction iterations

    Returns
    ----------
    sc: dict
      Simplicial complex object
    """

    edges = [(0, 1), (1, 2), (0, 2)]
    new_boundary = edges.copy()
    n = 3
    for s in range(steps):
        boundary = new_boundary
        new_boundary = []
        for ed in boundary:
            edges.append((ed[0], n))
            edges.append((ed[1], n))
            new_boundary.append((ed[0], n))
            new_boundary.append((ed[1], n))
            n += 1

    G = nx.from_edgelist(edges)
    sc = convert_graph_to_sc(G, dim=2)

    return sc


def NGF(d, N, s, beta, M=1):
    """
    Generates a Nework Geometry with Flavour (NGF) simplicial complex, as described in
    G. Bianconi and C. Rahmede "Network geometry with flavor: from complexity to quantum
     geometry" Physical Review E 93, 032315 (2016).
    The code is adapted from https://github.com/ginestrab/Network-Geometry-with-Flavor.

    Parameters
    ----------
    d: int
      Dimension of the simplicial complex
    N: int
      Number of nodes
    s: int
      Flavour of the NGF. Can be -1, 0 or 1
    beta: float
      Inverse temperature of the construction process
    M: int
      Number of simplices to attach at each step

    Returns
    ----------
    sc: dict
      Simplicial complex object
    """

    if d > 4:
        print("Dimension out of bounds. NGF implemented only for d=1,2,3,4")

    kappa = 1
    epsilon = np.random.rand(N) ** (1 / (kappa + 1))
    a = sp.lil_matrix((N, N))
    at = np.array([])
    a_occ = np.array([])
    node = np.zeros(((d + 1) + (N - (d + 1)) * d, d), dtype=int)

    # Initial condition: at time t=1 a single d-dimensional simplex (1,2,3,4)
    for i in range(d + 1):
        for j in range(i + 1, d + 1):
            a[i, j] = 1
            a[j, i] = 1

    for nt in range(d + 1):
        at = np.append(at, 1)
        a_occ = np.append(a_occ, 1)
        j = -1
        for i in range(d + 1):
            if i != nt:
                j += 1
                node[nt, j] = i
                at[nt] = at[nt] * np.exp(-beta * epsilon[i])

    it = d

    while it < N - 1:
        it += 1
        for m in range(M):
            mat = at * a_occ
            J = mat.nonzero()[0]
            V = np.squeeze(mat[mat.nonzero()])
            norm = np.sum(V)
            x = np.random.rand() * norm
            for nj1 in range(len(V)):
                x -= V[nj1]
                if x < 0:
                    nj = J[nj1]  # Index of the d-1 simplex where the next simplex is attached
                    break

            a_occ[nj] = a_occ[nj] + s

            # Attach the next simplex
            for n1 in range(d):
                j = node[nj, n1]
                a[it, j] = 1
                a[j, it] = 1

        for n1 in range(d):  # d (d-1)-simplices are added
            nt += 1
            at = np.append(at, 1)
            a_occ = np.append(a_occ, 1)
            node[nt, 0] = it
            j = 0
            for n2 in range(d):
                if n2 != n1:
                    j += 1
                    node[nt, j] = node[nj, n2]
                    at[nt] = at[nt] * np.exp(-beta * epsilon[node[nj, n2]])
                    a[it, node[nj, n2]] = 1
                    a[node[nj, n2], it] = 1

    a = a > 0
    G = nx.from_numpy_array(a)
    cliques = list(nx.enumerate_all_cliques(G))
    faces = []
    tetrahedra = []
    four_simplexes = []
    for c in cliques:
        l = len(c)
        if l == 3:
            faces.append(c)
        elif l == 4:
            tetrahedra.append(c)
        elif l == 5:
            four_simplexes.append(c)

    sc = {
        "nodes": np.reshape(np.arange(0, N), (N, 1)),
        "edges": np.unique(
            np.sort(np.array(list(G.edges()), dtype=int), axis=1), axis=0
        ),
        "faces": np.unique(
            np.sort(np.reshape(np.array(faces, dtype=int), (-1, 3)), axis=1), axis=0
        ),
        "tetrahedra": np.unique(
            np.sort(np.reshape(np.array(tetrahedra, dtype=int), (-1, 4)), axis=1),
            axis=0,
        ),
        "4-simplices": np.unique(
            np.sort(np.reshape(np.array(four_simplexes, dtype=int), (-1, 5)), axis=1),
            axis=0,
        ),
    }
    sc["n0"] = N
    sc["n1"] = sc["edges"].shape[0]
    sc["n2"] = sc["faces"].shape[0]
    sc["n3"] = sc["tetrahedra"].shape[0]
    sc["n4"] = sc["4-simplices"].shape[0]

    return sc
