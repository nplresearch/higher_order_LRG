from itertools import groupby

import networkx as nx
import numpy as np

from Scripts import scomplex


def compute_entropic_C(D, exm, exM, n_t):
    """
    Computes the Von Neumann entropy and Entropic susceptibility of a given Laplacian.

    Parameters
    ----------
    D: list of floats
      List of eigenvalues of the Laplacian matrix considered
    exm, exM: floats
    n_t: int
      Computes the quantities in n_t logarithmically spaced time points in the interval [10**exm,10**exM]

    Returns
    ----------
    specific_heat: numpy array
      Entropic susceptibility values
    tau_space: numpy array
      n_t - 1 logarithmically spaced time points
    S: numpy array
      Von Neumann entropy
    """

    N = len(D)
    D = np.abs(D)
    tau_space = np.logspace(exm, exM, num=n_t)
    S = np.zeros(n_t)
    for t in range(n_t):
        tau = tau_space[t]
        mu = np.zeros(N)
        for i in range(N):
            mu[i] = 1 / np.sum(np.exp(-tau * (D - D[i])))

        mu = mu[mu > 0]
        S[t] = -np.sum(mu * np.log(mu))
    entropic_susceptibility = -(np.diff(S) / np.diff(np.log(tau_space)))
    tau_space = tau_space[: n_t - 1]
    return entropic_susceptibility, tau_space, S


def compute_spectral_d(D, exm, exM, n_t):
    """
    Computes the spectral dimension associated to a diffusion process

    Parameters
    ----------
    D: list
      eigenvalues of the Laplacian matrix considered
    exm, exM: floats
    n_t: int
      Computes the quantities in n_t logarithmically spaced time points in the interval [10**exm,10**exM]

    Returns
    ----------
    dS: numpy array
      Spectral dimension values
    tau_space: numpy array
      n_t - 1 logarithmically spaced time points
    """

    tau_space = np.logspace(exm, exM, num=n_t)
    Z = np.zeros(n_t)
    for t in range(n_t):
        Z[t] = np.sum(np.exp(-tau_space[t] * D))

    dS = -2 * np.diff(np.log(Z)) / np.diff(np.log(tau_space))

    return dS, tau_space[1:]


def measure_SI(tau_space, sp_heat, epsilon=0.1, ymin=-5, ymax=1, ny=70):
    """
    Computes the scale-invariance parameter of an entropic susceptibility curve.

    Parameters
    ----------
    tau_space: numpy array
      Times in which the entropic susceptibility has been computed
    sp_heat: numpy array
      Entropic susceptibility curve
    epsilon: float
      Plateau threshold
    ymin: float
    ymax: float
      Respectively, the minimum and maximum value of log C to scan for plateaus
    ny: int
      Number of points in the interval [ymin,ymax] to scan for plateaus

    Returns
    ----------
    SIP: scale-invariance parameter
    """

    max_plateau = 0
    sp_heat = np.log(sp_heat)
    for y in np.linspace(ymin, ymax, ny):
        mask = np.abs(sp_heat - y) < epsilon
        list_s = [[a, len(list(k))] for a, k in groupby(mask)]
        for j in range(len(list_s)):
            if list_s[j][0]:
                if list_s[j][1] > max_plateau:
                    max_plateau = list_s[j][1]

    SIP = max_plateau * np.log(tau_space[1] / tau_space[0])
    return SIP


def induce_simplices(sc, mapnodes):
    """
    Finds induced simplices in the simplicial complex after its nodes are coarse grained.

    Parameters
    ----------
    sc: dict
      Simplicial complex object
    mapnodes: list of ints
      Mapping from each node in sc to the label of its signature

    Returns
    ----------
    new_sc: dict
      Coarse grained simplicial complex object
    """

    new_sc = {
        "nodes": np.sort(np.unique(mapnodes)),
    }
    new_sc["n0"] = len(new_sc["nodes"])
    new_sc["nodes"] = np.reshape(new_sc["nodes"], (new_sc["n0"], 1))

    # Connect supernodes with edges
    new_edges = []
    for i in range(sc["n1"]):
        node1 = mapnodes[sc["edges"][i, 0]]
        node2 = mapnodes[sc["edges"][i, 1]]
        if node1 != node2:
            new_edges.append([node1, node2])
    if len(new_edges) != 0:
        new_sc["edges"] = np.unique(
            np.sort(np.array(new_edges, dtype=int), axis=1), axis=0
        )
        new_sc["n1"] = new_sc["edges"].shape[0]
    else:
        new_sc["edges"] = np.zeros((0, 2), dtype=int)
        new_sc["n1"] = 0

    # Connect supernodes with triangles
    new_faces = []
    for i in range(sc["n2"]):
        node1 = mapnodes[sc["faces"][i][0]]
        node2 = mapnodes[sc["faces"][i][1]]
        node3 = mapnodes[sc["faces"][i][2]]
        if len(np.unique([node1, node2, node3])) == 3:
            new_faces.append([node1, node2, node3])
    if len(new_faces) != 0:
        new_sc["faces"] = np.unique(
            np.sort(np.array(new_faces, dtype=int), axis=1), axis=0
        )
        new_sc["n2"] = new_sc["faces"].shape[0]
    else:
        new_sc["faces"] = np.zeros((0, 3), dtype=int)
        new_sc["n2"] = 0

    # Connect supernodes with tetrahedra
    new_tetrahedra = []
    for i in range(sc["n3"]):
        node1 = mapnodes[sc["tetrahedra"][i][0]]
        node2 = mapnodes[sc["tetrahedra"][i][1]]
        node3 = mapnodes[sc["tetrahedra"][i][2]]
        node4 = mapnodes[sc["tetrahedra"][i][3]]
        if len(np.unique([node1, node2, node3, node4])) == 4:
            new_tetrahedra.append([node1, node2, node3, node4])
    if len(new_tetrahedra) != 0:
        new_sc["tetrahedra"] = np.unique(
            np.sort(np.array(new_tetrahedra, dtype=int), axis=1), axis=0
        )
        new_sc["n3"] = new_sc["tetrahedra"].shape[0]
    else:
        new_sc["tetrahedra"] = np.zeros((0, 4), dtype=int)
        new_sc["n3"] = 0

    # Connect supernodes with 4-simplices
    new_4_simplices = []
    for i in range(sc["n4"]):
        nodes = mapnodes[sc["4-simplices"][i, :]]
        if len(np.unique(nodes)) == 5:
            new_4_simplices.append(nodes)
    if len(new_4_simplices) != 0:
        new_sc["4-simplices"] = np.unique(
            np.sort(np.array(new_4_simplices, dtype=int), axis=1), axis=0
        )
        new_sc["n4"] = new_sc["4-simplices"].shape[0]
    else:
        new_sc["4-simplices"] = np.zeros((0, 5), dtype=int)
        new_sc["n4"] = 0

    return new_sc


def coarse_grain(sc, order, comp, ncomp):
    """
    Coarse grains the simplicial complex according to a partition of the k-simplices.

    Parameters
    ----------
    sc: dict
      Simplicial complex object
    order: int
      Order of the simplices which are partitioned
    comp: list of ints
      List of labels specifying the partition of the order-simplices
    ncomp: int
      Total number of labels

    Returns
    ----------
    mapnodes: list of ints
      Mapping from each node in sc to the label of its signature
    nodesclusters: list of sets
      Mapping from each node to its signature
    """

    name = f"n{order}"
    nk = sc[name]
    keys = ["nodes", "edges", "faces", "tetrahedra", "4-simplices"]
    simplices = keys[order]

    nodesclusters = [set() for _ in range(sc["n0"])]  # Map nodes to their clusters
    # Assign labels to nodes
    for i in range(nk):
        nodes = sc[simplices][i, :]  # Nodes in simplex i
        for j in range(order + 1):
            nodesclusters[nodes[j]] = nodesclusters[nodes[j]].union({comp[i]})

    # Give labels to unlabeled nodes
    id = ncomp
    for i in range(sc["n0"]):
        if len(nodesclusters[i]) == 0:
            nodesclusters[i] = nodesclusters[i].union({id})
            id += 1

    # for i in range(len(nodesclusters)):
    for i in range(sc["n0"]):
        nodesclusters[i] = np.array2string(np.sort(list(nodesclusters[i])))

    uq = np.unique(nodesclusters)
    d = {b: a for a, b in enumerate(uq)}

    mapnodes = np.zeros(
        sc["n0"], dtype=int
    )  # Maps each node to its image in the renormalized simplicial complex

    for i in range(sc["n0"]):
        mapnodes[i] = d[nodesclusters[i]]

    return mapnodes, nodesclusters


def renormalize_steps(sc, lmax, tau, diff_order=0, int_order=1, VERBOSE=False):
    """
    Performs multiple steps of the simplicial renormalization flow.

    Parameters
    ----------
    sc: dict
      Simplicial complex object
    lmax: int
      Number of renormalization steps
    tau: float
      Diffusion time for each step
    diff_order: int
      Order of the diffusing simplices
    int_order: int
      Order of the interacting simplices
    VERBOSE: bool
      If True print the number of nodes at each step

    Returns
    ----------
    sequence: list
      List of the renormalized simplicial complexes
    """

    if len(np.shape(tau)) == 0:
        tau = [tau for i in range(lmax)]

    sequence = []
    new_sc = sc
    for l in range(lmax):
        if l > 0 and new_sc["n0"] > 1:
            L = scomplex.XO_laplacian(new_sc, diff_order, int_order)
            D, U = np.linalg.eigh(L)
            rho = np.abs(U @ np.diag(np.exp(-tau[l] * D)) @ U.T)  # Heat kernel

            Gv = nx.Graph()
            Gv.add_nodes_from([i for i in range(new_sc[f"n{diff_order}"])])
            for i in range(new_sc[f"n{diff_order}"]):
                for j in range(i + 1, new_sc[f"n{diff_order}"]):
                    if rho[i, j] >= min(rho[i, i], rho[j, j]):
                        Gv.add_edge(i, j)

            idx_components = {
                u: i
                for i, node_set in enumerate(nx.connected_components(Gv))
                for u in node_set
            }
            clusters = [idx_components[u] for u in Gv.nodes]

            mapnodes, __ = coarse_grain(
                new_sc, diff_order, clusters, np.max(clusters) + 1
            )
            new_sc = induce_simplices(new_sc, mapnodes)

        if VERBOSE:
            print(new_sc["n0"])

        sequence.append(new_sc)

    return sequence


def renormalize_single_step(
    sc, tau, diff_order=0, int_order=1, D=None, U=None, VERBOSE=True
):
    """
    Performs a single step of higher-order Laplacian renormalization.

    Parameters
    ----------
    sc: dict
      Simplicial complex object
    tau: float
      Diffusion time
    diff_order: int
      Order of the diffusing simplices
    int_order: int
      Order of the interaction simplices
    D: list of floats
      The list of Laplacian eigenvlaues, if None computes them from scratch
    U: numpy array of size (len(D),len(D))
      Matrix of Laplacian eigenvectors, if None computes them from scratch
    VERBOSE: bool
      If True print the number of nodes after the coarse-graining

    Returns
    ----------
    new_sc: dict
      Renormalized simplicial complex
    mapnodes: list of ints
      List associating to each node in sc the node in new_sc it is mapped to
    clusters: list of ints
      Cluster label of each simplex of order diff_order
    """

    if (D is None) or (U is None):
        L = scomplex.XO_laplacian(sc, diff_order, int_order)
        D, U = np.linalg.eigh(L)

    rho = np.abs(U @ np.diag(np.exp(-tau * D)) @ U.T)

    Gv = nx.Graph()
    Gv.add_nodes_from([i for i in range(sc[f"n{diff_order}"])])
    for i in range(sc[f"n{diff_order}"]):
        for j in range(i + 1, sc[f"n{diff_order}"]):
            if rho[i, j] >= min(rho[i, i], rho[j, j]):
                Gv.add_edge(i, j)

    idx_components = {
        u: i for i, node_set in enumerate(nx.connected_components(Gv)) for u in node_set
    }
    clusters = [idx_components[u] for u in Gv.nodes]

    mapnodes, __ = coarse_grain(sc, diff_order, clusters, np.max(clusters) + 1)
    new_sc = induce_simplices(sc, mapnodes)

    if VERBOSE:
        print(new_sc["n0"])

    return new_sc, mapnodes, clusters


# Hypergraph functions


def induce_simplices_hg(sc, mapnodes):
    """
    Finds induced hyperedges in the hypergraph after its nodes are coarse grained.

    Parameters
    ----------
    sc: dict
      Hypergraph object
    mapnodes: list of ints
      Mapping from each node in sc to the label of its signature

    Returns
    ----------
    new_sc: dict
      Coarse grained hypergraph
    """

    keys = ["edges", "faces", "tetrahedra", "4-simplices"]
    new_sc = {
        "nodes": np.sort(np.unique(mapnodes)),
    }
    new_sc["n0"] = len(new_sc["nodes"])
    new_sc["nodes"] = np.reshape(new_sc["nodes"], (new_sc["n0"], 1))
    for key in keys:
        new_sc[key] = []

    # Connect supernodes with hyperedges
    for order, key in enumerate(keys):
        for i in range(sc[f"n{order+1}"]):
            nodes = mapnodes[sc[key][i, :]]
            un = np.unique(nodes)
            lun = len(un)
            if lun > 1:
                new_sc[keys[lun - 2]].append(un)

    # Remove duplicate hyperedges
    for order, key in enumerate(keys):
        if len(new_sc[key]) != 0:
            new_sc[key] = np.unique(
                np.sort(np.array(new_sc[key], dtype=int), axis=1), axis=0
            )
            new_sc[f"n{order+1}"] = new_sc[key].shape[0]
        else:
            new_sc[key] = np.zeros((0, order + 2), dtype=int)
            new_sc[f"n{order+1}"] = 0

    return new_sc


def renormalize_single_step_hg(
    sc, tau, diff_order=0, int_order=1, D=None, U=None, VERBOSE=True
):
    """
    Performs a single step of higher-order Laplacian renormalization for a hypergraph.

    Parameters
    ----------
    sc: dict
      Hypergraph object
    tau: float
      Diffusion time
    diff_order: int
      Order of the diffusing hyperedges
    int_order: int
      Order of the interaction hyperedges
    D: list
      List of Laplacian eigenvlaues, if None computes them from scratch
    U: numpy array
      Matrix of Laplacian eigenvectors, if None computes them from scratch
    VERBOSE: bool
      If True print the number of nodes after the coarse-graining

    Returns
    ----------
    new_sc: dict
      Renormalized hypergraph
    mapnodes: list of ints
      List associating to each node in sc the node in new_sc it is mapped to
    clusters: list of ints
      Cluster label of each simplex of order diff_order
    """

    if (D is None) or (U is None):
        L = scomplex.XO_laplacian_hg(sc, diff_order, int_order)
        D, U = np.linalg.eigh(L)

    rho = np.abs(U @ np.diag(np.exp(-tau * D)) @ U.T)

    Gv = nx.Graph()
    Gv.add_nodes_from([i for i in range(sc[f"n{diff_order}"])])
    for i in range(sc[f"n{diff_order}"]):
        for j in range(i + 1, sc[f"n{diff_order}"]):
            if rho[i, j] >= min(rho[i, i], rho[j, j]):
                Gv.add_edge(i, j)

    idx_components = {
        u: i for i, node_set in enumerate(nx.connected_components(Gv)) for u in node_set
    }
    clusters = [idx_components[u] for u in Gv.nodes]

    mapnodes, __ = coarse_grain(sc, diff_order, clusters, np.max(clusters) + 1)
    new_sc = induce_simplices_hg(sc, mapnodes)

    if VERBOSE:
        print(new_sc["n0"])

    return new_sc, mapnodes, clusters
