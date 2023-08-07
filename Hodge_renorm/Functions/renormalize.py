from collections import defaultdict

import networkx as nx
import numpy as np
from Functions import support, scomplex
from scipy.sparse import csgraph, csr_matrix
import scipy
import matplotlib.pyplot as plt


def renormalize_simplicial_VARIANTS(
    sc,
    order=0,
    L=None,
    U=None,
    D=None,
    tau=1,
    METHOD="representative",
    SPARSIFY=False,
    TRUE_CONNECTIONS=False,
    threshold=1,
    simple=False,
    rho=None,
    nc=10,
):
    # Perform a simplicial renormalization step
    # Inputs:
    # sc - simplicial complex object
    # order - {0,1,2,3} order of the renormalization
    # U - Laplacian of given order's eigenvectors
    # D - array containing Laplacian of given order's eigenvalues
    # tau - diffusion time
    # METHOD - {'closest','representative'}
    # SPARSIFY - {True,False}
    # TRUE_CONNECTIONS - {True,False}

    nk = sc["n" + str(order)]

    if rho is None:
        assert len(D) == nk

        D = np.abs(D)  # Ensure eigenvalues are non-negative

        # STEP I: Cluster the simplices

        # Compute the eigenvalues of exp(-tau*L)/trace(exp(-tau*L))
        Dtilde = np.zeros(nk)
        for i in range(nk):
            # Dtilde[i] = 1 / np.sum(np.exp(-tau * (D - D[i])))
            Dtilde[i] = np.exp(-tau * D[i])

        rho = U @ np.diag(Dtilde) @ U.T  # Normalized Heat Kernel
        rho = np.abs(np.triu(rho))  # Take its absolute value to ignore orientations
        # rho = np.abs(np.real(rho))
        ncut = np.sum(D > 1 / np.abs(tau))  # Number of simplices to remove

    else:
        assert len(rho) == nk
        ncut = nc

    if not simple:
        ncomp, comp = cluster_simplices(
            nk, ncut, rho, threshold, TRUE_CONNECTIONS, L, SPARSIFY
        )
    else:
        zeta = np.zeros((nk, nk))
        for i in range(nk):
            for j in range(i, nk):
                zeta[i, j] = (rho[i, j] >= rho[i, i]) or (rho[i, j] >= rho[j, j])
        zeta = csr_matrix(zeta)
        ncomp, comp = csgraph.connected_components(
            zeta, directed=True, connection="weak"
        )  # Clusters assigned to the simplices

    # STEP II: Perform the reduction
    mapnodes, nodesclusters = coarse_grain(sc, order, comp, ncomp, METHOD)

    new_sc = induce_simplices(sc, mapnodes)

    return new_sc, mapnodes, comp, nodesclusters


def cluster_simplices(
    nk, ncut, rho, threshold=1, TRUE_CONNECTIONS=False, L=None, SPARSIFY=False
):
    # Aggregate simplices by sorting the values of rho until ncut are removed
    if ncut >= threshold * nk:
        comp = np.ones(nk, dtype=int)
        ncomp = nk
    else:
        if TRUE_CONNECTIONS:  # Only adjacent simplices can be clustered together
            rho = rho * (np.abs(L) > 10**-7)

        idx = np.argsort(rho.ravel())[::-1]
        sh = rho.shape
        zeta = list(zip(*np.unravel_index(idx[:ncut], sh, order="C")))

        uf = support.UnionFind(nk)
        uf.add_edges(zeta)
        k = ncut + 1
        nc = uf.connected_components()
        while nc > nk - ncut:
            new_zeta = list(
                zip(*np.unravel_index(idx[k : k + nc - (nk - ncut)], sh, order="C"))
            )
            zeta = zeta + new_zeta
            uf.add_edges(new_zeta)
            k = k + nc - (nk - ncut)
            nc = uf.connected_components()

        ones = np.ones(len(zeta), np.uint32)
        zeta = tuple(zip(*zeta))
        if len(zeta) != 0:
            zeta = csr_matrix((ones, (zeta[0], zeta[1])), shape=(nk, nk))
        else:
            zeta = csr_matrix((nk, nk))
        # Sparsify
        if SPARSIFY:
            ii, jj = zeta.nonzero()
            for k in range(len(ii)):
                i = ii[k]
                j = jj[k]
                zeta[i, j] = zeta[i, j] * (rho[i, j] >= np.min([rho[i, i], rho[j, j]]))

        ncomp, comp = csgraph.connected_components(
            zeta, directed=True, connection="weak"
        )  # Clusters assigned to the simplices

    return ncomp, comp


def coarse_grain(sc, order, comp, ncomp, METHOD="representative"):
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

    # Build supernodes
    clusternodesinterface = [
        [] for _ in range(ncomp)
    ]  # clusternodesinterface: cluster -> interface nodes belonging to it
    clusternodes = [[] for _ in range(ncomp)]  # cluster -> nodes belonging to it
    for i in range(ncomp):
        for j in range(sc["n0"]):
            if i in nodesclusters[j]:  # If node j belongs to cluster i
                clusternodes[i].append(j)
                if (
                    len(nodesclusters[j]) > 1
                ):  # If node j belongs to more than one cluster then it is an interface
                    clusternodesinterface[i].append(j)

    # Assign signature nodes i.e. cluster representatives
    cluster_representatives = np.zeros(ncomp, dtype=int)
    for i in range(ncomp):
        notinterface = np.setdiff1d(
            clusternodes[i], clusternodesinterface[i]
        )  # Nodes in cluster i which are not interfaces
        if len(notinterface) == 0:
            cluster_representatives[i] = -1  # No representative
        else:
            cluster_representatives[i] = notinterface[0]

    mapnodes = np.zeros(
        sc["n0"], dtype=int
    )  # Maps each node to its image in the renormalized simplicial complex

    j = 0
    for i in range(sc["n0"]):
        if len(nodesclusters[i]) != 1:  # Interface nodes are left unchanged
            mapnodes[i] = j
            j += 1
        elif (
            i == cluster_representatives[list(nodesclusters[i])[0]]
        ):  # Representative nodes are left unchanged
            mapnodes[i] = j
            j += 1

    if METHOD == "representative":
        for i in range(sc["n0"]):
            if len(nodesclusters[i]) == 1:  # The node is not an interface
                mapnodes[i] = mapnodes[
                    cluster_representatives[list(nodesclusters[i])[0]]
                ]  # Collapse node i to the representative
    elif METHOD == "closest":
        G = nx.Graph()
        G.add_edges_from(
            [(e[0], e[1]) for e in sc["edges"]]
        )  # using a list of edge tuples
        for i in range(sc["n0"]):
            if len(nodesclusters[i]) == 1:  # The node is not an interface
                if (
                    cluster_representatives[list(nodesclusters[i])[0]] == -1
                ):  # If there is no representative in the cluster of node i
                    setint = clusternodesinterface[
                        nodesclusters[i]
                    ]  # Set of interfaces in the cluster of node i
                else:
                    setint = clusternodesinterface[list(nodesclusters[i])[0]] + [
                        cluster_representatives[list(nodesclusters[i])[0]]
                    ]
                dist = [
                    nx.algorithms.shortest_path_length(G, source=i, target=x)
                    for x in setint
                ]
                id = np.argmin(dist)
                mapnodes[i] = mapnodes[
                    setint[id]
                ]  # Collapse node i to the closest interface or representative
    else:
        raise ValueError("METHOD must be 'closest' or 'representative'")
    return mapnodes, nodesclusters


def induce_simplices(sc, mapnodes):
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

    # Connect supernodes with faces
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


def compute_heat(D, exm, exM, n_t):
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
    specific_heat = -(np.diff(S) / np.diff(np.log(tau_space)))
    # specific_heat = -(np.diff(S) / np.diff(tau_space))
    tau_space = tau_space[: n_t - 1]
    return specific_heat, tau_space, S


def renormalize_simplicial_Dirac(
    sc,
    orders,
    Us,
    Ds,
    taus,
):
    # Perform a simplicial renormalization step
    # Inputs:
    # sc - simplicial complex object
    # order - {0,1,2,3} order of the renormalization
    # U - Laplacian of given order's eigenvectors
    # D - array containing Laplacian of given order's eigenvalues
    # tau - diffusion time
    # METHOD - {'closest','representative'}
    # SPARSIFY - {True,False}
    # TRUE_CONNECTIONS - {True,False}

    Ps = []
    for o in range(len(orders)):
        order = orders[o]
        D = np.abs(Ds[o])  # Ensure eigenvalues are non-negative
        U = Us[o]
        tau = taus[o]

        if order == 0:
            nk = sc["n0"]
            assert len(D) == nk
        elif order == 1:
            nk = sc["n1"]
            assert len(D) == nk
        elif order == 2:
            nk = sc["n2"]
            assert len(D) == nk
        elif order == 3:
            nk = sc["n3"]
            assert len(D) == nk
        elif order == 4:
            nk = sc["n4"]
            assert len(D) == nk
        else:
            raise ValueError("Order must be 0, 1, 2, 3 or 4")

        # STEP I: Cluster the simplices

        # Compute the eigenvalues of exp(-tau*L)/trace(exp(-tau*L))
        Dtilde = np.zeros(nk)
        for i in range(nk):
            Dtilde[i] = 1 / np.sum(np.exp(-tau * (D - D[i])))

        rho = U @ np.diag(Dtilde) @ U.T  # Normalized Heat Kernel
        rho = np.abs(np.triu(rho))  # Take its absolute value to ignore orientations

        ncut = np.sum(D > 1 / tau)  # Number of simplices to remove

        ncomp, comp = cluster_simplices(nk, ncut, rho, 1, False, 0, False)

        mapnodes, _ = coarse_grain(sc, order, comp, ncomp)
        nc = len(np.unique(mapnodes))
        Ps.append(support.map2partition(mapnodes, nc))

    P = support.meet(Ps[0], Ps[1])

    mapnodes, nc = support.partition2map(P, sc["n0"])
    new_sc = induce_simplices(sc, mapnodes)

    return new_sc, mapnodes


def plot_heats(d, sc, tmin=-1, tmax=3, nt=200, plot=True, L1_=None):
    B1, B2, __, __, edge_dict, face_dict, tet_dict = scomplex.boundary_matrices_3(sc)
    L0 = (B1 @ B1.T).todense()
    if L1_ is None:
        L1 = (B1.T @ B1 + B2 @ B2.T).todense()
    else:
        L1 = L1_
    L2 = (B2.T @ B2).todense()

    D0 = scipy.linalg.eigh(L0)[0]
    specific_heat0, tau_space, S0 = compute_heat(D0, tmin, tmax, nt)
    fpeaks = scipy.signal.find_peaks(specific_heat0)[0]

    D1 = scipy.linalg.eigh(L1)[0]
    specific_heat1, tau_space, S1 = compute_heat(D1, tmin, tmax, nt)

    fpeaks1 = scipy.signal.find_peaks(specific_heat1)[0]

    if d >= 2:
        D2 = scipy.linalg.eigh(L2)[0]
        specific_heat2, tau_space, S2 = compute_heat(D2, tmin, tmax, nt)
        fpeaks2 = scipy.signal.find_peaks(specific_heat2)[0]
    else:
        D2 = []
        fpeaks2 = []

    if plot:
        f, ax = plt.subplots(1, 1, figsize=(6.5, 4))

        ax.semilogx(tau_space, specific_heat0)
        ax.semilogx(tau_space, specific_heat1)

        if d >= 2:
            ax.semilogx(tau_space, specific_heat2)

        for p in fpeaks:
            ax.vlines(
                x=tau_space[p],
                ymin=specific_heat0[p] - 0.1,
                ymax=specific_heat0[p] + 0.1,
                color="tab:blue",
            )
        for p in fpeaks1:
            ax.vlines(
                x=tau_space[p],
                ymin=specific_heat1[p] - 0.1,
                ymax=specific_heat1[p] + 0.1,
                color="tab:orange",
            )
        if d >= 2:
            for p in fpeaks2:
                ax.vlines(
                    x=tau_space[p],
                    ymin=specific_heat2[p] - 0.1,
                    ymax=specific_heat2[p] + 0.1,
                    color="tab:green",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("$C_k$")

        ax2 = ax.twinx()

        ax2.semilogx(tau_space, 1 - S0[1:] / np.log(sc["n0"]), "--")
        ax2.semilogx(tau_space, 1 - S1[1:] / np.log(sc["n1"]), "--")
        if d >= 2:
            ax2.semilogx(tau_space, 1 - S2[1:] / np.log(sc["n2"]), "--")

        ax.legend([f"$L_{i}$" for i in range(d + 1)])

    return D0, D1, D2, tau_space, fpeaks, fpeaks1, fpeaks2


def compute_psi(Z, tol=10):
    thresholds = Z[:, 2]  # np.unique(Z[:,2].round(decimals = tol))
    psi = np.zeros(len(thresholds) - 1)
    for i in range(len(thresholds) - 1):
        psi[i] = np.log10(thresholds[i + 1]) - np.log10(thresholds[i])

    psi = psi / (np.log10(thresholds[-1]) - np.log10(thresholds[0]))

    return psi, thresholds


def compute_psi_nlog(Z, tol=10):
    thresholds = np.unique(Z[:, 2].round(decimals=tol))
    psi = np.zeros(len(thresholds) - 1)
    for i in range(len(thresholds) - 1):
        psi[i] = (thresholds[i + 1]) - (thresholds[i])

    psi = psi / ((thresholds[-1]) - (thresholds[0]))

    return psi, thresholds


def coarse_grain_interfaces(sc, order, comp, ncomp, METHOD="representative"):
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

    for i in range(len(nodesclusters)):
        nodesclusters[i] = np.array2string(np.sort(list(nodesclusters[i])))
    uq = np.unique(nodesclusters)
    d = {b: a for a, b in enumerate(uq)}

    mapnodes = np.zeros(
        sc["n0"], dtype=int
    )  # Maps each node to its image in the renormalized simplicial complex
    for i in range(sc["n0"]):
        mapnodes[i] = d[nodesclusters[i]]

    return mapnodes, nodesclusters
