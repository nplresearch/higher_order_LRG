import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix


def make_dict(sc):
    edge_dict = {}
    for i, edge in enumerate(sc["edges"]):
        # Create a tuple to represent the edge (order doesn't matter)
        edge_key = tuple(sorted(edge))
        # Store the edge index in the dictionary
        edge_dict[edge_key] = i

    face_dict = {}
    for i, face in enumerate(sc["faces"]):
        # Create a tuple to represent the edge (order doesn't matter)
        face_key = tuple(sorted(face))
        # Store the edge index in the dictionary
        face_dict[face_key] = i

    return edge_dict, face_dict


def boundary_matrices_3(sc):
    n0 = sc["n0"]
    n1 = sc["n1"]
    n2 = sc["n2"]
    n3 = sc["n3"]

    B1 = lil_matrix((n0, n1), dtype=np.int8)
    B2 = lil_matrix((n1, n2), dtype=np.int8)
    B3 = lil_matrix((n2, n3), dtype=np.int8)

    edge_dict, face_dict = make_dict(sc)

    for e in range(n1):
        nodes = sc["edges"][e, :]
        B1[nodes[0], e] = -1
        B1[nodes[1], e] = 1

    for face_idx, face in enumerate(sc["faces"]):
        # Iterate over the three edges of the face
        for i in range(3):
            # Determine the start and end nodes of the edge
            edge = face[np.arange(3) != i]

            # Find the corresponding edge index in the edges list
            edge_idx = edge_dict[tuple(sorted(edge))]
            # Set the appropriate entry in `boundary_2`
            B2[edge_idx, face_idx] = (-1) ** i

    for tetra_idx, tetrahedron in enumerate(sc["tetrahedra"]):
        # Iterate over the four faces of the tetrahedron
        for i in range(4):
            # Determine the three nodes of the face
            face = tetrahedron[np.arange(4) != i]

            # Find the corresponding face index in the faces list
            face_idx = face_dict[tuple(sorted(face))]

            # Set the appropriate entry in `boundary_3`
            B3[face_idx, tetra_idx] = (-1) ** i

    B1 = B1.tocsc()
    B2 = B2.tocsc()
    B3 = B3.tocsc()

    return B1, B2, B3, edge_dict, face_dict


def generalized_degree(sc, edge_dict, face_dict, d):
    if d == 1:
        deg = [np.zeros(sc["n0"])]
        for i in range(sc["n1"]):
            edge = sc["edges"][i]
            deg[0][edge[0]] += 1
            deg[0][edge[1]] += 1
    elif d == 2:
        deg = [np.zeros(sc["n0"]), np.zeros(sc["n1"])]
        for i in range(sc["n2"]):
            face = sc["faces"][i]
            deg[0][face[0]] += 1
            deg[0][face[1]] += 1
            deg[0][face[2]] += 1
            deg[1][edge_dict[tuple((face[0], face[1]))]] += 1
            deg[1][edge_dict[tuple((face[0], face[2]))]] += 1
            deg[1][edge_dict[tuple((face[1], face[2]))]] += 1
    elif d == 3:
        deg = [np.zeros(sc["n0"]), np.zeros(sc["n1"]), np.zeros(sc["n2"])]
        for i in range(sc["n3"]):
            tet = sc["tetrahedra"][i]
            deg[0][tet[0]] += 1
            deg[0][tet[1]] += 1
            deg[0][tet[2]] += 1
            deg[0][tet[3]] += 1
            deg[1][edge_dict[tuple((face[0], face[1]))]] += 1
            deg[1][edge_dict[tuple((face[0], face[2]))]] += 1
            deg[1][edge_dict[tuple((face[0], face[3]))]] += 1
            deg[1][edge_dict[tuple((face[1], face[2]))]] += 1
            deg[1][edge_dict[tuple((face[1], face[3]))]] += 1
            deg[1][edge_dict[tuple((face[2], face[3]))]] += 1

            deg[2][face_dict[tuple((face[0], face[1], face[2]))]] += 1
            deg[2][face_dict[tuple((face[0], face[1], face[3]))]] += 1
            deg[2][face_dict[tuple((face[0], face[2], face[3]))]] += 1
            deg[2][face_dict[tuple((face[1], face[2], face[3]))]] += 1

    return deg


def NGF_d1(N, s, beta):
    """
    Generate NGF in dimension d=2 and flavour s=-1,0,1.

    Args:
    - N: maximal number of nodes in the NGF
    - s: flavour of the NGF (-1, 0, or 1)
    - beta: inverse temperature (beta > 0 or beta = 0)

    Returns:
    - a: adjacency matrix
    - kn: vector of generalized degrees k_{1,0} (the degree) of the nodes
    """

    # Initialization
    a = sp.lil_matrix((N, N))
    a_occ = np.zeros(N)

    # Assign energies to the nodes
    epsilon = np.random.rand(N) ** (1 / 2)

    # Initial condition: at time t=1 a single link (1,2)
    a[0, 1] = np.exp(-beta * (epsilon[0] + epsilon[1]))
    a[1, 0] = a[0, 1]
    a_occ[0] = 1
    a_occ[1] = 1

    # Addition of new links at time t=in-1 the node in is added to the network geometry with flavour
    for in_ in range(2, N):
        # Choose the node to which attach a new link
        V = np.exp(-beta * epsilon) * a_occ
        norm = np.sum(V)
        x = np.random.rand() * norm
        if norm > 0:
            for nj1 in range(in_):
                x -= V[nj1]
                if x < 0:
                    j = nj1
                    break

        # Attach the new link between node in and node j
        a[in_, j] = np.exp(-beta * (epsilon[in_] + epsilon[j]))
        a[j, in_] = a[in_, j]
        a_occ[in_] = 1
        a_occ[j] += s

    # Generalized degree (degree) of the nodes
    a = a > 0
    G = nx.from_numpy_matrix(a)
    sc = {
        "nodes": np.arange(0, N),
        "edges": np.array(list(G.edges())),
        "faces": np.array([]),
        "tetrahedra": np.array([]),
    }
    sc["n0"] = N
    sc["n1"] = sc["edges"].shape[0]
    sc["n2"] = 0
    sc["n3"] = 0
    sc["edges"] = np.unique(np.sort(sc["edges"], axis=1), axis=0)

    return sc


def NGF_d2(N, s, beta):
    # If you use this code, please cite the following paper:
    # G. Bianconi and C. Rahmede
    # "Network geometry with flavour: from complexity to quantum geometry"
    # Physical Review E 93, 032315 (2016).

    # Initialization
    a = csr_matrix((N, N))
    a_occ = csr_matrix((N, N))
    a_occ2 = csr_matrix((N, N))
    sc = {
        "nodes": np.arange(0, N),
        "edges": np.array([]),
        "faces": np.array([]),
        "tetrahedra": np.array([]),
    }

    # Assign energies to the nodes
    kappa = 1
    epsilon = np.random.rand(N) ** (1 / (kappa + 1))

    # Initial condition at time t=1 including a single triangle between nodes 1, 2, 3
    L = 0
    for i1 in range(1, 4):
        for i2 in range(i1 + 1, 4):
            L += 1
            a[i1 - 1, i2 - 1] = np.exp(-beta * (epsilon[i1 - 1] + epsilon[i2 - 1]))
            a[i2 - 1, i1 - 1] = np.exp(-beta * (epsilon[i1 - 1] + epsilon[i2 - 1]))
            a_occ[i1 - 1, i2 - 1] = 1
            a_occ[i2 - 1, i1 - 1] = 1
            a_occ2[i1 - 1, i2 - 1] = 1
            a_occ2[i2 - 1, i1 - 1] = 1

    nt = 1
    sc["faces"] = np.array([[0, 1, 2]])

    r = np.zeros((N, 2))
    d_N = np.zeros(N)
    D = np.ones(N)
    theta = np.zeros(N)

    for i in range(3):
        r[i, :] = [np.cos(2 * np.pi * i / 3), np.sin(2 * np.pi * i / 3)]
        d_N[i] = 1
        theta[i] = i / 3

    # At each time t=in-2 we attach a new triangle
    for in_ in range(3 + 1, N + 1):
        # Choose edge (l1,l2) to which we will attach the new triangle
        mat = csr_matrix(a.multiply(a_occ))
        I, J = mat.nonzero()
        V = np.squeeze(mat[mat.nonzero()])
        norm = np.sum(V)
        x = np.random.rand() * norm
        if norm > 0:
            for nj1 in range(np.shape(V)[1]):
                x -= V[0, nj1]
                if x < 0:
                    nj = nj1
                    break
            l1 = I[nj]
            l2 = J[nj]

            d_N[in_ - 1] = min(d_N[l1], d_N[l2]) + 1
            D[in_ - 1] = max(d_N)
            r[in_ - 1, :] = r[l1, :] + r[l2, :]
            r[in_ - 1, :] /= np.sqrt(r[in_ - 1, 0] ** 2 + r[in_ - 1, 1] ** 2)
            theta[in_ - 1] = min(theta[l1], theta[l2]) + 0.5 * abs(
                theta[l1] - theta[l2]
            )
            if theta[l1] == 0 and theta[l2] >= 2 / 3:
                theta[in_ - 1] = theta[l2] + 0.5 * (1 - theta[l2])
            if theta[l2] == 0 and theta[l1] >= 2 / 3:
                theta[in_ - 1] = theta[l1] + 0.5 * (1 - theta[l1])

            a_occ[l1, l2] += s
            a_occ[l2, l1] += s
            a_occ2[l1, l2] += 1
            a_occ2[l2, l1] += 1

            nt += 1
            sc["faces"] = np.concatenate((sc["faces"], np.array([[in_ - 1, l1, l2]])))

            # Attach the new node in to the node l1
            L += 1
            a[in_ - 1, l1] = np.exp(-beta * (epsilon[l1] + epsilon[in_ - 1]))
            a[l1, in_ - 1] = np.exp(-beta * (epsilon[l1] + epsilon[in_ - 1]))
            a_occ[in_ - 1, l1] = 1
            a_occ[l1, in_ - 1] = 1
            a_occ2[in_ - 1, l1] = 1
            a_occ2[l1, in_ - 1] = 1

            # Attach the new node in to the node l2
            L += 1
            a[in_ - 1, l2] = np.exp(-beta * (epsilon[l2] + epsilon[in_ - 1]))
            a[l2, in_ - 1] = np.exp(-beta * (epsilon[l2] + epsilon[in_ - 1]))
            a_occ[in_ - 1, l2] = 1
            a_occ[l2, in_ - 1] = 1
            a_occ2[in_ - 1, l2] = 1
            a_occ2[l2, in_ - 1] = 1

    I, J = csr_matrix(a).nonzero()
    sc["edges"] = np.column_stack((I, J))

    sc["edges"] = np.unique(np.sort(sc["edges"], axis=1), axis=0)
    sc["faces"] = np.unique(np.sort(sc["faces"], axis=1), axis=0)
    sc["n0"] = N
    sc["n1"] = sc["edges"].shape[0]
    sc["n2"] = sc["faces"].shape[0]
    sc["n3"] = 0

    return sc


def NGF_d3(N, s, beta):
    # If you use this code, please cite the following paper:
    # G. Bianconi and C. Rahmede
    # "Network geometry with flavour: from complexity to quantum geometry"
    # Physical Review E 93, 032315 (2016).

    # Initialization
    a = csr_matrix((N, N))
    at = np.zeros((0))
    a_occ = np.zeros((0))
    a_occ3 = np.zeros((0))
    sc = {
        "nodes": np.arange(0, N),
        "edges": np.array([]),
        "faces": np.zeros((0, 3)),
        "tetrahedra": np.array([]),
    }

    # Assign energies to the nodes
    kappa = 1
    epsilon = np.random.rand(N) ** (1 / (kappa + 1))

    r = np.zeros((N, 3))
    d_N = np.ones(N)
    D = np.ones(N)

    r[0, :] = [1, 1, 1] / np.sqrt(3)
    r[1, :] = [-1, -1, 1] / np.sqrt(3)
    r[2, :] = [-1, 1, -1] / np.sqrt(3)
    r[3, :] = [1, -1, -1] / np.sqrt(3)

    # Initial condition: a single tetrahedron (0, 1, 2, 3)
    sc["tetrahedra"] = np.array([[0, 1, 2, 3]])

    for i1 in range(4):
        for i2 in range((i1 + 1), 4):
            a[i1, i2] = 1
            a[i2, i1] = 1
            for i3 in range((i2 + 1), 4):
                sc["faces"] = np.concatenate((sc["faces"], np.array([[i1, i2, i3]])))
                at = np.concatenate(
                    (at, [np.exp(-beta * (epsilon[i1] + epsilon[i2] + epsilon[i3]))])
                )
                a_occ = np.concatenate((a_occ, [1]))
                a_occ3 = np.concatenate((a_occ3, [1]))

    # At each time t=in-3 we attach a new tetrahedron
    for in_ in range(4, N):
        # Choose triangular face to which to attach the new tetrahedron
        mat = sp.csr_matrix(at * a_occ)
        I, J = mat.nonzero()
        V = np.squeeze(mat[mat.nonzero()])
        norm = np.sum(V)
        x = np.random.rand() * norm
        for nj1 in range(np.shape(V)[1]):
            x = x - V[0, nj1]
            if x < 0:
                nj = J[nj1]
                break

        l = [int(sc["faces"][nj, 0]), int(sc["faces"][nj, 1]), int(sc["faces"][nj, 2])]

        d_N[in_] = min([d_N[l[0]], d_N[l[1]], d_N[l[2]]]) + 1
        D[in_] = max(d_N)
        for sas in range(3):
            r[in_, sas] = r[in_, sas] / np.sqrt(np.sum(r[in_, :] ** 2))
        a_occ[nj] = a_occ[nj] + s
        a_occ3[nj] = a_occ3[nj] + 1

        # Add the tetrahedron
        for n in range(3):
            a[in_, l[n]] = 1
            a[l[n], in_] = 1

        for n1 in range(3):
            for n2 in range(n1 + 1, 3):
                a[l[n1], l[n2]] += 1
                a[l[n2], l[n1]] += 1

        sc["tetrahedra"] = np.concatenate(
            (sc["tetrahedra"], np.array([[in_, l[0], l[1], l[2]]]))
        )

        for n in range(3):
            for n2 in range(n + 1, 3):
                sc["faces"] = np.concatenate(
                    (sc["faces"], np.array([[l[n], l[n2], in_]]))
                )
                at = np.concatenate(
                    (
                        at,
                        [
                            np.exp(
                                -beta * (epsilon[l[n]] + epsilon[l[n2]] + epsilon[in_])
                            )
                        ],
                    )
                )
                a_occ = np.concatenate((a_occ, [1]))
                a_occ3 = np.concatenate((a_occ, [1]))

    I, J = csr_matrix(a).nonzero()
    sc["edges"] = np.column_stack((I, J))

    sc["edges"] = np.unique(np.sort(sc["edges"], axis=1), axis=0)
    sc["faces"] = np.unique(np.sort(sc["faces"], axis=1), axis=0)
    sc["tetrahedra"] = np.unique(np.sort(sc["tetrahedra"], axis=1), axis=0)
    sc["n0"] = N
    sc["n1"] = sc["edges"].shape[0]
    sc["n2"] = sc["faces"].shape[0]
    sc["n3"] = sc["tetrahedra"].shape[0]

    return sc
