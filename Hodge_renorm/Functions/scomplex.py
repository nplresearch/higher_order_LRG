import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix, tril


def make_dict(sc):
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


def boundary_matrices_3(sc):
    n0 = sc["n0"]
    n1 = sc["n1"]
    n2 = sc["n2"]
    n3 = sc["n3"]
    n4 = sc["n4"]

    B1 = lil_matrix((n0, n1), dtype=np.int8)
    B2 = lil_matrix((n1, n2), dtype=np.int8)
    B3 = lil_matrix((n2, n3), dtype=np.int8)
    B4 = lil_matrix((n3, n4), dtype=np.int8)

    edge_dict, face_dict, tet_dict = make_dict(sc)

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

    for four_simp_idx, four_simplex in enumerate(sc["4-simplices"]):
        # Iterate over the four tetrahedra of the 4-simplex
        for i in range(5):
            # Determine the four nodes of the tetrahedron
            tetrahedron = four_simplex[np.arange(5) != i]

            # Find the corresponding tetrahedron index in the tetrahedra list
            tet_idx = tet_dict[tuple(sorted(tetrahedron))]

            # Set the appropriate entry in `boundary_3`
            B4[tet_idx, four_simp_idx] = (-1) ** i

    B1 = B1.tocsc()
    B2 = B2.tocsc()
    B3 = B3.tocsc()
    B4 = B4.tocsc()

    return B1, B2, B3, B4, edge_dict, face_dict, tet_dict


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
            deg[1][edge_dict[tuple((tet[0], tet[1]))]] += 1
            deg[1][edge_dict[tuple((tet[0], tet[2]))]] += 1
            deg[1][edge_dict[tuple((tet[0], tet[3]))]] += 1
            deg[1][edge_dict[tuple((tet[1], tet[2]))]] += 1
            deg[1][edge_dict[tuple((tet[1], tet[3]))]] += 1
            deg[1][edge_dict[tuple((tet[2], tet[3]))]] += 1

            deg[2][face_dict[tuple((tet[0], tet[1], tet[2]))]] += 1
            deg[2][face_dict[tuple((tet[0], tet[1], tet[3]))]] += 1
            deg[2][face_dict[tuple((tet[0], tet[2], tet[3]))]] += 1
            deg[2][face_dict[tuple((tet[1], tet[2], tet[3]))]] += 1

    return deg


def NGF(d, N, s, beta):
    if d > 4:
        print("Dimension out of bounds. NGF implemented only for d=1,2,3,4")

    kappa = 1
    epsilon = np.random.rand(N) ** (1 / (kappa + 1))
    a = sp.lil_matrix((N, N))
    at = np.array([])
    a_occ = np.array([])
    node = np.zeros(((d + 1) + (N - (d + 1)) * d, d), dtype=int)

    # Initial condition: at time t=1 a single d-dimensional hypercube (1,2,3,4)
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

    it = d  # + 1

    while it < N - 1:
        it += 1

        mat = at * a_occ
        J = mat.nonzero()[0]
        V = np.squeeze(mat[mat.nonzero()])
        norm = np.sum(V)
        x = np.random.rand() * norm
        for nj1 in range(len(V)):
            x -= V[nj1]
            if x < 0:
                nj = J[nj1]
                break

        a_occ[nj] = a_occ[nj] + s

        for n1 in range(d):
            j = node[nj, n1]
            a[it, j] = 1
            a[j, it] = 1

        for n1 in range(d):
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
    G = nx.from_numpy_matrix(a)
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
