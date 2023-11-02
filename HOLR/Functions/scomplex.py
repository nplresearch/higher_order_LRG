import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix, spdiags
from itertools import combinations
import xgi
#import graph_tool.all as gt

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


def boundary_matrices(sc):
    # Compute bounadry matrices up to the 4-th order
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


def NGF(d, N, s, beta, M = 1):
    # Generate a d-dimensional NGF simplicial complex
    # 
    # 
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

    it = d  # + 1

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
                    nj = J[nj1] # Index of the d-1 simplex where the next simplex is attached
                    break

            a_occ[nj] = a_occ[nj] + s

            # Attach the next simplex
            for n1 in range(d): 
                j = node[nj, n1]
                a[it, j] = 1
                a[j, it] = 1

        for n1 in range(d): # d (d-1)-simplices are added
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


def convert_graph_to_sc(G, dim = 2, type = 'clique'):
    # Converts a graph G to its clique complex of dimension dim 
    # type: 'clique', 'inference'
    G = nx.convert_node_labels_to_integers(G)
    N = len(G.nodes)
    sc = {
        "nodes": np.reshape(np.array(G.nodes), (-1, 1)),
        "n0": N,
        "edges": np.sort(np.array(G.edges), 1),
    }
    all_cliques = list(nx.enumerate_all_cliques(nx.from_edgelist(sc["edges"])))
    sc["n1"] = len(sc["edges"])
    if type == 'clique':
        cliques = all_cliques
    # elif type == 'inference':
    #     g = gt.Graph(directed=False)
    #     g.add_edge_list(G.edges())
    #     state = gt.CliqueState(g)
    #     state.mcmc_sweep(niter=10000)
    #     cliques = []
    #     for v in state.f.vertices():      
    #         if state.is_fac[v]:
    #             continue          
    #         if state.x[v] > 0:
    #             cliques.append(list(state.c[v]))
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


def adjacency_of_order(sc,k,l, sparse = False):
    # sc: simplicial complex object
    # k: order of the diffusing simplices
    # l: order of the interaction simplices

    keys = ["nodes", "edges", "faces", "tetrahedra", "4-simplices"]
    nk = sc[f"n{k}"]
    if sparse:
        adj = lil_matrix((nk,nk), dtype = int)
    else:
        adj = np.zeros((nk,nk),dtype = int)
    
    assert l != k, "The interaction order should be different from the order of the diffusing simplices"
    assert l >= 0, "The interaction order should be greater or equal than 0"
    assert k >= 0, "The order of the diffusing simplices should be greater or equal than 0"
    assert (l <= 4) and (k <= 4), "Simplices of order greater than 4 are not supported"


    if l < k: 
        diff_units = sc[keys[k]]
        for i in range(nk):
            for j in range(i+1,nk):
                intersection = (set(diff_units[i,:]) & set(diff_units[j,:]))
                if len(intersection) == l + 1:
                    adj[i,j] += 1

    elif l > k:
        edge_dict, face_dict, tet_dict = make_dict(sc)
        dicts = [{(i,):i for i in range(sc["n0"])},edge_dict,face_dict,tet_dict]
        int_simplices = sc[keys[l]]

        for i in range(sc[f"n{l}"]):
            simp = int_simplices[i,:]
            combs = list(combinations(simp, k+1))
            ncombs = len(combs)
            combs_ids = np.zeros(ncombs,dtype=int)
            for n in range(ncombs):
                combs_ids[n] = dicts[k][combs[n]]
            combs_ids = np.sort(combs_ids)
            for n in range(ncombs):
                for m in range(n+1,ncombs):
                    adj[combs_ids[n],combs_ids[m]] += 1

    if sparse:
        adj = csr_matrix(adj)

    return adj + adj.T

def diffusion_laplacian(sc,k,l, sparse = False):
    A = adjacency_of_order(sc,k,l,sparse)
    K = np.sum(A, 0)
    if sparse:
        lenK = K.shape[1]
        L = spdiags(K,0,lenK,lenK) - A
    else:
        L = np.diag(K) - A
    return L


def pseudofractal_d2(steps):
    edges = [(0,1),(1,2),(0,2)]
    n = 3
    for s in range(steps):
        boundary = edges.copy()
        for ed in boundary:
            edges.append((ed[0],n))
            edges.append((ed[1],n))
            n += 1

    G = nx.from_edgelist(edges)
    sc = convert_graph_to_sc(G,dim = 2)
    return sc

def pseudofractal_d3(steps):
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    faces = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    n = 4
    for s in range(steps):
        boundary = faces.copy()
        for fa in boundary:
            edges.append((fa[0],n))
            edges.append((fa[1],n))
            edges.append((fa[2],n))
            faces.append((fa[0],fa[1],n))
            faces.append((fa[0],fa[2],n))
            faces.append((fa[1],fa[2],n))
            n += 1

    G = nx.from_edgelist(edges)
    sc = convert_graph_to_sc(G,dim = 3)

    return sc

def apollonian_d2(steps):
    edges = [(0,1),(1,2),(0,2)]
    new_boundary = edges.copy()
    n = 3
    for s in range(steps):
        boundary = new_boundary
        new_boundary = []
        for ed in boundary:
            edges.append((ed[0],n))
            edges.append((ed[1],n))
            new_boundary.append((ed[0],n))
            new_boundary.append((ed[1],n))
            n += 1

    G = nx.from_edgelist(edges)
    sc = convert_graph_to_sc(G, dim = 2)

    return sc


def import_network_data(f, d):
    # Imports a network from file and returns its clique complex 
    # INPUTS
    # f: file
    # d: maximal dimension of the cliques
    # OUTPUTS
    # sc: clique complex
    i = 0
    edges = []
    for line in f:
        if i != 0:
            words = line.split()
            edges.append((words[0],words[1]))
        else:
            i += 1
    f.close()
    G = nx.from_edgelist(edges)
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    sc = convert_graph_to_sc(G,dim = d)

    return sc

def import_network_data_xgi(name,d):
    # Imports a higher-order network from xgi and reuturns the associated 
    # simplicial complex 
    # INPUTS
    # name: name of the dataset 
    # {"email-enron","email-eu","hospital-lyon”,"contact-high-school”,
    # “contact-primary-school”,“tags-ask-ubuntu”,“congress-bills”,
    # “disgenenet”,“diseasome”,"ndc-substances”,“coauth-mag-geology”,
    # “coauth-mag-history”}
    # d: maximal dimension of the simplices
    # OUTPUTS
    # sc: simplicial complex

    H = xgi.load_xgi_data(name, max_order=d)
    H.cleanup()
    H = xgi.SimplicialComplex(H.edges.members())
    H.close()

    sc = {}
    sc["nodes"] = np.sort(np.array([H.nodes]).T,0)
    sc["n0"] = sc["nodes"].shape[0]
    keys = ["edges","faces","tetrahedra","4-simplices"]

    for k in keys:
        sc[k] = []
    for e in H.edges.members():
        sc[keys[len(e)-2]].append(list(e))

    for i,k in enumerate(keys):
        if len(sc[k]) == 0:
            sc[k] = np.zeros((0,i+2))
        else:
            sc[k] = np.unique(np.sort(np.array(sc[k]),1),axis =0)
        
        sc[f"n{i+1}"] = sc[k].shape[0] 
    
    return sc  

def subdivide(G):
    # Returns the barycentric subdivision of a given graph, i.e. the 
    # original graph where each edge is replaced with two edges  
    # INPUTS
    # G: networkx graph
    # OUTPUTS
    # G: subdivided graph
    
    n = len(G.nodes) 
    G2 = G.copy()
    for e in G2.edges:
        e0 = e[0]
        e1 = e[1]
        G.remove_edge(e0,e1)
        G.add_edge(e0,n)
        G.add_edge(n,e1)
        n += 1
        
    return G