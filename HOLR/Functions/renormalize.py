import networkx as nx
import numpy as np
from Functions import scomplex
from itertools import groupby



def compute_heat(D, exm, exM, n_t):
    # Computes the Von Neumann entropy and Entropic susceptibility
    # INPUTS
    # D: eigenvalues of the Laplacian matrix considered
    # exm, exM, n_t: computes the quantities in n_t logarithmically spaced time points
    #  in the interval [10**exm,10**exM] 

    # OUTPUTS
    # specific_heat: numpy array containing the entropic susceptibility values
    # tau_space: numpy array containing n_t - 1 logarithmically spaced time points
    # S: numpy array containing the Von Neumann entropy

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

def compute_spectral_d(D,exm,exM,n_t):
    # Computes the spectral dimension associated to a diffusion process
    # INPUTS
    # D: eigenvalues of the Laplacian matrix considered
    # exm, exM, n_t: computes the spectral dimension in n_t logarithmically spaced time points
    #  in the interval [10**exm,10**exM] 

    # OUTPUTS
    # dS: numpy array containing the spectral dimension values
    # tau_space: numpy array containing n_t - 1 logarithmically spaced time points

    tau_space = np.logspace(exm, exM, num=n_t)
    Z = np.zeros(n_t)
    for t in range(n_t):
        Z[t] = np.sum(np.exp(- tau_space[t]*D))

    dS = -2*np.diff(np.log(Z))/np.diff(np.log(tau_space))
    return dS, tau_space[1:]



def measure_SI(tau_space,sp_heat, epsilon = 0.1,ymin = -5,ymax = 1, ny = 70):
    # Computes the scale-invariance parameter of an entropic susceptibility curve
    # INPUTS
    # tau_space: numpy array containing the times in which the entropic susc. has been computed
    # sp_heat: numpy array entropic susc. curve
    # epsilon: pleateau threshold
    # ymin, ymax: respectively, the minimum and maximum value of log C to scan for plateaus
    # ny: number of points in the interval [ymin,ymax] to scan for plateaus
  
    # OUTPUTS
    # SIP: scale-invariance parameter

    max_plateau = 0
    sp_heat =  np.log(sp_heat)
    for y in np.linspace(ymin,ymax,ny):
        mask = np.abs(sp_heat-y)<epsilon
        list_s = [[a,len(list(k))] for a,k in groupby(mask)]
        for j in range(len(list_s)):
            if list_s[j][0]:
                if list_s[j][1] > max_plateau:
                    max_plateau = list_s[j][1]

    SIP = max_plateau*np.log(tau_space[1]/tau_space[0])
    return SIP

def induce_simplices(sc, mapnodes):
    # Finds induced simplices in the simplicial complex after its nodes are coarse grained 
    # INPUTS
    # sc: simplicial complex object
    # mapnodes: mapping from each node in sc to the label of its signature
  
    # OUTPUTS
    # new_sc: coarse grained simplicial complex object

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


def coarse_grain_interfaces(sc, order, comp, ncomp):
    # Coarse grains the simplicial complex accordin to a partition of the k-simplices
    # INPUTS
    # sc: simplicial complex object
    # order: order of the simplices which are partitioned
    # comp: list of labels specifying the partition of the order-simplices
    # ncomp: total number of labels

    # OUTPUTS
    # mapnodes: mapping from each node in sc to the label of its signature
    # nodesclusters: mapping from each node to its signature

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


    #for i in range(len(nodesclusters)):
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



def renormalize_steps(sc,lmax,tau, diff_order =0, int_order = 1, VERBOSE = False):
    # Performs multiple steps of the simplicial renormalization flow
    # INPUTS
    # sc: simplicial complex object
    # lmax: number of renormalization steps
    # tau: diffusion time for each step
    # diff_order: order of the diffusing simplices
    # int_order: order of the interacting simplices
    # VERBOSE: if True print the number of nodes at each step

    # OUTPUTS
    # sequence: list of the renormalized simplicial complexes

    if len(np.shape(tau)) == 0:
        tau = [tau for i in range(lmax)]


    sequence = [] 
    new_sc = sc
    for l in range(lmax):
        if l > 0 and new_sc["n0"]>1:
            L = scomplex.diffusion_laplacian(new_sc, diff_order, int_order)
            #if l > 1:
            #    df = D[1]
            D,U = np.linalg.eigh(L)
            #if l > 1:
            #    L = D*(df/D[1])
            rho  = np.abs(U@np.diag(np.exp(-tau[l]*D))@U.T)

            Gv = nx.Graph()
            Gv.add_nodes_from([i for i in range(new_sc[f"n{diff_order}"])])
            for i in range(new_sc[f"n{diff_order}"]):
                for j in range(i+1,new_sc[f"n{diff_order}"]):
                    if rho[i,j] >= min(rho[i,i],rho[j,j]):
                        Gv.add_edge(i,j)

                
            idx_components = {u:i for i,node_set in enumerate(nx.connected_components(Gv)) for u in node_set}
            clusters = [idx_components[u] for u in Gv.nodes]

            mapnodes,__ = coarse_grain_interfaces(new_sc,diff_order,clusters,np.max(clusters)+1)
            new_sc = induce_simplices(new_sc, mapnodes)

        if VERBOSE:
            print(new_sc["n0"])
        
        sequence.append(new_sc)
                
    return sequence  


def renormalize_single_step(sc,tau, diff_order =0, int_order = 1, D = None, U = None, VERBOSE = True):
    # Performs a single step of the simplicial renormalization flow
    # INPUTS
    # sc: simplicial complex object
    # tau: diffusion time
    # diff_order: order of the diffusing simplices
    # int_order: order of the interacting simplices
    # D: the list of Laplacian eigenvlaues, if none computes them from scratch
    # U: the list of Laplacian eigenvectors, if none computes them from scratch 
    # VERBOSE: if True print the number of nodes at each step

    # OUTPUTS
    # new_sc: renormalized simplicial complex
    # mapnodes: array associating to each node in sc the node in new_sc it is mapped to
    # clusters: cluster label of each simplex of order diff_order
  

    if (D is None) or (U is None):
        L = scomplex.diffusion_laplacian(sc, diff_order, int_order)
        D,U = np.linalg.eigh(L)

    rho  = np.abs(U@np.diag(np.exp(-tau*D))@U.T)

    Gv = nx.Graph()
    Gv.add_nodes_from([i for i in range(sc[f"n{diff_order}"])])
    for i in range(sc[f"n{diff_order}"]):
        for j in range(i+1,sc[f"n{diff_order}"]):
            if rho[i,j] >= min(rho[i,i],rho[j,j]):
                Gv.add_edge(i,j)

        
    idx_components = {u:i for i,node_set in enumerate(nx.connected_components(Gv)) for u in node_set}
    clusters = [idx_components[u] for u in Gv.nodes]

    mapnodes,__ = coarse_grain_interfaces(sc,diff_order,clusters,np.max(clusters)+1)
    new_sc = induce_simplices(sc, mapnodes)

    if VERBOSE:
        print(new_sc["n0"])

        
    return new_sc, mapnodes, clusters  



def renormalize_steps_rescale(sc,lmax,tau, diff_order =0, int_order = 1, VERBOSE = False):
    # Performs multiple steps of the simplicial renormalization flow
    # INPUTS
    # sc: simplicial complex object
    # lmax: number of renormalization steps
    # tau: diffusion time for each step
    # diff_order: order of the diffusing simplices
    # int_order: order of the interacting simplices
    # VERBOSE: if True print the number of nodes at each step

    # OUTPUTS
    # sequence: list of the renormalized simplicial complexes

    if len(np.shape(tau)) == 0:
        tau = [tau for i in range(lmax)]


    sequence = [] 
    new_sc = sc
    for l in range(lmax):
        if l > 0 and new_sc["n0"]>1:
            L = scomplex.diffusion_laplacian(new_sc, diff_order, int_order)
            if l > 1:
                df = D[1]
            D,U = np.linalg.eigh(L)
            if l > 1:
                L = D*(df/D[1])
            rho  = np.abs(U@np.diag(np.exp(-tau[l]*D))@U.T)

            Gv = nx.Graph()
            Gv.add_nodes_from([i for i in range(new_sc[f"n{diff_order}"])])
            for i in range(new_sc[f"n{diff_order}"]):
                for j in range(i+1,new_sc[f"n{diff_order}"]):
                    if rho[i,j] >= min(rho[i,i],rho[j,j]):
                        Gv.add_edge(i,j)

                
            idx_components = {u:i for i,node_set in enumerate(nx.connected_components(Gv)) for u in node_set}
            clusters = [idx_components[u] for u in Gv.nodes]

            mapnodes,__ = coarse_grain_interfaces(new_sc,diff_order,clusters,np.max(clusters)+1)
            new_sc = induce_simplices(new_sc, mapnodes)

        if VERBOSE:
            print(new_sc["n0"])
        
        sequence.append(new_sc)
                
    return sequence  