def NGF(d, N, s, beta):
    """
    Generate NGF in dimension d and flavour s=-1,0,1.

    Args:
    - N: maximal number of nodes in the NGF
    - s: flavour of the NGF (-1, 0, or 1)
    - beta: inverse temperature (beta > 0 or beta = 0)

    Returns:
    - a: adjacency matrix
    - kn: vector of generalized degrees k_{1,0} (the degree) of the nodes
    """
    if d == 1:
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
            "edges": np.array(list(G.edges()), dtype=int),
            "faces": np.array([], dtype=int),
            "tetrahedra": np.array([], dtype=int),
        }
        sc["n0"] = N
        sc["n1"] = sc["edges"].shape[0]
        sc["n2"] = 0
        sc["n3"] = 0
        sc["edges"] = np.unique(np.sort(sc["edges"], axis=1), axis=0)

        return sc
    elif d == 2:
        a = csr_matrix((N, N))
        a_occ = csr_matrix((N, N))
        a_occ2 = csr_matrix((N, N))
        sc = {
            "nodes": np.arange(0, N),
            "edges": np.array([], dtype=int),
            "faces": np.array([], dtype=int),
            "tetrahedra": np.array([], dtype=int),
        }

        # Assign energies to the nodes
        kappa = 1
        epsilon = np.random.rand(N) ** (1 / (kappa + 1))

        # Initial condition at time t=1 including a single triangle between nodes 1, 2, 3
        L = 0
        for i1 in range(3):
            for i2 in range(i1 + 1, 3):
                L += 1
                a[i1, i2] = np.exp(-beta * (epsilon[i1] + epsilon[i2]))
                a[i2, i1] = np.exp(-beta * (epsilon[i1] + epsilon[i2]))
                a_occ[i1, i2] = 1
                a_occ[i2, i1] = 1
                a_occ2[i1, i2] = 1
                a_occ2[i2, i1] = 1

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
        for in_ in range(3, N):
            # Choose edge (l1,l2) to which we will attach the new triangle

            mat = tril(a.multiply(a_occ), format="csr")
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

                d_N[in_] = min(d_N[l1], d_N[l2]) + 1
                D[in_] = max(d_N)
                r[in_, :] = r[l1, :] + r[l2, :]
                r[in_, :] /= np.sqrt(r[in_, 0] ** 2 + r[in_, 1] ** 2)
                theta[in_] = min(theta[l1], theta[l2]) + 0.5 * abs(
                    theta[l1] - theta[l2]
                )
                if theta[l1] == 0 and theta[l2] >= 2 / 3:
                    theta[in_] = theta[l2] + 0.5 * (1 - theta[l2])
                if theta[l2] == 0 and theta[l1] >= 2 / 3:
                    theta[in_] = theta[l1] + 0.5 * (1 - theta[l1])

                a_occ[l1, l2] += s
                a_occ[l2, l1] += s
                a_occ2[l1, l2] += 1
                a_occ2[l2, l1] += 1

                nt += 1
                sc["faces"] = np.concatenate((sc["faces"], np.array([[in_, l1, l2]])))

                # Attach the new node in to the node l1
                L += 1
                a[in_, l1] = np.exp(-beta * (epsilon[l1] + epsilon[in_]))
                a[l1, in_] = np.exp(-beta * (epsilon[l1] + epsilon[in_]))
                a_occ[in_, l1] = 1
                a_occ[l1, in_] = 1
                a_occ2[in_, l1] = 1
                a_occ2[l1, in_] = 1

                # Attach the new node in to the node l2
                L += 1
                a[in_, l2] = np.exp(-beta * (epsilon[l2] + epsilon[in_]))
                a[l2, in_] = np.exp(-beta * (epsilon[l2] + epsilon[in_]))
                a_occ[in_, l2] = 1
                a_occ[l2, in_] = 1
                a_occ2[in_, l2] = 1
                a_occ2[l2, in_] = 1

        I, J = tril(a, format="csr").nonzero()
        sc["edges"] = np.column_stack((I, J))

        sc["edges"] = np.unique(np.sort(sc["edges"], axis=1), axis=0)
        sc["faces"] = np.unique(np.sort(sc["faces"], axis=1), axis=0)
        sc["n0"] = N
        sc["n1"] = sc["edges"].shape[0]
        sc["n2"] = sc["faces"].shape[0]
        sc["n3"] = 0

        return sc

    elif d == 3:
        # Initialization
        a = csr_matrix((N, N))
        at = np.zeros((0))
        a_occ = np.zeros((0))
        a_occ3 = np.zeros((0))
        sc = {
            "nodes": np.arange(0, N),
            "edges": np.array([], dtype=int),
            "faces": np.zeros((0, 3), dtype=int),
            "tetrahedra": np.array([], dtype=int),
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
                    sc["faces"] = np.concatenate(
                        (sc["faces"], np.array([[i1, i2, i3]]))
                    )
                    at = np.concatenate(
                        (
                            at,
                            [np.exp(-beta * (epsilon[i1] + epsilon[i2] + epsilon[i3]))],
                        )
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

            l = [
                int(sc["faces"][nj, 0]),
                int(sc["faces"][nj, 1]),
                int(sc["faces"][nj, 2]),
            ]

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
                                    -beta
                                    * (epsilon[l[n]] + epsilon[l[n2]] + epsilon[in_])
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

    else:
        print("Dimension out of bounds. NGF implemented only for d=1,2,3")
