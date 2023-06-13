import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Functions import support
import scipy

plt.rcParams["text.usetex"] = True
palette = np.array(
    [
        [0.3647, 0.2824, 0.1059],
        [0.8549, 0.6314, 0.3294],
        [0.4745, 0.5843, 0.5373],
        [0.4745, 0.3843, 0.7373],
        [107.0 / 255, 42.0 / 255, 2.0 / 255],
    ]
)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_complex(sc, ax, color):
    if sc["n1"] == 0:
        G = nx.Graph()
        G.add_nodes_from(range(sc["n0"]))
        nx.draw_networkx(
            G,
            pos=nx.spring_layout(G),
            node_color=color,
            node_size=200,
            with_labels=False,
            edge_color="k",
            linewidths=2,
            ax=ax,
        )
    else:
        G = nx.Graph()
        G.add_edges_from(sc["edges"])
        pos = nx.spring_layout(G, iterations=200)
        for i in range(sc["n2"]):
            f = sc["faces"][i, :]
            x = [pos[f[0]][0], pos[f[1]][0], pos[f[2]][0]]
            y = [pos[f[0]][1], pos[f[1]][1], pos[f[2]][1]]
            ax.fill(x, y, color=color, alpha=0.2)

        nx.draw_networkx(
            G,
            pos=pos,
            node_color=color,
            node_size=10,
            with_labels=False,
            edge_color=color,
            linewidths=1,
            ax=ax,
        )
        # plt.hold(True)
        ax.axis("off")
        # ax.draw()





def plot_deg_dist(deg_dist, path = None, limsup = None):
    dim = support.list_dim(deg_dist)
    N = len(deg_dist[0][0][0][0])
    renorms = dim[2]
    d = dim[3] 
    n_tau = dim[1]
    rep = dim[0]

    Ns = np.zeros((rep, renorms, n_tau), dtype=int)
    for r in range(rep):
        for i in range(renorms):
            for t in range(n_tau):
                Ns[r, i, t] = len(deg_dist[r][t][i][0])

    names = ["Node", "Link", "Face", "Tetrahedron", "4_Simplex"]

    deg_distance = np.zeros((rep, renorms, d, n_tau))
    for r in range(rep):
        for norml in range(renorms):
            for degg in range(d):
                deg1 = deg_dist[r][0][0][degg]
                for tau in range(n_tau):
                    deg2 = deg_dist[r][tau][norml][degg]
                    if len(deg2) == 0:
                        deg2 = [0]
                    # KS Distance
                    test = scipy.stats.kstest(deg1, deg2)
                    deg_distance[r, norml, degg, tau] = test.statistic


    fig, axv = plt.subplots(1, d, figsize=(5 * d, 4))
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=5)

    for i in range(d):
        if d == 1:
            ax = axv
        else:
            ax = axv[i]
        ax.set_xlabel("Coarse graining rate")
        ax.set_ylabel("KS distance")

        for j in range(renorms):
            for r in range(rep):
                if r == 0:
                    lab = "$L_" + str(j) + "$"
                else:
                    lab = ""
                id = np.argwhere(Ns[r, j, :] > 3)
                ax.plot(
                    1 - Ns[r, j, id] / N,
                    deg_distance[r, j, i, id],
                    color=palette[j, :],
                    alpha=0.3,
                    linewidth=0.8,
                )
                ax.plot(
                    1 - Ns[r, j, id] / N,
                    deg_distance[r, j, i, id],
                    "o",
                    alpha=0.8,
                    color=palette[j, :],
                    ms=4,
                    label=lab,
                )
        ax.legend()
        ax.set_title(r"\textbf{" + names[i] + "-" + names[d] + "}", fontsize=14)
        if limsup is not None:
            ax.set_xlim(right = limsup)

    if path is not None:
        plt.savefig(path + "/deg_errors.pdf", format="pdf")  # , bbox_inches="tight")
    else:
        plt.show()
