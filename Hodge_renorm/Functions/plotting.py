import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.matlib as mtlb

from Functions import support
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

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

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_complex(sc, ax, color, edge_alpha = 1, layout = "spring"):
    if sc["n1"] == 0:
        G = nx.Graph()
        G.add_nodes_from(range(sc["n0"]))
        nx.draw_networkx(
            G,
            pos=nx.spring_layout(G),
            node_color=color,
            alpha = edge_alpha,
            node_size=200,
            with_labels=False,
            edge_color="k",
            linewidths=1,
            ax=ax,
        )
    else:
        G = nx.Graph()
        G.add_edges_from(sc["edges"])
        if layout == "spring":
            pos = nx.spring_layout(G, iterations=200)
        elif layout == "circle":
            pos = nx.circular_layout(G)
        for i in range(sc["n2"]):
            f = sc["faces"][i, :]
            x = [pos[f[0]][0], pos[f[1]][0], pos[f[2]][0]]
            y = [pos[f[0]][1], pos[f[1]][1], pos[f[2]][1]]
            ax.fill(x, y, color=color, alpha=0.2)

        nx.draw_networkx(
            G,
            pos=pos,
            node_color=color,
            alpha = edge_alpha,
            node_size=10,
            with_labels=False,
            edge_color=color,
            linewidths=1,
            ax=ax,
        )
        # plt.hold(True)
        ax.axis("off")
        # ax.draw()


def plot_deg_dist(
    deg_dist,
    measure=None,
    path=None,
    limsup=None,
    return_data=None,
    labels=None,
    colors=None,
    lscale=False,
):
    if colors is None:
        colors = palette

    dim = support.list_dim(deg_dist)
    N = len(deg_dist[0][0][0][0])
    renorms = dim[2]
    d = dim[3]
    n_tau = dim[1]
    rep = dim[0]

    if measure is None:

        def measure(deg1, deg2):
            test = scipy.stats.kstest(deg1, deg2)
            return test.statistic

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
                    # test = scipy.stats.kstest(deg1, deg2)
                    # deg_distance[r, norml, degg, tau] = test.statistic
                    deg_distance[r, norml, degg, tau] = measure(deg1, deg2)

    fig, axv = plt.subplots(1, d, figsize=(5 * d, 4))
    plt.locator_params(axis="y", nbins=6)
    plt.locator_params(axis="x", nbins=5)

    kernel = 1 * RBF(
        length_scale=0.1, length_scale_bounds="fixed"
    )  # , length_scale_bounds=(8*1e-2, 1e2))

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
                    if labels is not None:
                        lab = labels[j]
                    else:
                        lab = "$L_" + str(j) + "$"
                else:
                    lab = ""
                id = np.argwhere(Ns[r, j, :] > 3)
                ax.plot(
                    1 - Ns[r, j, id] / N,
                    deg_distance[r, j, i, id],
                    color=colors[j, :],
                    alpha=0.0,
                    linewidth=0.8,
                )
                ax.plot(
                    1 - Ns[r, j, id] / N,
                    deg_distance[r, j, i, id],
                    "o",
                    alpha=0.3,
                    color=colors[j, :],
                    ms=1.5,
                    label=lab,
                )
            gaussian_process = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=9
            )
            gaussian_process.fit(
                np.reshape(1 - Ns[:, j, id] / N, (-1, 1)),
                np.reshape(deg_distance[:, j, i, id], (-1, 1)),
            )
            X = np.linspace(0, 1 - np.min(Ns[:, j, id]) / N, 100)
            mean_prediction, __ = gaussian_process.predict(
                np.reshape(X, (-1, 1)), return_std=True
            )
            ax.plot(X, mean_prediction, color=colors[j, :])

        ax.legend(loc="upper left")
        ax.set_title(r"\textbf{" + names[i] + "-" + names[d] + "}", fontsize=14)
        if lscale:
            ax.set_yscale("log")
        if limsup is not None:
            ax.set_xlim(right=limsup)
            ax.set_ylim(top=np.max(deg_distance[r, j, i, id]) + 0.1)

    if path is not None:
        plt.savefig(path + "/deg_errors.pdf", format="pdf")  # , bbox_inches="tight")
    else:
        plt.show()
    if return_data:
        return Ns, deg_distance, axv
    else:
        return axv
