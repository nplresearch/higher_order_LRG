import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.matlib as mtlb

import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


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


def plot_complex(
    sc,
    ax,
    node_color=["black"],
    edge_color=["black"],
    face_color=["black"],
    face_alpha=0.2,
    edge_alpha=1,
    edge_width=1,
    layout="spring",
    pos = None,
    node_size=10,
    iterations=1000,
):
    if sc["n1"] == 0:
        G = nx.Graph()
        G.add_nodes_from(range(sc["n0"]))
        nx.draw_networkx(
            G,
            pos=nx.spring_layout(G),
            node_color=node_color,
            alpha=edge_alpha,
            node_size=200,
            with_labels=False,
            edge_color="k",
            linewidths=1,
            ax=ax,
        )
    else:
        if len(face_color) == 1:
            face_color = [face_color[0] for i in range(sc["n2"])]

        G = nx.Graph()
        G.add_edges_from(sc["edges"])
        if pos == None:
            if layout == "spring":
                pos = nx.spring_layout(G, iterations=iterations)
            elif layout == "circle":
                pos = nx.circular_layout(G)
            elif layout == "spectral":
                pos = nx.spectral_layout(G)
        for i in range(sc["n2"]):
            f = sc["faces"][i, :]
            x = [pos[f[0]][0], pos[f[1]][0], pos[f[2]][0]]
            y = [pos[f[0]][1], pos[f[1]][1], pos[f[2]][1]]
            ax.fill(x, y, color=face_color[i], alpha=face_alpha)

        nx.draw_networkx(
            G,
            pos=pos,
            node_color=node_color,
            alpha=edge_alpha,
            node_size=node_size,
            with_labels=False,
            width=edge_width,
            edge_color=edge_color,
            linewidths=1,
            ax=ax,
        )
        # plt.hold(True)
        ax.axis("off")
        # ax.draw()

