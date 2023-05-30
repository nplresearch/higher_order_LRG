import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_complex(sc, ax):
    if sc["n1"] == 0:
        G = nx.Graph()
        G.add_nodes_from(range(sc["n0"]))
        nx.draw_networkx(
            G,
            pos=nx.spring_layout(G),
            node_color="k",
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
            ax.fill(x, y, color=[0.8, 0.8, 0.8])

        nx.draw_networkx(
            G,
            pos=pos,
            node_color="k",
            node_size=10,
            with_labels=False,
            edge_color="k",
            linewidths=1,
            ax=ax,
        )
        # plt.hold(True)
        ax.axis("off")
        # ax.draw()
