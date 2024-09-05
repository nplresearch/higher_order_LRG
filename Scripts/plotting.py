import matplotlib.pyplot as plt
import networkx as nx


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
    """
    Plots a simplicial complex.

    Parameters
    ----------
    sc: dict
      Simplicial complex object
    ax: matplotlib axis 
      Axis where to plot
    node_color: list
      List of the node colors
    edge_color: list
      List of the edge colors
    face_color: list
      List of the face colors
    face_alpha: float
      Opacity of the faces
    edge_alpha: float
      Opacity of the edges
    edge_width: float
      Width of the edges
    layout: string
      Layout of the underlying graph. Can be "spring", "circle", "spectral" or "kamada_kawai"
    pos: numpy array of size (number of points, 2)
      Pre-computed node positions layout
    node_size: int
      Size of the nodes
    iterations: int
      Number of iterations with which the "spring" layout is computed
    """

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
        ax.axis("off")
    else:
        if len(face_color) == 1:
            face_color = [face_color[0] for i in range(sc["n2"])]

        G = nx.Graph()
        G.add_nodes_from([i for i in range(sc["n0"])])
        G.add_edges_from(sc["edges"])
        
        if pos == None:
            if layout == "spring":
                pos = nx.spring_layout(G, iterations=iterations)
            elif layout == "circle":
                pos = nx.circular_layout(G)
            elif layout == "spectral":
                pos = nx.spectral_layout(G)
            elif layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
                    
        for i in range(sc["n2"]):
            f = sc["faces"][i, :]
            pf0 = pos[f[0]]
            pf1 = pos[f[1]]
            pf2 = pos[f[2]]
            x = [pf0[0], pf1[0], pf2[0]]
            y = [pf0[1], pf1[1], pf2[1]]
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

