import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph_from_adj(A: np.ndarray,
                        directed=False,
                        weighted=False,
                        threshold=0.0,
                        labels=None,
                        layout="circular",
                        with_edge_labels=None,
                        bad_nodes=None,
                        seed=42):
    """
    Vẽ đồ thị từ ma trận kề numpy.
    bad_nodes: list hoặc set các chỉ số node cần tô đỏ.
    """
    A = np.asarray(A)
    n = A.shape[0]
    assert A.ndim == 2 and n == A.shape[1], "A phải là ma trận vuông"

    # Giữ cạnh có trọng số > threshold
    B = A.copy()
    B[B <= threshold] = 0.0

    # Tạo đồ thị
    G = nx.from_numpy_array(B, create_using=nx.DiGraph if directed else nx.Graph)

    # Label
    if labels is None:
        labels = {i: str(i) for i in range(n)}
    else:
        labels = {i: str(lbl) for i, lbl in enumerate(labels)}

    # Weight mặc định
    if not weighted:
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=seed)

    # Xác định màu node
    bad_nodes = set(bad_nodes or [])
    node_colors = ["red" if i in bad_nodes else "blue" for i in range(n)]

    # Vẽ node và cạnh
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, arrows=directed, alpha=0.7)
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    if with_edge_labels or weighted:
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2g}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("topology.jpg")

if __name__ == "__main__":
    a = np.loadtxt("./configs/erdos_renyi.txt")
    bad_clis = np.loadtxt("configs/bad_clients.txt").tolist()
    print(a.shape)
    plot_graph_from_adj(A = a, bad_nodes = bad_clis)
