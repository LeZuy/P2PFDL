import numpy as np
import networkx as nx
import os 
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from utils import build_projection_mats
from model.ResNet_Cifar import ResNet18_CIFAR

def chord_adjacency_matrix(n: int, directed=False):
    """
    Sinh ma trận kề cho đồ thị có cạnh (i, i + 2^j mod n)
    như trong mạng Chord / hypercube ring.

    Args:
        n (int): số node
        directed (bool): True nếu muốn đồ thị có hướng

    Returns:
        A (np.ndarray): ma trận kề kích thước n x n
    """
    A = np.zeros((n, n), dtype=int)
    max_pow = int(np.floor(np.log2(n)))

    for i in range(n):
        for j in range(max_pow):
            neighbor = (i + 2**j) % n
            A[i, neighbor] = 1
            if not directed:
                A[neighbor, i] = 1  # làm cho đồ thị vô hướng

    return A


def ring_lattice_matrix(n: int, degree:int, directed=False):
    A = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(degree):
            neighbor = (i + j ) % n
            A[i, neighbor] = 1
            if not directed:
                A[neighbor, i] = 1  # làm cho đồ thị vô hướng

    return A

def erdos_renyi_matrix(n: int, k: int, directed=False, seed: int = None):
    """
    Sinh ma trận kề cho đồ thị Erdős-Rényi ngẫu nhiên.

    Args:
        n (int): số node
        k (int): bậc của đồ thị
        directed (bool): True nếu muốn đồ thị có hướng
        seed (int): hạt giống cho bộ sinh số ngẫu nhiên
    Returns:
        A (np.ndarray): ma trận kề kích thước n x n
    """
    if seed is not None:
        np.random.seed(seed)

    m = int(n*k/2)  # 20 edges
    seed = 20160  # seed random number generators for reproducibility
    A = np.zeros((n, n), dtype=int)
    while np.any(np.sum(A, axis=1) < 2):
    # Use seed for reproducibility
        G = nx.gnm_random_graph(n, m, seed=seed)
        A = nx.to_numpy_array(G, dtype=int)
        seed += 1
    if not directed:
        A = np.maximum(A, A.T)  # làm cho đồ thị vô hướng       
    
    return A

def load_projection(file_path):
    """Đọc file .npy đã lưu bởi save_projected_model"""
    data = np.load(file_path)  # shape = (num_nodes, num_layers, 2)
    n_nodes, n_layers, dim = data.shape
    assert dim == 2, f"Projection dimension must be 2, got {dim}"
    return data

def plot_projection(before, after, layer_idx=0, title_prefix=""):
    """
    Vẽ tọa độ (2D) của tất cả node trong một layer trước & sau consensus.
    - before: np.ndarray (n_nodes, n_layers, 2)
    - after:  np.ndarray (n_nodes, n_layers, 2)
    """
    n_nodes = before.shape[0]
    fig, ax = plt.subplots(figsize=(6,6))

    before_xy = before[:, layer_idx, :]
    after_xy = after[:, layer_idx, :]

    # Đọc danh sách Byzantine nodes
    bad_clients = np.loadtxt("./configs/bad_clients.txt", dtype=int).tolist()
    # bad_clients = []
    if isinstance(bad_clients, int):  # trường hợp chỉ có 1 node
        bad_clients = [bad_clients]
    good_clients = [i for i in range(n_nodes) if i not in bad_clients]

    # --- Plot good clients ---
    ax.scatter(before_xy[good_clients, 0], before_xy[good_clients, 1],
               c='red', label='Good (before)', alpha=0.6)
    ax.scatter(after_xy[good_clients, 0], after_xy[good_clients, 1],
               c='blue', label='Good (after)', alpha=0.6)

    # --- Plot Byzantine (bad) clients ---
    ax.scatter(before_xy[bad_clients, 0], before_xy[bad_clients, 1],
               c='red', marker='x', label='Byzantine')
    
    # # --- Nối các điểm (đường di chuyển của mỗi node) ---
    # for i in range(n_nodes):
    #     ax.plot([before_xy[i,0], after_xy[i,0]],
    #             [before_xy[i,1], after_xy[i,1]],
    #             color='gray', alpha=0.3, linewidth=1)

    ax.set_title(f"{title_prefix} {n_nodes} nodes")
    ax.set_xlabel("Projection dim 1")
    ax.set_ylabel("Projection dim 2")
    ax.legend()
    ax.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{title_prefix.replace(' ', '_').lower()}_layer_{layer_idx}.png")

def visualize_epoch(epoch=20, layer_idx=0):
    """Tải 2 file before/after của một epoch và plot"""
    before_path = f"./results/tverberg/proj_weights_before_epoch_{epoch}.npy"
    after_path  = f"./results/tverberg/proj_weights_after_epoch_{epoch}.npy"
    
    if not os.path.exists(before_path) or not os.path.exists(after_path):
        print(f"❌ Missing files for epoch {epoch}")
        return
    
    before = load_projection(before_path)
    after  = load_projection(after_path)

    print(f"Loaded epoch {epoch}: shape {before.shape}")
    plot_projection(before, after, layer_idx, title_prefix=f"Epoch {epoch}")

def plot_vectors(node_id, layer):

    vectors = np.loadtxt(f"./debugs/Node_{node_id}_vectors_{layer}.txt")
    v_bar = np.loadtxt(f"./debugs/Node_{node_id}_consensus_{layer}.txt")

    plt.figure()
    plt.scatter(vectors[:, 0], vectors[:, 1], c='blue', label='Vectors from neighbors', alpha = 0.7)
    plt.scatter(v_bar[0], v_bar[1], color='red', label='Consensus Point', alpha = 0.7)
    plt.title(f'Node {node_id} Layer {layer} Vectors and Consensus')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid()
    plt.savefig(f'Node_{node_id}_layer_{layer}_vectors.png')


def count_images_in_split(root_dir):
    """
    Đếm số ảnh của mỗi class trong các split con.
    root_dir: thư mục chứa train/ hoặc test/
    """
    summary = defaultdict(Counter)

    # Duyệt qua các split (split_0, split_1, ...)
    for split_name in sorted(os.listdir(root_dir)):
        split_path = os.path.join(root_dir, split_name)
        if not os.path.isdir(split_path):
            continue

        # Duyệt qua từng class folder (0..9)
        for cls_name in sorted(os.listdir(split_path)):
            class_dir = os.path.join(split_path, cls_name)
            if not os.path.isdir(class_dir):
                continue
            n_images = len([
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            summary[split_name][cls_name] += n_images

    return summary


def print_summary(summary, title=""):
    print(f"\n===== {title} =====")
    all_classes = sorted({cls for counts in summary.values() for cls in counts})
    header = "split_id: " + "".join(all_classes) + " total"
    print(header)

    for split, counts in summary.items():
        total = sum(counts.values())
        row = [split] + [str(counts.get(cls, 0)) for cls in all_classes] + [str(total)]
        print(" ".join(row))

def visualize__tverberg_point(vectors):
    # vectors = np.loadtxt(path)
    plt.figure()
    plt.scatter(vectors[:-1, 0], vectors[:-1, 1], c='blue', label='Vectors from neighbors', alpha = 0.7)
    plt.scatter(vectors[-1, 0], vectors[-1, 1], color='red', label='Consensus Point', alpha = 0.7)
    plt.legend()
    plt.grid()
    plt.savefig(f'./fig.png')

# if __name__ == "__main__":
    # Ví dụ: vẽ layer 0 tại epoch 20
    # print_summary(
    #     count_images_in_split("./data_splits/train"),
    #     title="Train set class distribution"
    # )
    # print_summary(
    #     count_images_in_split("./data_splits/test"),
    #     title="Test set class distribution"
    # )
    # for epoch in range(5):
    #     visualize_epoch(epoch=epoch, layer_idx=0)
    # visualize__tverberg_point("./debugs/node29_layer_fc2.bias_failed.txt")
    # base_model = ResNet18_CIFAR()
    
    # projection_mats = build_projection_mats(base_model, 2)
    # np.savetxt("./configs/proj_mat.txt", projection_mats)

def choose_bad_nodes_even(N=64, B=8):
    step = N // B
    bad_nodes = np.array([(k * step) % N for k in range(B)], dtype=int)
    return np.sort(bad_nodes)


def count_bad_neighbors(adj, bad_nodes):
    bad_mask = np.zeros(adj.shape[0], dtype=bool)
    bad_mask[bad_nodes] = True
    return np.sum(adj[:, bad_mask] == 1, axis=1)


# ======================
# RUN DEMO
# ======================
if __name__ == "__main__":
    N = 64
    B = 7    # ví dụ muốn 8 bad nodes

    adj = np.loadtxt("./configs/chord.txt")
    bad_nodes = choose_bad_nodes_even(N, B)
    # bad_nodes = np.loadtxt("./bad_clis.txt").astype(np.int16)
    bad_counts = count_bad_neighbors(adj, bad_nodes)
    np.savetxt("bad_clis.txt",bad_nodes, fmt="%d")
    print("Bad nodes:", bad_nodes)
    print("Bad neighbor counts:")
    for i in range(N):
        print(f"Node {i:02d}: {bad_counts[i]}")

    # kiểm tra điều kiện < 1/3 (degree = 6 -> <2)
    valid = np.all(bad_counts < 3)
    print("\nConstraint satisfied? ->", valid)

