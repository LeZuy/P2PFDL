import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from typing import List
from aggregator import geo_median, trimmed_mean, krum, pseudo_krum, centerpoint_2d

def draw_cf_matrix(cm):
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - Node {self.id}')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix.jpg")

def flatten_weights(state_dict):
    # nối tất cả param thành 1 vector 1D
    return np.concatenate([p.cpu().numpy().ravel() for p in state_dict.values()])

# if __name__ == "__main__":
#     loss = np.loadtxt("loss_array.txt")
#     bvc_loss = loss[:,7:]

#     avg_loss = np.loadtxt("loss_array.txt")[:,7:]

#     print(np.max(bvc_loss - avg_loss))

    # file = "avg_models"
    # checkpoint = torch.load(f"{file}.pth", map_location="cpu")

    # models = {k: v for k, v in checkpoint.items() if isinstance(v, dict)}

    # vecs = {name: flatten_weights(sd) for name, sd in models.items()}
    
    # names = list(vecs.keys())
    # n = len(names)
    # sim_matrix = np.zeros((10, 10))

    # for i in range(10):
    #     for j in range(10):
    #         v1, v2 = vecs[names[i]], vecs[names[j]]
    #         sim_matrix[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(sim_matrix, xticklabels=names[:10], yticklabels=names[:10],
    #             annot=True, fmt=".2f", cmap="coolwarm")
    # plt.savefig(f"{file}_compare.jpg")

import numpy as np
import glob
import matplotlib.pyplot as plt
import os

def load_projected_models(pattern="./results/proj_weights_*.npy"):
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} projected model files.")
    return files

def analyze_projection(file_path):
    data = np.load(file_path)
    n_nodes, n_layers, proj_dim = data.shape
    print(f"{os.path.basename(file_path)}: {n_nodes} nodes, {n_layers} layers, dim={proj_dim}")
    
    # Δproj_max = max ptp toàn mạng
    diffs = np.ptp(data, axis=0)               # (n_layers, proj_dim)
    layerwise_max = np.max(diffs, axis=1)      # (n_layers,)
    global_max = np.max(diffs)
    return layerwise_max, global_max

def compare_before_after(epoch):
    before_path = f"./results/proj_weights_before_epoch_{epoch}.npy"
    after_path  = f"./results/proj_weights_after_epoch_{epoch}.npy"
    if not (os.path.exists(before_path) and os.path.exists(after_path)):
        print(f"❌ Missing files for epoch {epoch}")
        return

    before_layerwise, before_global = analyze_projection(before_path)
    after_layerwise, after_global   = analyze_projection(after_path)

    print(f"\nEpoch {epoch}:")
    print(f"Δproj_max (before) = {before_global:.6f}")
    print(f"Δproj_max (after)  = {after_global:.6f}")
    print("Layer-wise reduction:")
    for i, (b, a) in enumerate(zip(before_layerwise, after_layerwise)):
        print(f"  Layer {i:02d}: {b:.6f} → {a:.6f}  (↓{(b-a)/b*100:.2f}%)")

    # ---- Tuỳ chọn: vẽ biểu đồ ----
    plt.figure(figsize=(6,4))
    plt.plot(before_layerwise, 'r-o', label="Before consensus")
    plt.plot(after_layerwise,  'g-o', label="After consensus")
    plt.title(f"Layer-wise Δproj before vs after (epoch {epoch})")
    plt.xlabel("Layer index")
    plt.ylabel("Δproj (max difference)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("proj_comparison_epoch_{epoch}.png")
    plt.close()

def load_projection(file_path):
    data = np.load(file_path)  # shape = (num_nodes, num_layers, 2)
    n_nodes, n_layers, dim = data.shape
    assert dim == 2, f"Projection dimension must be 2, got {dim}"
    return data
def load_payloads(file_path):
    payloads = np.load(file_path, allow_pickle=True).item()
    return {int(k): v for k, v in payloads.items()}

def _payload_layer_order(payloads):
    sample = next(iter(payloads.values()), None)
    if not sample:
        raise ValueError("Payload snapshot is empty.")
    return list(sample.keys())

def plot_projection(before, after, bad_clis, layer_idx=0, title_prefix=""):
 
    n_nodes = before.shape[0]
    all_idx = np.arange(n_nodes)
    good_clis = np.setdiff1d(all_idx, bad_clis)

    fig, ax = plt.subplots(figsize=(6, 6))
    before_xy = before[:, layer_idx, :]
    after_xy = after[:, layer_idx, :]

    before_good = before_xy[good_clis]
    before_bad = before_xy[bad_clis]
    after_good = after_xy[good_clis]
    after_bad = after_xy[bad_clis]

    ax.set_xlim(-10, 5)
    ax.set_ylim(-7.5, 10)

    ax.scatter(before_good[:, 0], before_good[:, 1],c='b',label='Good (before)', alpha=0.7)
    ax.scatter(before_bad[:, 0], before_bad[:, 1],marker='x', s=80, label='Bad (before)', alpha=0.8)

    ax.scatter(after_good[:, 0], after_good[:, 1], c='r', label='Good (after)', alpha=0.7)
    ax.scatter(after_bad[:, 0], after_bad[:, 1],c='r', marker='x', s=80, label='Bad (after)', alpha=0.8)

    # --- (Tùy chọn) Vẽ đường nối good nodes ---
    # for i in good_clis:
    #     ax.plot([before_xy[i, 0], after_xy[i, 0]],
    #             [before_xy[i, 1], after_xy[i, 1]],
    #             color='gray', alpha=0.3, linewidth=1)

    ax.set_title(f"{title_prefix} Layer {layer_idx} — {n_nodes} nodes")
    ax.set_xlabel("Projection dim 1")
    ax.set_ylabel("Projection dim 2")
    ax.legend()
    ax.grid(True)
    # plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{title_prefix.replace(' ', '_').lower()}_layer_{layer_idx}.png", dpi=150)
    plt.close()

def plot_node_projection(before, after, adj_matrix, node_id, bad_clis, layer_idx=0, filename="fig.jpg"):
    before_xy = before[:, layer_idx, :]
    after_xy  = after[:, layer_idx, :]
    # bad_clis = []
    all_idx = np.arange(before.shape[0])
    good_clis = np.setdiff1d(all_idx, bad_clis)

    # neighbors = np.where(adj_matrix[node_id] != 0)[0]
    neighbors = np.where(adj_matrix[:, node_id] != 0)[0]
    good_neighbors = [n for n in neighbors if n in good_clis]
    bad_neighbors  = [n for n in neighbors if n in bad_clis]

    fig, ax = plt.subplots(figsize=(6,6))

    if good_neighbors:
        ax.scatter(before_xy[good_neighbors, 0], before_xy[good_neighbors, 1],
                   c='b', label='Good neighbors (before)', alpha=0.7)
        # ax.scatter(after_xy[good_neighbors, 0], after_xy[good_neighbors, 1],
        #            c='r', label='Good neighbors (after)', alpha=0.7)

    if bad_neighbors:
        ax.scatter(before_xy[bad_neighbors, 0], before_xy[bad_neighbors, 1],
                   c='b', marker='x', s=80, label='Bad neighbors (before)', alpha=0.8)
        # ax.scatter(after_xy[bad_neighbors, 0], after_xy[bad_neighbors, 1],
        #            c='r', marker='x', s=80, label='Bad neighbors (after)', alpha=0.8)

    v_krum = krum(np.vstack([before_xy[neighbors], before_xy[node_id]]) , f=np.ceil(len(bad_neighbors)/3).astype(int))
    v_tverberg, _ = centerpoint_2d(np.vstack([before_xy[neighbors], before_xy[node_id]]))
    ax.scatter(v_tverberg[0], v_tverberg[1],
               c='green', s=150, label='Tverberg', alpha=0.8)
    ax.scatter(v_krum[0], v_krum[1],
               c='orange', s=150, label='Krum', alpha=0.8) 

    ax.scatter(before_xy[node_id, 0], before_xy[node_id, 1],
               c='blue', marker='*', s=250, label='Self (before)')
    ax.scatter(after_xy[node_id, 0], after_xy[node_id, 1],
               c='red', marker='*', s=250, label='Self (after)')

    # Line connect center to neighbors ===
    # for n in neighbors:
    #     ax.plot([before_xy[node_id, 0], before_xy[n, 0]],
    #             [before_xy[node_id, 1], before_xy[n, 1]],
    #             color='gray', alpha=0.3, linewidth=1)

    ax.set_title(f"Node {node_id} and its neighbors — layer {layer_idx}")
    ax.legend()
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_payload(payloads, after, node_id, layer_name, filename=None):
    node_payload = payloads.get(node_id)
    if node_payload is None:
        print(f"Node {node_id} not found in payload snapshot.")
        return
    vectors = node_payload.get(layer_name)
    if vectors is None:
        print(f"Layer {layer_name} not found for node {node_id}.")
        return
    layer_order = _payload_layer_order(payloads)
    try:
        layer_idx = layer_order.index(layer_name)
    except ValueError:
        print(f"Layer {layer_name} not present in payload ordering.")
        return

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(vectors[:,0], vectors[:,1], alpha=0.7, label='Received payloads')

    v_tverberg, _ = centerpoint_2d(vectors)
    ax.scatter(v_tverberg[0], v_tverberg[1], c='green', label='centerpoint')
    if(len(vectors) < 2):
        v_krum = vectors[0]
    else:
        v_krum = krum(vectors, max(1, int(np.floor(len(vectors) / 3))))
    ax.scatter(v_krum[0], v_krum[1], c='orange', label='krum')
    ax.scatter(after[node_id, layer_idx, 0], after[node_id, layer_idx, 1],
               c='red', marker='*', label='After consensus')

    ax.set_title(f"Node {node_id} — {layer_name}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()
    plt.close()

def visualize_epoch(consensus_type, epoch=20, layer_idx=0):
    before_path = f"./results/{consensus_type}/proj_weights_before_epoch_{epoch}.npy"
    after_path  = f"./results/{consensus_type}/proj_weights_after_epoch_{epoch}.npy"
    
    if not os.path.exists(before_path) or not os.path.exists(after_path):
        print(f"❌ Missing files for epoch {epoch}")
        return
    
    if consensus_type != "optimal" and consensus_type != "tverberg_no_attack":
        bad_clis = np.loadtxt("configs/bad_clients_rand.txt", dtype=int, ndmin=1) 
    else:
        bad_clis = []

    before = load_projection(before_path)
    after  = load_projection(after_path)
    
    adj_matrix = np.loadtxt("configs/erdos_renyi.txt")

    # print(f"Loaded epoch {epoch}: shape {before.shape}")
    # plot_node_projection(before, after, adj_matrix, 0, bad_clis, 0, filename=f"Node_0_{epoch}.jpg")
    plot_projection(before, after, bad_clis, layer_idx,  title_prefix=f"Epoch {epoch}_{consensus_type}")

def visualize_payload(consensus_type, epoch, node_id, layer_name, filename=None):
    """Plot the raw payloads a node received before consensus.

    Example:
        visualize_payload("tverberg", epoch=1, node_id=0,
                          layer_name="model.conv1.weight",
                          filename="payload_node0_epoch1.png")
    """
    payload_path = f"./results/{consensus_type}/payloads_epoch_{epoch}.npy"
    after_path  = f"./results/{consensus_type}/proj_weights_after_epoch_{epoch}.npy"

    if not os.path.exists(payload_path):
        print(f"❌ Missing payload file for epoch {epoch}")
        return
    if not os.path.exists(after_path):
        print(f"❌ Missing post-consensus file for epoch {epoch}")
        return

    payloads = load_payloads(payload_path)
    after = load_projection(after_path)
    plot_payload(payloads, after, node_id, layer_name, filename=filename)

def visualize_all_payloads(consensus_type, epoch, layer_name, out_dir="./payload_plots"):
    payload_path = f"./results/{consensus_type}/payloads_epoch_{epoch}.npy"
    after_path  = f"./results/{consensus_type}/proj_weights_after_epoch_{epoch}.npy"
    if not os.path.exists(payload_path) or not os.path.exists(after_path):
        print(f"❌ Missing payload or after file for epoch {epoch}")
        return
    os.makedirs(out_dir, exist_ok=True)
    payloads = load_payloads(payload_path)
    after = load_projection(after_path)
    for node_id in sorted(payloads.keys()):
        filename = os.path.join(out_dir, f"payload_node_{node_id}_epoch_{epoch}.png")
        plot_payload(payloads, after, node_id, layer_name, filename=filename)

def load_test_accuracies(result_dir):

    pattern = os.path.join(result_dir, "test_results_epoch_*.txt")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    accs = []

    for f in files:
        data = np.loadtxt(f)
        # lấy accuracy ở cột 1 (giữa), trung bình qua tất cả dòng
        acc = np.mean(data[:, 1])
        accs.append(acc)

    return np.arange(len(accs)), np.array(accs)

def count_bad_neighbors(adj_matrix: np.ndarray, bad_clients: List[int]) -> np.ndarray:
    n = adj_matrix.shape[0]
    bad_mask = np.zeros(n, dtype=bool)
    bad_mask[bad_clients] = True

    bad_counts = np.sum(adj_matrix[:, bad_mask] != 0, axis=1)
    return bad_counts

# ----- Main -----
if __name__ == "__main__":
    # # ==== đường dẫn tới kết quả ====
    # methods = {
    #     "krum": "results2/krum",
    #     "tverberg": "results/tverberg"
    # }

    # plt.figure(figsize=(8, 5))

    # for name, path in methods.items():
    #     if not os.path.exists(path):
    #         print(f"⚠️ Không tìm thấy thư mục {path}")
    #         continue
    #     epochs, accs = load_test_accuracies(path)
    #     epochs = np.arange(33)
    #     accs = accs[0:33]
    #     print(epochs)
    #     plt.plot(epochs*5, accs, label=name, linewidth=2)

    # plt.xlabel("Epoch")
    # plt.ylabel("Test Accuracy")
    # plt.title("Test Accuracy")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("fig.jpg")
    # img_dir = "./"  # thư mục chứa ảnh
    # aggregators = ["tverberg", "krum"]
    # epochs = [1, 41, 81,  121]
    # layer = 0  # nếu cần thay đổi layer

    # # # ==== 2️⃣ Tạo Figure ====
    # fig, axes = plt.subplots(len(epochs), len(aggregators), figsize=(12, 12))
    # fig.subplots_adjust(wspace=0.05, hspace=0.1)

    # for i, epoch in enumerate(epochs):
    #     for j, agg in enumerate(aggregators):
    #         filename = f"epoch_{epoch}_{agg}_layer_{layer}.png"
    #         path = os.path.join(img_dir, filename)
    #         ax = axes[i, j]

    #         if os.path.exists(path):
    #             img = Image.open(path)
    #             ax.imshow(img)
    #             ax.set_title(f"{agg}_epoch_{epoch}", fontsize=10)
    #         else:
    #             ax.text(0.5, 0.5, "Missing", ha='center', va='center', fontsize=12, color='red')

    #         ax.axis("off")
    #         if j == 0:
    #             ax.set_ylabel(f"Epoch {epoch}", fontsize=12)

    # plt.tight_layout()
    # plt.savefig("fig2.jpg")

    epochs = [1, 41, 81,  121]
    for i in range(10):
        visualize_epoch("mean",  epoch=i)
        # visualize_payload("tverberg", 
        #                 epoch=i,
        #                 node_id=0,
        #                 layer_name="model.conv1.weight",
        #                 filename=f"payload_node0_epoch{i}.png" )

    # adj = np.loadtxt("./configs/chord.txt")
    # bad_clients = np.loadtxt("./configs/bad_clients.txt").astype(np.int16)
    # bad_counts = count_bad_neighbors(adj, bad_clients)
    # for i, c in enumerate(bad_counts):
    #     print(f"Node {i}: {c} bad neighbors")
