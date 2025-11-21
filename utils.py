import torch
import numpy as np
from scipy.spatial import Delaunay
from sklearn.random_projection import GaussianRandomProjection


def safe_normalize(v, eps=1e-12):
    """Normalize vector v to unit L2; if near-zero, return a random unit vector."""
    norm = np.linalg.norm(v)
    if norm <= eps:
        # fallback: random unit vector with same shape
        rand = np.random.randn(*v.shape)
        rnorm = np.linalg.norm(rand)
        return rand / (rnorm + eps)
    return v / norm


def flatten_weights(weights_dict):
    return np.concatenate([w[0] for w in weights_dict.values()])


def model_params(model):
    """
    Trả về dict chỉ chứa các tham số có thể học được.
    Dạng: {layer_name: [1-D np.array, torch.Size]}.
    """
    params = {}
    for name, tensor in model.named_parameters():  # chỉ lấy tham số cần học
        if tensor.requires_grad:
            params[name] = [
                tensor.detach().cpu().numpy().flatten(),
                tuple(tensor.shape)
            ]
    return params


def restore_state_dict(model, weights):
    state_dict = model.state_dict()
    for name, (values, shape) in weights.items():
        if name in state_dict:
            tensor = torch.tensor(values).view(shape)
            state_dict[name].copy_(tensor)
    return state_dict


def get_projection_matrix(original_dim, lower_dim):
    projection = GaussianRandomProjection(n_components=lower_dim)
    dummy_data = np.random.rand(1, original_dim)
    projection.fit(dummy_data)
    return projection.components_


def random_project(x_d, Pi):
    return Pi @ x_d


def build_projection_mats(model, k_dim):
    """
    Build random projection matrices for each layer in model.
    model : TinyCNN (or any torch.nn.Module)
    k_dim : target lower dimension (int)
    rp : module random_projection
    """
    W = model_params(model)
    mats = {
        layer: get_projection_matrix(flat.size, k_dim)
        for layer, (flat, _) in W.items()
    }
    total_dim = sum(flat.size for flat, _ in W.values())
    mats["__flat__"] = get_projection_matrix(total_dim, k_dim)
    return mats


def save_projection_mats(projection_mats, path):
    """
    Persist projection matrices to disk for later reuse.
    projection_mats : dict of layer_name -> np.ndarray
    path : destination .npz file or Path
    """
    names = np.array(list(projection_mats.keys()))
    matrices = np.empty(len(names), dtype=object)
    for idx, name in enumerate(names):
        matrices[idx] = np.asarray(projection_mats[name])
    np.savez_compressed(str(path), names=names, mats=matrices)


def load_projection_mats(path):
    """
    Load projection matrices saved with save_projection_mats.
    """
    with np.load(str(path), allow_pickle=True) as archive:
        names = archive["names"]
        mats = archive["mats"]
    if names.shape[0] != mats.shape[0]:
        raise ValueError(
            "Invalid projection matrix archive: length mismatch between names and mats."
        )
    return {
        str(name): np.asarray(mat)
        for name, mat in zip(names.tolist(), mats.tolist())
    }


def is_point_in_convex_hull(point, hull_points):
    """
    Checks if a 2D point is inside the convex hull formed by a set of points.

    Args:
        point (np.array): The 2D point to check, e.g., np.array([x, y]).
        hull_points (np.array): An array of 2D points defining the convex hull,
                                 e.g., np.array([[x1, y1], [x2, y2], ...]).

    Returns:
        bool: True if the point is inside or on the boundary of the convex hull,
              False otherwise.
    """
    hull_points = np.asarray(hull_points)
    if len(hull_points) <= hull_points.shape[1]:
        # không đủ điểm để tạo convex hull (d+1)
        return True  # coi như nằm trong hull
    try:
        hull = Delaunay(hull_points)
        return hull.find_simplex(point) >= 0
    except Exception:
        return True


def save_model(procs, path="./results/models.pth"):
    torch.save({f"Model_{p.id}": p.model.state_dict() for p in procs}, path)


def save_projected_model(procs, projected_dims, path):
    """
    Lưu mảng (num_nodes, num_layers, projected_dims) sang .npy
    - Hỗ trợ cả torch.Tensor (CUDA) lẫn np.ndarray.
    - Sẽ assert nếu chiều chiếu thực tế != projected_dims.
    """
    num_nodes = len(procs)
    # dùng thứ tự layer từ node 0 để cố định cột
    layer_items = list(procs[0].proj_weights.items())
    num_layers = len(layer_items)

    projected_weights = np.zeros(
        (num_nodes, num_layers, projected_dims), dtype=np.float32
    )

    for i, p in enumerate(procs):
        # đảm bảo thứ tự layer như node 0
        for j, (k0, _) in enumerate(layer_items):

            flat_weights, _shape = p.proj_weights[k0]

            if isinstance(flat_weights, torch.Tensor):
                vec = flat_weights.detach().to("cpu").float().numpy()
            else:
                vec = np.asarray(flat_weights, dtype=np.float32)

            if vec.ndim != 1:
                vec = vec.reshape(-1)

            if vec.shape[0] != projected_dims:
                Pi_dict = getattr(p, "Pi", None)
                if Pi_dict is None or k0 not in Pi_dict:
                    raise ValueError(
                        f"[save_projected_model] Missing projection matrix for layer '{k0}' on node {i}."
                    )

                # Re-project to the desired dimension when we were given an unprojected vector.
                vec = random_project(vec, Pi_dict[k0])
                vec = np.asarray(vec, dtype=np.float32).reshape(-1)

                if vec.shape[0] != projected_dims:
                    raise ValueError(
                        f"[save_projected_model] Layer '{k0}' của node {i} có dim={vec.shape[0]} "
                        f"không khớp projected_dims={projected_dims}. "
                        f"Hãy kiểm tra Pi[k] hoặc dùng bản lưu .npz per-layer ở dưới."
                    )

            projected_weights[i, j, :] = vec

    np.save(path, projected_weights)


def save_consensus_payloads(procs, path, *, project_before_saving: bool = False):
    """
    Save the exact buffers each node aggregated during consensus.
    Stored format: dict[node_id][layer_name] = np.ndarray (deg+1, dim).
    """
    payloads = {}
    for p in procs:
        buffers = getattr(p, "B_proj", None)
        if not buffers:
            continue
        export_buffers = (
            [p.project(buffer) for buffer in buffers]
            if project_before_saving
            else buffers
        )
        if not export_buffers:
            continue
        layer_payloads = {}
        for layer in export_buffers[0]:
            vectors = np.stack(
                [
                    p._to_numpy(buffer[layer][0])
                    for buffer in export_buffers
                ],
                axis=0,
            )
            layer_payloads[layer] = vectors
        payloads[p.id] = layer_payloads
    np.save(path, payloads, allow_pickle=True)
