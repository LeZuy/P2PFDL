import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

def group_by_class(dataset):
    class_dict = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_dict[label].append(idx)
    return class_dict

def save_split(dataset, class_indices, out_dir, num_splits=64):
    os.makedirs(out_dir, exist_ok=True)
    num_classes = len(class_indices)
    
    # tạo folder con 0..63
    for split_id in range(num_splits):
        split_dir = os.path.join(out_dir, f"split_{split_id}")
        os.makedirs(split_dir, exist_ok=True)
    
    for cls, indices in class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices) 
        splits = np.array_split(indices, num_splits)  
        
        for split_id, idxs in enumerate(splits):
            split_dir = os.path.join(out_dir, f"split_{split_id}")
            class_dir = os.path.join(split_dir, str(cls))
            os.makedirs(class_dir, exist_ok=True)
            
            for i in idxs:
                img, label = dataset[i]
                img = torchvision.transforms.ToPILImage()(img)
                img.save(os.path.join(class_dir, f"{i}.png"))

def _download_data(dataset_name="emnist") -> Tuple[Dataset, Dataset]:
    """Download the requested dataset. Currently supports cifar10, mnist, and fmnist.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training dataset, the test dataset.
    """
    trainset, testset = None, None
    if dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4),
                        mode="reflect",
                    ).data.squeeze()
                ),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "fmnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    else:
        raise NotImplementedError

    return trainset, testset

def partition_data_dirichlet(
    num_clients, alpha, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    # get the targets
    tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_clients)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset

def save_dirichlet_datasets(clients, testset, root="./data_splits"):
    """
    Lưu dữ liệu Dirichlet (list các Subset) thành ảnh thật, theo cấu trúc thư mục client/class.

    Parameters
    ----------
    clients : List[Subset]
        Danh sách các tập con (mỗi client 1 subset).
    testset : Dataset
        Tập kiểm thử (test set).
    root : str
        Thư mục gốc để lưu dữ liệu.
    """

    os.makedirs(root, exist_ok=True)

    # --- Lưu trainset của từng client ---
    for cid, subset in enumerate(clients):
        print(f"Saving data for client {cid} ...")
        client_dir = os.path.join(root, f"client_{cid}")
        os.makedirs(client_dir, exist_ok=True)

        for idx in tqdm(range(len(subset))):
            img, label = subset[idx]
            class_dir = os.path.join(client_dir, f"class_{label}")
            os.makedirs(class_dir, exist_ok=True)

            img_path = os.path.join(class_dir, f"img_{idx:05d}.png")

            # Nếu ảnh là tensor, convert sang 3-channel
            if isinstance(img, torch.Tensor):
                if img.ndim == 2:  # grayscale
                    img = img.unsqueeze(0)
                save_image(img, img_path)
            else:
                # Trường hợp PIL image
                img.save(img_path)

    # --- Lưu test set ---
    print("Saving test set ...")
    test_dir = os.path.join(root, "test")
    os.makedirs(test_dir, exist_ok=True)

    for idx in tqdm(range(len(testset))):
        img, label = testset[idx]
        class_dir = os.path.join(test_dir, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"img_{idx:05d}.png")

        if isinstance(img, torch.Tensor):
            if img.ndim == 2:
                img = img.unsqueeze(0)
            save_image(img, img_path)
        else:
            img.save(img_path)

    print(f"All data saved to {root}")

def analyze_dirichlet_distribution(root="./data_splits", num_classes=10):
    """
    Phân tích số lượng ảnh mỗi class trong từng client (đã lưu dạng folder).

    Parameters
    ----------
    root : str
        Thư mục chứa dữ liệu đã lưu (gốc của client_i/).
    num_classes : int
        Số lớp của dataset (VD: 10 cho CIFAR-10).
    """
    client_dirs = [d for d in os.listdir(root) if d.startswith("client_")]
    client_dirs.sort(key=lambda x: int(x.split("_")[1]))

    class_counts = defaultdict(list)  # {class_k: [counts per client]}
    summary_table = []

    print("📊 Counting images per class per client ...")

    for cid, cname in enumerate(client_dirs):
        client_path = os.path.join(root, cname)
        total_imgs = 0
        per_class = []
        for k in range(num_classes):
            class_dir = os.path.join(client_path, f"class_{k}")
            count = len(os.listdir(class_dir)) if os.path.exists(class_dir) else 0
            total_imgs += count
            per_class.append(count)
            class_counts[k].append(count)
        summary_table.append(per_class)
        print(f"Client {cid:02d}: total={total_imgs} | per_class={per_class}")

    # Convert to numpy for plotting
    counts = np.array(summary_table)  # shape = (num_clients, num_classes)

    # --- Plot stacked bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(client_dirs))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for k in range(num_classes):
        ax.bar(
            np.arange(len(client_dirs)),
            counts[:, k],
            bottom=bottom,
            label=f"class {k}",
            color=colors[k]
        )
        bottom += counts[:, k]

    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution per client")
    ax.legend(ncol=5, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("noniid.jpg")

    # --- Print summary statistics ---
    print("\n📈 Summary: class imbalance ratio (max/min per class)")
    for k in range(num_classes):
        arr = np.array(class_counts[k])
        if arr.sum() == 0:
            continue
        imbalance = arr.max() / max(arr.min(), 1)
        print(f"Class {k}: max={arr.max()}, min={arr.min()}, ratio={imbalance:.2f}")

if __name__ == "__main__":
    # CIFAR-10 dataset
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    clients, testset = partition_data_dirichlet(num_clients=64, alpha=0.8, dataset_name="cifar10")
    save_dirichlet_datasets(clients, testset, root="./data_splits")
    analyze_dirichlet_distribution()
    # trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # train_class_indices = group_by_class(trainset)
    # test_class_indices = group_by_class(testset)
    # save_split(trainset, train_class_indices, out_dir="./data_splits/train", num_splits=64)
    # save_split(testset, test_class_indices, out_dir="./data_splits/test", num_splits=64)
    # save_dirichlet_global_split(trainset, alpha=0.5, num_clients=64, out_dir="./data_splits_dirichlet/train", seed=42)
    # save_dirichlet_global_split(testset, alpha=0.5, num_clients=64, out_dir="./data_splits_dirichlet/test", seed=42)