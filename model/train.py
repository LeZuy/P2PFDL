import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Improved add_pixel_pattern
# ----------------------------
def add_pixel_pattern(params, ori_image, adversarial_index):
    """
    Safe, vectorized insert of pixel trigger into a single image tensor.
    - ori_image: torch.Tensor with shape (C,H,W)
    - params: dict containing 'trigger_num', '<i>_poison_pattern' lists, optional 'trigger_value'
    - adversarial_index: -1 => aggregate all triggers, else use specific pattern
    Returns: new tensor (clone) with trigger pixels set (on same device/dtype as ori_image)
    """
    if not torch.is_tensor(ori_image):
        raise ValueError("ori_image must be a torch.Tensor (C,H,W)")

    image = ori_image.clone()  # fast, keeps dtype/device
    C, H, W = image.shape
    trigger_value = float(params.get('trigger_value', 1.0))

    # build poison_patterns list of (r,c)
    poison_patterns = []
    if adversarial_index == -1:
        n_tr = int(params.get('trigger_num', 0))
        for i in range(n_tr):
            p = params.get(f"{i}_poison_pattern", [])
            if p is None:
                continue
            if isinstance(p, (list, tuple)):
                poison_patterns.extend(p)
            else:
                try:
                    poison_patterns.extend(list(p))
                except Exception:
                    continue
    else:
        p = params.get(f"{adversarial_index}_poison_pattern", [])
        if isinstance(p, (list, tuple)):
            poison_patterns = list(p)
        else:
            try:
                poison_patterns = list(p)
            except Exception:
                poison_patterns = []

    if len(poison_patterns) == 0:
        return image

    # normalize positions and build tensor
    coords = []
    for pos in poison_patterns:
        try:
            r = int(pos[0]); c = int(pos[1])
            coords.append((r, c))
        except Exception:
            continue

    if not coords:
        return image

    coord_t = torch.tensor(coords, dtype=torch.long, device=image.device)  # (N,2)
    rows = coord_t[:, 0]; cols = coord_t[:, 1]

    # bounds check
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    if valid.sum().item() == 0:
        return image

    rows = rows[valid]; cols = cols[valid]
    val = torch.tensor(trigger_value, dtype=image.dtype, device=image.device)

    # For color images (C>=3), set across all channels. For grayscale (C==1), set channel 0.
    if C >= 3:
        # image[:, rows, cols] has shape (C, N) -> assignment broadcasts
        image[:, rows, cols] = val
    else:
        image[0, rows, cols] = val

    return image


# ----------------------------
# PoisonedDataset wrapper
# ----------------------------
class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, params, adversarial_index=-1):
        self.base = base_dataset
        self.params = params
        self.adversarial_index = adversarial_index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _ = self.base[idx]  # (image, orig_label)
        poisoned_image = add_pixel_pattern(self.params, image, self.adversarial_index)
        poisoned_label = torch.tensor(self.params['poison_label_swap'], dtype=torch.long)
        return poisoned_image, poisoned_label

# ----------------------------
# Loaders (use batch_size arg)
# ----------------------------
def get_trainloader(root_path, batch_size=64, dataset_name="cifar10", num_workers=0):
    if dataset_name.lower() == "cifar10":
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    else:
        # fallback nếu là dataset khác (giữ nguyên kiểu cũ)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    trainset = datasets.ImageFolder(root=root_path, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    return trainloader

def get_testloader(root_path, batch_size=64, dataset_name="cifar10", num_workers=0):
    if dataset_name.lower() == "cifar10":
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    testset = datasets.ImageFolder(root=root_path, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return testloader

def get_poison_testloader_from_dataset(
    root_path, params, batch_size=64, num_workers=0,
    adversarial_index=-1, shuffle=False, dataset_name="cifar10"
):
    if dataset_name.lower() == "cifar10":
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
    else:
        normalize_mean = (0.5, 0.5, 0.5)
        normalize_std = (0.5, 0.5, 0.5)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    base_testset = datasets.ImageFolder(root=root_path, transform=transform)
    poisoned_ds = PoisonedDataset(base_testset, params, adversarial_index=adversarial_index)
    loader = DataLoader(poisoned_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader

# ----------------------------
# get_poison_batch (on-the-fly poisoning for batches) - corrected
# ----------------------------
def get_poison_batch(params, bptt, device, adversarial_index=-1, evaluation=False, dpr=None):
    """
    bptt: (images, targets) where images: (B,C,H,W), targets: (B,)
    params: dict config, supports:
        - 'poisoning_per_batch' (int)   # explicit count (highest priority)
        - 'DPR' (float in [0,1])       # fraction of batch to poison if no explicit count
        - 'poison_label_swap' (int)    # label to assign to poisoned samples
        - 'poison_random' (bool)       # whether to choose poisoned indices randomly (default True)
        - 'poison_seed' (int)          # optional seed for deterministic selection
    evaluation: if True, poison entire batch (useful for eval)
    dpr: optional override for DPR
    """
    images, targets = bptt
    # defensive checks
    if images is None or targets is None:
        raise ValueError("get_poison_batch: images and targets must be provided")

    new_images = images.clone()
    new_targets = targets.clone()
    B = new_images.size(0)
    poison_count = 0

    if B == 0:
        # nothing to do
        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images, new_targets, 0

    # Move to device early so add_pixel_pattern (which may expect same device) works correctly
    new_images = new_images.to(device)
    new_targets = new_targets.to(device)

    # determine per_batch (priority: explicit 'poisoning_per_batch' > dpr param > params['DPR'] > 0)
    if 'poisoning_per_batch' in params:
        per_batch = int(params.get('poisoning_per_batch', 0))
    else:
        if dpr is None:
            dpr = float(params.get('DPR', 0.0))
        # clamp dpr
        try:
            dpr = float(dpr)
        except Exception:
            dpr = 0.0
        per_batch = int(round(dpr * B))

    per_batch = max(0, min(per_batch, B))

    # choose indices to poison
    if evaluation:
        idxs = torch.arange(B, device=new_images.device, dtype=torch.long)
    else:
        if per_batch == 0:
            idxs = torch.tensor([], dtype=torch.long, device=new_images.device)
        else:
            poison_random = bool(params.get('poison_random', True))
            if poison_random:
                seed = params.get('poison_seed', None)
                if seed is None:
                    idxs = torch.randperm(B, device=new_images.device)[:per_batch]
                else:
                    # deterministic selection via numpy RNG -> convert to tensor
                    rng = np.random.default_rng(seed)
                    sel = rng.choice(B, size=per_batch, replace=False)
                    idxs = torch.tensor(sel, dtype=torch.long, device=new_images.device)
            else:
                idxs = torch.arange(per_batch, device=new_images.device, dtype=torch.long)

    # apply poisoning
    poison_label = params.get('poison_label_swap', None)
    if poison_label is None:
        raise KeyError("params must contain 'poison_label_swap' for poisoned label value")

    for i in idxs.tolist():  # idxs is small so .tolist() is fine
        new_targets[i] = poison_label
        # call add_pixel_pattern — keep adversarial_index param as before
        new_images[i] = add_pixel_pattern(params, new_images[i], adversarial_index)
        poison_count += 1

    # finalize
    new_targets = new_targets.long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)

    return new_images, new_targets, poison_count


# ----------------------------
# train_batch / training_step (unchanged semantics, but robust)
# ----------------------------
def train_batch(model, batch, criterion, optimizer, device, params,
                agent_name_key=None, is_bad=False, adversarial_index=-1, evaluation=False, dpr=None):
    model.train()
    images, labels = batch

    if is_bad:
        images, labels, poison_count = get_poison_batch(
            params, (images, labels), device,
            adversarial_index=adversarial_index,
            evaluation=evaluation,
            dpr=dpr
        )
        # ensure dtype/device (get_poison_batch already moved to device)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    else:
        poison_count = 0
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        correct = preds.eq(labels).sum().item()
        batch_size = labels.size(0)

    return loss.item(), correct, batch_size, poison_count


def training_step(model, trainloader, criterion, optimizer, device, params,
                  agent_name_key=None, is_bad=False, adversarial_index=-1, dpr=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_poisoned = 0
    n_batches = 0

    for batch in trainloader:
        loss_item, correct, batch_size, poison_count = train_batch(
            model=model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            params=params,
            agent_name_key=agent_name_key,
            is_bad=is_bad,
            adversarial_index=adversarial_index,
            evaluation=False,
            dpr=dpr
        )
        total_loss += loss_item
        total_correct += correct
        total_samples += batch_size
        total_poisoned += poison_count
        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, acc, total_poisoned

@torch.no_grad()
def save_poisoned_batch(model, trainloader, device, params,
                        out_dir="./debug_poisoned_batch",
                        adversarial_index=-1,
                        dpr=None,
                        n_show=8):
    """
    Lưu 1 batch ảnh (gốc + sau khi chèn trigger) để kiểm tra kết quả.

    Args:
        model: (optional) mô hình đang train, chỉ dùng để consistency, không cần thiết.
        trainloader: DataLoader trả về (images, labels)
        device: thiết bị (cuda/cpu)
        params: dict chứa 'poison_label_swap', 'trigger_value', v.v.
        out_dir: thư mục lưu ảnh
        adversarial_index: pattern cụ thể (-1 = all)
        dpr: dynamic poisoning ratio (tùy chọn)
        n_show: số ảnh đầu tiên sẽ được lưu
    """
    os.makedirs(out_dir, exist_ok=True)

    # Lấy 1 batch duy nhất
    try:
        batch = next(iter(trainloader))
    except StopIteration:
        print("⚠️ Trainloader rỗng.")
        return

    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    # Chèn trigger
    poisoned_images, poisoned_labels, poison_count = get_poison_batch(
        params, (images, labels), device,
        adversarial_index=adversarial_index,
        evaluation=False,
        dpr=dpr
    )

    print(f"✅ Đã chèn {poison_count}/{len(images)} ảnh bị poison.")

    # Giới hạn số ảnh hiển thị/lưu
    n_show = min(n_show, len(images))

    # Normalize ảnh về [0,1] nếu cần
    def normalize_for_save(imgs):
        if imgs.min() < 0 or imgs.max() > 1:
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
        return imgs

    # Lưu từng ảnh (gốc và poison)
    for i in range(n_show):
        img_orig = normalize_for_save(images[i].detach().cpu())
        img_pois = normalize_for_save(poisoned_images[i].detach().cpu())

        label_orig = labels[i].item()
        label_pois = poisoned_labels[i].item()

        # Tên file dễ nhận biết
        fname_orig = f"orig_{i:02d}_label{label_orig}.png"
        fname_pois = f"pois_{i:02d}_label{label_pois}.png"

        save_image(img_orig, os.path.join(out_dir, fname_orig))
        save_image(img_pois, os.path.join(out_dir, fname_pois))

    print(f"📁 Ảnh đã được lưu trong thư mục: {out_dir}")

# Example main (fixes previous calling bugs)
# ----------------------------
if __name__ == "__main__":
    # load params
    with open('./configs/params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # trainloader = get_trainloader('../data_splits/split_0', batch_size=64)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # import torchvision.transforms as T
    # t = T.Compose([T.ToTensor()])
    # img, lbl = datasets.ImageFolder('../data_splits/split_0')[0]
    # pimg = add_pixel_pattern(params_loaded, img, 0)
    # print(pimg)
    # plt.saveimg("save.jpg", pimg)
    # from collections import Counter
    # trainset = datasets.ImageFolder(root="../data_splits/client_0")
    # print("Số lượng lớp:", len(trainset.classes))
    # print("Phân bố mẫu:", Counter([label for _, label in trainset.samples]))

    target_path = "./data_splits/test"
    root_path = "./data_splits"
    base_testset = datasets.ImageFolder(root=target_path, transform=transforms.ToTensor())
    poision_test = PoisonedDataset(base_testset, params, adversarial_index=-1)    

    print("Saving test set ...")
    test_dir = os.path.join(root_path, "test_poisioned")
    os.makedirs(test_dir, exist_ok=True)

    for idx in tqdm(range(len(poision_test))):
        img, label = poision_test[idx]
        class_dir = os.path.join(test_dir, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"img_{idx:05d}.png")

        if isinstance(img, torch.Tensor):
            if img.ndim == 2:
                img = img.unsqueeze(0)
            save_image(img, img_path)
        else:
            img.save(img_path)
