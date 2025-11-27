import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from decen_learn.models.resnet_cifar import ResNet20


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def test_epoch(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data augmentation (chuẩn CIFAR-10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    # Create model
    model = ResNet20().to(device)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    # Scheduler MultiStep
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    # Training loop
    EPOCHS = 200
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = test_epoch(model, testloader, criterion, device)
        scheduler.step()

        print(f"[Epoch {epoch:3d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%   "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    torch.save(model.state_dict(), "resnet20_cifar10.pth")
    print("Training done. Model saved.")


if __name__ == "__main__":
    main()
    #[Epoch   1] Train Loss: 1.6893 | Train Acc: 37.91%   Test Loss: 1.4807 | Test Acc: 47.75%
    #[Epoch  50] Train Loss: 0.4366 | Train Acc: 84.78%   Test Loss: 0.8248 | Test Acc: 74.58%
    #[Epoch 100] Train Loss: 0.4126 | Train Acc: 85.83%   Test Loss: 0.6188 | Test Acc: 80.00%
    #[Epoch 150] Train Loss: 0.1367 | Train Acc: 95.26%   Test Loss: 0.3928 | Test Acc: 88.45%
    #[Epoch 200] Train Loss: 0.0284 | Train Acc: 99.36%   Test Loss: 0.2753 | Test Acc: 92.34%

    #No Scheduler:
    #[Epoch   1] Train Loss: 1.7354 | Train Acc: 36.05%   Test Loss: 1.4746 | Test Acc: 47.00%
    #[Epoch  50] Train Loss: 0.4418 | Train Acc: 84.77%   Test Loss: 0.8268 | Test Acc: 74.53%
    #[Epoch 100] Train Loss: 0.4201 | Train Acc: 85.44%   Test Loss: 0.6330 | Test Acc: 79.36%
    #[Epoch 200] Train Loss: 0.4017 | Train Acc: 86.26%   Test Loss: 0.8783 | Test Acc: 73.06%