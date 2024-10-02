import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from timm.optim import create_optimizer_v2
from cs.models.pyramidvit import pvt_medium_semantic as CustomCNN


def train_model(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()  # 每个epoch更新学习率
    return running_loss / len(train_loader)


def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), correct / total


def main():
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载ImageNet数据集
    train_dataset = datasets.ImageNet(
        root="./data/imagenet", split="train", transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # 加载验证集
    val_dataset = datasets.ImageNet(
        root="./data/imagenet", split="val", transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 初始化模型和迁移到设备
    model = CustomCNN().to(device)

    # 使用 Timm 创建优化器和调度器
    optimizer = create_optimizer_v2(model, opt="adamw", lr=0.001, weight_decay=0.02)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=90, lr_min=1e-6, warmup_lr_init=1e-4, warmup_t=5
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    num_epochs = 90
    for epoch in range(num_epochs):
        train_loss = train_model(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
        )

    # 保存模型
    torch.save(model.state_dict(), "pvt_medium_semantic.pth")


if __name__ == "__main__":
    main()
