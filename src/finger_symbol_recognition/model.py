import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EMNISTNet(nn.Module):
    def __init__(self, num_classes=47):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def train_model(
    model: nn.Module,
    device: torch.device,
    epochs: int,
    batch_size: int = 128,
    data_root: Path | str = PROJECT_ROOT / "data",
):
    model.to(device)

    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    emnist_transform = transforms.Compose(
        [
            lambda img: transforms.functional.rotate(img, -90),
            lambda img: transforms.functional.hflip(img),
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
        ]
    )

    train_dataset = EMNIST(
        root=str(data_root), split="balanced", train=True, transform=emnist_transform
    )
    test_dataset = EMNIST(
        root=str(data_root), split="balanced", train=False, transform=emnist_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()

        running_loss, total, correct = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                _, predicted = output.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(
            f"Epoch [{epoch + 1:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} -"
            f" Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )
    return model


if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = EMNISTNet(num_classes=47)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = train_model(model, device, epochs=20, batch_size=512)

    print("Export to ONNX")
    model.eval()
    model.to("cpu")

    onnx_file = PROJECT_ROOT / "models" / "emnist_balanced.onnx"
    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_file),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model saved as: {onnx_file}")
