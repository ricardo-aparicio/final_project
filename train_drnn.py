#!/usr/bin/env python3
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt 
import numpy as np 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("dataset_split")
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3

# --- Dataset & DataLoaders ---

train_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),              # [0,1]
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # opcional
])

val_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

train_ds = ImageFolder(DATA_ROOT / "train", transform=train_tfms)
val_ds   = ImageFolder(DATA_ROOT / "val",   transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)


# --- Definición de un bloque residual simple (DRNN-like) ---

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class DRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = DRNN(num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# --- Función para graficar métricas ---  <--- NUEVO

def plot_metrics(train_losses, val_losses, train_accs, val_accs,
                 out_file="drnn_training_curves.png"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.grid(True)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train acc")
    plt.plot(epochs, val_accs, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Training curves saved to {out_file}")


# --- Entrenamiento y validación ---

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch}: Train loss {epoch_loss:.4f}, "
          f"acc {epoch_acc:.4f}")
    return epoch_loss, epoch_acc   # <--- NUEVO


def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    loss_val = running_loss / total
    acc_val = correct / total
    print(f"  Val loss {loss_val:.4f}, acc {acc_val:.4f}")
    return loss_val, acc_val


best_val = 0.0

# Listas para guardar métricas  <--- NUEVO
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(epoch)  # <--- cambio
    val_loss, acc_val = evaluate()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(acc_val)

    if acc_val > best_val:
        best_val = acc_val
        torch.save(model.state_dict(), "best_drnn.pth")
        print("  -> New best model saved")

print("Training finished. Best val acc:", best_val)

#Guardar historial
np.savez(
    "drnn_history_20ep.npz",
    train_loss=np.array(train_losses, dtype=float),
    val_loss=np.array(val_losses, dtype=float),
    train_acc=np.array(train_accs, dtype=float),
    val_acc=np.array(val_accs, dtype=float),
)
print("DRNN history saved to drnn_history_20ep.npz")

# Graficar curvas de entrenamiento  <--- NUEVO
plot_metrics(train_losses, val_losses, train_accs, val_accs)

# --- Evaluación en test set ---

test_tfms = val_tfms  # mismas transformaciones que val

test_ds = ImageFolder(DATA_ROOT / "test", transform=test_tfms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)

model.load_state_dict(torch.load("best_drnn.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Test accuracy: {test_acc:.4f}")
print("Test samples:", total)

