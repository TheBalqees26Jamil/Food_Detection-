"""
train_efficientnet.py
تدريب EfficientNet مع تحسينات (augmentation, AdamW, label smoothing, mixed-precision...).
استخدمي args لتعديل المسارات والهباراميتر بسهولة.
"""

import os
import time
import copy
import argparse
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# حاول استيراد timm لمرونة أعلى؛ إذا غير متوفر نستخدم torchvision
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

# ---------------- Helpers ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(model_name, num_classes, pretrained=True):
    """
    يحاول استخدام timm إذا متاح (أفضل تشكيلة)، وإلا يستخدم torchvision EfficientNet-B0 كاحتياط.
    model_name examples: 'efficientnet_b0', 'efficientnet_b3' (timm names may differ)
    """
    if TIMM_AVAILABLE:
        # إذا استخدمت timm، استخدمي اسماء timm (مثال 'efficientnet_b0' يعمل)
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        return model
    else:
        # fallback to torchvision efficientnet_b0
        from torchvision import models
        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
            return model
        else:
            # if user asked for b3 but torchvision not available, default to resnet18 as fallback
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = torch.sum(preds == labels).item()
    return correct

# --------------- Training Loop ---------------
def train_loop(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.15)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "test")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=val_dir,   transform=val_transform)

    num_classes = len(train_dataset.classes)
    print("Number of classes:", num_classes)
    print("Classes:", train_dataset.classes)

    # توازن العيّنات (اختياري، يفيد لو فيه عدم توازن)
    class_counts = [0]*num_classes
    for _, lbl in train_dataset:
        class_counts[lbl] += 1
    print("Train class counts:", class_counts)

    if args.balance_sampler:
        # وزن عكسي حسب عدد العينات في كل فئة
        class_weights = [0.0]*num_classes
        for i, c in enumerate(class_counts):
            class_weights[i] = 1.0 / (c + 1e-6)
        sample_weights = [class_weights[label] for _, label in train_dataset.imgs]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    model = build_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)

    # criterion with label smoothing (if PyTorch supports)
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    except TypeError:
        # older torch versions may not support label_smoothing
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler: cosine annealing or ReduceLROnPlateau
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # bookkeeping
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "train_log.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += compute_accuracy(outputs, labels)
            total_samples += labels.size(0)

        train_loss = running_loss / total_samples
        train_acc  = running_corrects / total_samples * 100.0

        # validation
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += compute_accuracy(outputs, labels)
                val_total += labels.size(0)

        val_loss = val_running_loss / (val_total + 1e-12)
        val_acc  = val_running_corrects / (val_total + 1e-12) * 100.0

        # scheduler step
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_acc)

        # logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  LR: {lr:.6f}  time: {time.time()-t0:.1f}s")

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, round(train_loss,4), round(train_acc,4), round(val_loss,4), round(val_acc,4), lr])

        # early stopping & save best
        if val_acc > best_val_acc + 0.0001:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            save_path = os.path.join(args.save_dir, f"best_model_{args.model}.pth")
            torch.save(best_model_wts, save_path)
            print(f"  >> New best model saved: {save_path}  (Val Acc: {best_val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"  >> No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    # load best weights and save final
    model.load_state_dict(best_model_wts)
    final_path = os.path.join(args.save_dir, f"final_{args.model}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.2f}%")
    print(f"Final model saved to: {final_path}")
    print(f"CSV log saved to: {csv_path}")

# ----------------- CLI -----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="food_data_base", help="path to dataset (train/ test/)")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="model name (timm or 'efficientnet_b0')")
    parser.add_argument("--img_size", type=int, default=224, help="input image size")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--scheduler", type=str, default="reduce", choices=["reduce","cosine"])
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--balance_sampler", action="store_true", help="use weighted sampler to handle imbalance")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
