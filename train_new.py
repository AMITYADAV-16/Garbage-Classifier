import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm
import numpy as np


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATASET_SRC  = Path("Dataset/standardized_256")  # new dataset
OUTPUT_DIR   = Path("models")
DATA_DIR     = Path("prepared_data")

NUM_EPOCHS   = 15
BATCH_SIZE   = 16
LR           = 1e-3
TRAIN_RATIO  = 0.8
SEED         = 42
# ─────────────────────────────────────────────


def prepare_data(src, dst_root, train_ratio=0.8, seed=42):
    random.seed(seed)
    if dst_root.exists():
        print(f"--- {dst_root} already exists, skipping preparation ---")
        return

    print(f"--- Preparing data in {dst_root} ---")
    for split in ["train", "val"]:
        (dst_root / split).mkdir(parents=True, exist_ok=True)

    for class_dir in src.iterdir():
        if not class_dir.is_dir():
            continue

        images = [p for p in class_dir.iterdir() if p.is_file()]
        random.shuffle(images)
        n_train = int(len(images) * train_ratio)

        for split_name, split_imgs in [("train", images[:n_train]), ("val", images[n_train:])]:
            dst = dst_root / split_name / class_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dst / img.name)

        print(f"  {class_dir.name:<15} train={n_train}  val={len(images)-n_train}")

    print("--- Data preparation complete ---\n")


def get_weighted_sampler(dataset):
    class_counts = np.bincount(dataset.targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in dataset.targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def train_model(data_dir, model_dir, num_epochs=15, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(data_dir / "train", train_transforms)
    val_dataset   = datasets.ImageFolder(data_dir / "val",   val_transforms)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    class_counts = np.bincount(train_dataset.targets)
    print("\nTraining set distribution:")
    for name, count in zip(class_names, class_counts):
        print(f"  {name:<20} {count} images")
    print()

    sampler      = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,   num_workers=0)

    # MobileNetV2
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    class_weights_tensor = torch.tensor(
        [1.0 / c for c in class_counts], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    model_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        print("-" * 45)

        for phase, loader, dataset in [
            ("train", train_loader, train_dataset),
            ("val",   val_loader,   val_dataset),
        ]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(loader, desc=f"  {phase}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataset)
            epoch_acc  = running_corrects.double() / len(dataset)
            print(f"  {phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} ({epoch_acc*100:.1f}%)")

            if phase == "val":
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), model_dir / "checkpoint_best.pth")
                    print(f"  ✓ New best saved! ({best_acc*100:.1f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_acc*100:.1f}%")

    print("\nExporting model for Java...")
    model.load_state_dict(torch.load(model_dir / "checkpoint_best.pth", map_location=device))
    model.eval()
    example = torch.randn(1, 3, 224, 224).to(device)
    traced  = torch.jit.trace(model, example)
    traced.save(str(model_dir / "garbage_mobilenetv2.pt"))
    (model_dir / "synset.txt").write_text("\n".join(class_names))
    print("Done! Model saved to:", model_dir / "garbage_mobilenetv2.pt")


def main():
    if not DATASET_SRC.exists():
        print(f"Error: Dataset folder not found at {DATASET_SRC}")
        print("Make sure 'Dataset/standardized_256' exists in your project folder.")
        return

    prepare_data(DATASET_SRC, DATA_DIR, TRAIN_RATIO, SEED)
    train_model(DATA_DIR, OUTPUT_DIR, NUM_EPOCHS, BATCH_SIZE, LR)


if __name__ == "__main__":
    main()