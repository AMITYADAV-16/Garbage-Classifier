"""
FAST FINE-TUNER — ~15-20 mins on CPU
Freezes all layers except the final classifier.
Fixes class bias without full retraining.
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("models/garbage_mobilenetv2.pt")   # your existing model
DATA_DIR     = Path("prepared_data")
OUTPUT_PATH  = Path("models/garbage_mobilenetv2.pt")   # overwrite with fixed model
SYNSET_PATH  = Path("models/synset.txt")

NUM_EPOCHS   = 5       # only 5 epochs needed since we freeze most layers
BATCH_SIZE   = 16
LR           = 1e-3
# ─────────────────────────────────────────────────────────────────────────────

def get_weighted_sampler(dataset):
    class_counts = np.bincount(dataset.targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in dataset.targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load class names from synset ─────────────────────────────────────────
    class_names = SYNSET_PATH.read_text().strip().splitlines()
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", train_transforms)
    val_dataset   = datasets.ImageFolder(DATA_DIR / "val",   val_transforms)

    # Print distribution
    class_counts = np.bincount(train_dataset.targets)
    print("\nClass distribution:")
    for name, count in zip(class_names, class_counts):
        print(f"  {name:<20} {count} images")
    print()

    sampler      = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    # ── Load existing model ───────────────────────────────────────────────────
    print(f"Loading existing model from {MODEL_PATH}...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # Load the traced model weights back into the architecture
    # We load via TorchScript then copy weights
    traced_model = torch.jit.load(str(MODEL_PATH), map_location=device)

    # Copy weights from traced model to regular model
    traced_state = {k: v for k, v in traced_model.state_dict().items()}
    model.load_state_dict(traced_state, strict=False)
    model = model.to(device)

    # ── FREEZE all layers except the classifier ───────────────────────────────
    # This is the key — we only retrain the final layer (very fast!)
    for param in model.features.parameters():
        param.requires_grad = False

    # Only the classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print("(Frozen everything except final classifier — this is why it's fast)\n")

    # ── Loss with class weights ───────────────────────────────────────────────
    class_weights_tensor = torch.tensor(
        [1.0 / c for c in class_counts], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Only optimize the unfrozen classifier parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # ── Fine-tuning loop ──────────────────────────────────────────────────────
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.2e})")
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
                    # Save best checkpoint
                    torch.save(model.state_dict(), "models/finetune_best.pth")
                    print(f"  ✓ Best saved ({best_acc*100:.1f}%)")

    print(f"\nFine-tuning complete. Best val accuracy: {best_acc*100:.1f}%")

    # ── Export for Java ───────────────────────────────────────────────────────
    print("\nExporting fixed model for Java...")
    model.load_state_dict(torch.load("models/finetune_best.pth", map_location=device))
    model.eval()
    example = torch.randn(1, 3, 224, 224).to(device)
    traced  = torch.jit.trace(model, example)
    traced.save(str(OUTPUT_PATH))
    print(f"✓ Model saved to {OUTPUT_PATH}")
    print("Restart your Java app to use the fixed model!")

if __name__ == "__main__":
    main()