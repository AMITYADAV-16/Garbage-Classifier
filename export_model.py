from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

MODEL_DIR   = Path("models")
SYNSET_PATH = MODEL_DIR / "synset.txt"

# These must match exactly what was trained
CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "paper", "plastic", "shoes", "trash"
]

def main():
    device = torch.device("cpu")

    print("Loading MobileNetV2 architecture...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

    print("Loading best checkpoint...")
    checkpoint = torch.load(MODEL_DIR / "checkpoint_best.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("Exporting to TorchScript for Java...")
    example = torch.randn(1, 3, 224, 224)
    traced  = torch.jit.trace(model, example)
    traced.save(str(MODEL_DIR / "garbage_mobilenetv2.pt"))

    print("Writing synset.txt...")
    SYNSET_PATH.write_text("\n".join(CLASS_NAMES))

    print("\n✓ Done! Files saved:")
    print(f"  {MODEL_DIR / 'garbage_mobilenetv2.pt'}")
    print(f"  {SYNSET_PATH}")
    print("\nNow replace HelloController.java and restart your Java app!")

if __name__ == "__main__":
    main()
