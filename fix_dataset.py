from pathlib import Path
from PIL import Image

data_dir = Path("prepared_data")
corrupted = []

for img_path in data_dir.rglob("*.*"):
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            corrupted.append(img_path)

print(f"Found {len(corrupted)} corrupted files:")
for p in corrupted:
    print(p)
    p.unlink()

print("Done!")