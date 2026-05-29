from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_IMAGE_DIR = PROJECT_ROOT / "datasets" / "siminspect_v1" / "raw_images"
DATASET_DIR = PROJECT_ROOT / "datasets" / "siminspect_v1"

SPLITS = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def make_dirs():
    for split in SPLITS:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def main():
    make_dirs()

    images = [
        path for path in RAW_IMAGE_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise RuntimeError(f"No images found in {RAW_IMAGE_DIR}")

    random.seed(42)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split, split_images in split_map.items():
        for image_path in split_images:
            dest = DATASET_DIR / "images" / split / image_path.name
            shutil.copy2(image_path, dest)

        print(f"{split}: {len(split_images)} images")

    print("Dataset split complete.")

if __name__ == "__main__":
    main()