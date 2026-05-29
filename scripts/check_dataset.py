from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "siminspect_v1"

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

EXPECTED_OBJECTS_PER_IMAGE = 3
VALID_CLASS_IDS = {"0", "1", "2"}


def check_split(split):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    if not image_dir.exists():
        print(f"\n{split.upper()}")
        print(f"Missing image folder: {image_dir}")
        return

    if not label_dir.exists():
        print(f"\n{split.upper()}")
        print(f"Missing label folder: {label_dir}")
        return

    images = [
        path for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = [
        path for path in label_dir.iterdir()
        if path.suffix.lower() == ".txt"
    ]

    missing_labels = []
    extra_labels = []
    wrong_line_count = []
    bad_rows = []

    image_stems = {image_path.stem for image_path in images}
    label_stems = {label_path.stem for label_path in labels}

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        lines = [
            line.strip()
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        if len(lines) != EXPECTED_OBJECTS_PER_IMAGE:
            wrong_line_count.append((label_path.name, len(lines)))

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                bad_rows.append((label_path.name, line, "Expected 5 values"))
                continue

            class_id = parts[0]

            if class_id not in VALID_CLASS_IDS:
                bad_rows.append((label_path.name, line, "Invalid class ID"))
                continue

            try:
                values = [float(value) for value in parts[1:]]
            except ValueError:
                bad_rows.append((label_path.name, line, "Box values are not numbers"))
                continue

            for value in values:
                if value < 0 or value > 1:
                    bad_rows.append((label_path.name, line, "Box value outside 0-1 range"))
                    break

    for label_path in labels:
        if label_path.stem not in image_stems:
            extra_labels.append(label_path.name)

    print(f"\n{split.upper()}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Extra labels without image: {len(extra_labels)}")
    print(f"Wrong label line count: {len(wrong_line_count)}")
    print(f"Bad rows: {len(bad_rows)}")

    if missing_labels:
        print("\nExamples missing labels:")
        for name in missing_labels[:10]:
            print(f"  - {name}")

    if extra_labels:
        print("\nExamples extra labels:")
        for name in extra_labels[:10]:
            print(f"  - {name}")

    if wrong_line_count:
        print("\nExamples with wrong number of objects:")
        for name, count in wrong_line_count[:10]:
            print(f"  - {name}: {count} lines")

    if bad_rows:
        print("\nExamples with bad rows:")
        for name, row, reason in bad_rows[:10]:
            print(f"  - {name}: {row} ({reason})")


def main():
    for split in SPLITS:
        check_split(split)


if __name__ == "__main__":
    main()