from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


CLASSES = [
    "stop_sign", "person", "bicycle", "bus", "truck", "car", "motorbike",
    "reflective_cone", "ashcan", "warning_column", "spherical_roadblock",
    "pole", "dog", "tricycle", "fire_hydrant",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_image(images_dir: Path, image_id: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    matches = list(images_dir.glob(f"{image_id}.*"))
    if not matches:
        raise FileNotFoundError(f"Image not found for {image_id}")
    return matches[0]


def convert_annotation(xml_path: Path, output_path: Path) -> Counter[str]:
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("size/width", "0"))
    height = float(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")
    lines: list[str] = []
    counts: Counter[str] = Counter()
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        if label not in CLASS_TO_ID:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = max(float(box.findtext("xmin", "0")), 0.0)
        ymin = max(float(box.findtext("ymin", "0")), 0.0)
        xmax = min(float(box.findtext("xmax", "0")), width)
        ymax = min(float(box.findtext("ymax", "0")), height)
        box_width, box_height = xmax - xmin, ymax - ymin
        if box_width <= 1 or box_height <= 1:
            continue
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        lines.append(
            f"{CLASS_TO_ID[label]} {x_center:.6f} {y_center:.6f} "
            f"{box_width / width:.6f} {box_height / height:.6f}"
        )
        counts[label] += 1
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OD VOC files to Ultralytics YOLO layout")
    parser.add_argument("--source", type=Path, required=True, help="VOC root containing JPEGImages/Annotations/ImageSets/Main")
    parser.add_argument("--output", type=Path, default=Path("datasets/obstacle_yolo"))
    args = parser.parse_args()
    images_dir = args.source / "JPEGImages"
    annotations_dir = args.source / "Annotations"
    splits_dir = args.source / "ImageSets" / "Main"
    for required in (images_dir, annotations_dir, splits_dir):
        if not required.exists():
            raise SystemExit(f"Required directory missing: {required}")

    total_counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise SystemExit(f"Split file missing: {split_file}")
        image_output = args.output / "images" / split
        label_output = args.output / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        image_ids = [line.strip().split()[0] for line in split_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        converted = 0
        for image_id in image_ids:
            image_path = find_image(images_dir, image_id)
            xml_path = annotations_dir / f"{image_id}.xml"
            if not xml_path.exists():
                print(f"WARNING: annotation missing for {image_id}; skipped")
                continue
            shutil.copy2(image_path, image_output / image_path.name)
            total_counts.update(convert_annotation(xml_path, label_output / f"{image_id}.txt"))
            converted += 1
        print(f"{split}: converted {converted}/{len(image_ids)} images")

    print("\nObject counts:")
    for label in CLASSES:
        print(f"  {label}: {total_counts[label]}")
    missing = [label for label in CLASSES if total_counts[label] == 0]
    if missing:
        raise SystemExit(f"Dataset validation failed; zero objects for: {missing}")
    print(f"\nDataset ready at: {args.output.resolve()}")


if __name__ == "__main__":
    main()
