from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CLASSES = [
    "stop_sign",
    "person",
    "bicycle",
    "bus",
    "truck",
    "car",
    "motorbike",
    "reflective_cone",
    "ashcan",
    "warning_column",
    "spherical_roadblock",
    "pole",
    "dog",
    "tricycle",
    "fire_hydrant",
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
        raise FileNotFoundError(
            f"Image not found for {image_id} in {images_dir}"
        )

    return matches[0]


def convert_annotation(
    xml_path: Path,
    output_path: Path,
) -> Counter[str]:

    root = ET.parse(xml_path).getroot()

    width = float(root.findtext("size/width", "0"))
    height = float(root.findtext("size/height", "0"))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    lines: list[str] = []
    counts: Counter[str] = Counter()

    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")

        if label not in CLASS_TO_ID or box is None:
            continue

        xmin = max(float(box.findtext("xmin", "0")), 0.0)
        ymin = max(float(box.findtext("ymin", "0")), 0.0)
        xmax = min(float(box.findtext("xmax", "0")), width)
        ymax = min(float(box.findtext("ymax", "0")), height)

        box_width = xmax - xmin
        box_height = ymax - ymin

        if box_width <= 1 or box_height <= 1:
            continue

        center_x = ((xmin + xmax) / 2) / width
        center_y = ((ymin + ymax) / 2) / height
        normalized_width = box_width / width
        normalized_height = box_height / height

        lines.append(
            f"{CLASS_TO_ID[label]} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{normalized_width:.6f} "
            f"{normalized_height:.6f}"
        )

        counts[label] += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert obstacle VOC annotations to YOLO format"
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the Obstacle Dataset folder",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/obstacle_yolo"),
        help="Output YOLO dataset folder",
    )

    args = parser.parse_args()

    source_dir = args.source
    output_dir = args.output

    image_sets_dir = source_dir / "ImageSets" / "Main"

    if not image_sets_dir.exists():
        raise SystemExit(f"Missing directory: {image_sets_dir}")

    total_counts: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        images_dir = source_dir / f"img-{split}"
        annotations_dir = source_dir / f"ann-{split}"
        split_file = image_sets_dir / f"{split}.txt"

        for required in (
            images_dir,
            annotations_dir,
            split_file,
        ):
            if not required.exists():
                raise SystemExit(f"Required path missing: {required}")

        image_output = output_dir / "images" / split
        label_output = output_dir / "labels" / split

        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        ids = [
            line.split()[0]
            for line in split_file.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        converted = 0

        for image_id in ids:
            image_path = find_image(images_dir, image_id)
            xml_path = annotations_dir / f"{image_id}.xml"

            if not xml_path.exists():
                print(
                    f"WARNING: annotation missing for {image_id}; skipped"
                )
                continue

            shutil.copy2(
                image_path,
                image_output / image_path.name,
            )

            counts = convert_annotation(
                xml_path,
                label_output / f"{image_id}.txt",
            )

            total_counts.update(counts)
            converted += 1

        print(
            f"{split}: converted {converted}/{len(ids)} images"
        )

    print("\nObject counts:")

    for label in CLASSES:
        print(f"  {label}: {total_counts[label]}")

    print(
        f"\nDataset ready at: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()