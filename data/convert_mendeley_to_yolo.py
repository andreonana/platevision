"""
Conversion du dataset Mendeley Ghana → format YOLOv5 (même structure que Roboflow Nigeria).

Les images Mendeley sont des gros plans de plaques (plaque = image entière),
donc bbox = centre image, taille pleine : 0 0.5 0.5 1.0 1.0

Sortie : data/raw/mendeley_yolo/
  train/images/  train/labels/
  valid/images/  valid/labels/
  test/images/   test/labels/
  data.yaml

Usage :
  python data/convert_mendeley_to_yolo.py
  python data/convert_mendeley_to_yolo.py --split 0.7 0.15 0.15
  python data/convert_mendeley_to_yolo.py --src data/raw/mendeley --out data/raw/mendeley_yolo
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF"}

YOLO_LABEL = "0 0.5 0.5 1.0 1.0\n"   # classe 0, plaque = image entière

DATA_YAML = """\
path: {output_dir}
train: train/images
val:   valid/images
test:  test/images

nc: 1
names:
  - plaque_immatriculation
"""


def collect_images(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix in IMG_EXTS)


def convert(src: Path, out: Path, split: tuple[float, float, float], seed: int) -> None:
    images = collect_images(src)
    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans : {src}")

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])

    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_valid],
        "test":  images[n_train + n_valid:],
    }

    for split_name, imgs in splits.items():
        img_dir = out / split_name / "images"
        lbl_dir = out / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            dest_img = img_dir / img_path.name
            # En cas de doublon de nom entre régions, préfixer avec le dossier parent
            if dest_img.exists():
                dest_img = img_dir / f"{img_path.parent.name}_{img_path.name}"
            shutil.copy2(img_path, dest_img)

            lbl_path = lbl_dir / (dest_img.stem + ".txt")
            lbl_path.write_text(YOLO_LABEL, encoding="utf-8")

        print(f"  {split_name:5s} : {len(imgs):5d} images")

    (out / "data.yaml").write_text(
        DATA_YAML.format(output_dir=str(out.resolve())), encoding="utf-8"
    )

    print(f"\n  Total : {n} images converties → {out}")
    print(f"  data.yaml généré.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convertit le dataset Mendeley Ghana au format YOLOv5"
    )
    parser.add_argument(
        "--src", default="data/raw/mendeley",
        help="Dossier source Mendeley (défaut: data/raw/mendeley)"
    )
    parser.add_argument(
        "--out", default="data/raw/mendeley_yolo",
        help="Dossier de sortie (défaut: data/raw/mendeley_yolo)"
    )
    parser.add_argument(
        "--split", nargs=3, type=float, default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VALID", "TEST"),
        help="Proportions train/valid/test (défaut: 0.70 0.15 0.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Graine aléatoire (défaut: 42)"
    )
    args = parser.parse_args()

    total = sum(args.split)
    if abs(total - 1.0) > 1e-6:
        parser.error(f"La somme des proportions doit être 1.0 (obtenu : {total:.3f})")

    src = Path(args.src)
    out = Path(args.out)

    print(f"\nConversion Mendeley Ghana → YOLOv5")
    print(f"  Source : {src}")
    print(f"  Sortie : {out}")
    print(f"  Split  : train={args.split[0]:.0%}  valid={args.split[1]:.0%}  test={args.split[2]:.0%}\n")

    convert(src, out, tuple(args.split), args.seed)


if __name__ == "__main__":
    main()
