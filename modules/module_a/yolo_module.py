"""
PlateVision — Module A — YOLO
Détection de plaques d'immatriculation
MINT/DGI Cameroun — UCAC-ICAM
"""

import os
import cv2
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = r"C:\Users\ibnal\Desktop\PROJET IA X3\archive\train"
TEST_PATH  = r"C:\Users\ibnal\Desktop\PROJET IA X3\archive\test"
VALID_PATH = r"C:\Users\ibnal\Desktop\PROJET IA X3\archive\valid"

PATH_YOLO_DS = "data/processed/yolo_dataset"
PATH_FIGURES = "outputs/figures"

for split in ['train', 'val', 'test']:
    os.makedirs(
        os.path.join(PATH_YOLO_DS, "images", split),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(PATH_YOLO_DS, "labels", split),
        exist_ok=True
    )
os.makedirs(PATH_FIGURES, exist_ok=True)


# ════════════════════════════════════════════════
# PRÉPARATION DU DATASET YOLO
# ════════════════════════════════════════════════

def preparer_dataset_yolo(df_train, df_test, df_valid):
    """
    Convertit les annotations CSV au format YOLO.
    Format YOLO : class_id x_centre y_centre largeur hauteur
    (valeurs normalisées 0-1)
    """
    print("\n" + "─" * 55)
    print("  YOLO — PRÉPARATION DU DATASET")
    print("─" * 55)

    splits_config = {
        'train': (df_train, TRAIN_PATH),
        'val'  : (df_valid, VALID_PATH),
        'test' : (df_test,  TEST_PATH),
    }

    total = 0

    for split_nom, (df_split, img_folder) in splits_config.items():
        if df_split.empty:
            continue

        img_out = os.path.join(PATH_YOLO_DS, "images", split_nom)
        lbl_out = os.path.join(PATH_YOLO_DS, "labels", split_nom)

        groupes = df_split.groupby('filename')
        print(f"\n  [{split_nom}] {len(groupes)} images...")

        for filename, groupe in tqdm(
            groupes, desc=f"  Conv. {split_nom}"
        ):
            src = os.path.join(img_folder, filename)
            if not os.path.exists(src):
                continue

            img = cv2.imread(src)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            # Copier l'image
            shutil.copy2(src, os.path.join(img_out, filename))

            # Créer le label YOLO
            nom_base   = os.path.splitext(filename)[0]
            label_path = os.path.join(lbl_out, f"{nom_base}.txt")
            lignes     = []

            for _, row in groupe.iterrows():
                xmin = float(row['xmin'])
                ymin = float(row['ymin'])
                xmax = float(row['xmax'])
                ymax = float(row['ymax'])

                x_c = max(0, min(1, ((xmin + xmax) / 2) / w_img))
                y_c = max(0, min(1, ((ymin + ymax) / 2) / h_img))
                w   = max(0, min(1, (xmax - xmin) / w_img))
                h   = max(0, min(1, (ymax - ymin) / h_img))

                lignes.append(
                    f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                )

            with open(label_path, 'w') as f:
                f.write('\n'.join(lignes))
            total += 1

    # Créer data.yaml
    yaml_path    = os.path.join(PATH_YOLO_DS, 'data.yaml')
    yaml_content = f"""# PlateVision — YOLO Dataset
path: {os.path.abspath(PATH_YOLO_DS)}
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: License_Plate
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Vérification
    print(f"\n  ✓ {total} images converties")
    for split in ['train', 'val', 'test']:
        n_imgs = len(os.listdir(
            os.path.join(PATH_YOLO_DS, "images", split)
        ))
        n_lbls = len(os.listdir(
            os.path.join(PATH_YOLO_DS, "labels", split)
        ))
        print(f"  [{split:5s}] images:{n_imgs:4d} | "
              f"labels:{n_lbls:4d}")

    print(f"  ✓ data.yaml : {yaml_path}")
    return yaml_path


# ════════════════════════════════════════════════
# ENTRAÎNEMENT YOLO
# ════════════════════════════════════════════════

def entrainer_yolo(yaml_path, device,
                   epochs=30, imgsz=640):
    """
    Fine-tune YOLOv8n sur le dataset de plaques.
    Transfer learning depuis les poids COCO.
    """
    print("\n" + "─" * 55)
    print("  YOLO — ENTRAÎNEMENT YOLOv8n")
    print("─" * 55)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ⚠ pip install ultralytics")
        return None

    # Vérifier qu'il y a des images
    n_train = len(os.listdir(
        os.path.join(PATH_YOLO_DS, "images", "train")
    ))
    if n_train == 0:
        print("  ⚠ Aucune image dans train/ — "
              "lance d'abord preparer_dataset_yolo()")
        return None

    print(f"\n  Modèle  : YOLOv8n (transfer learning COCO)")
    print(f"  Epochs  : {epochs}")
    print(f"  ImgSize : {imgsz}×{imgsz}")
    print(f"  Device  : {device}")
    print(f"  Train   : {n_train} images")

    model   = YOLO('yolov8n.pt')
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=0 if device == 'cuda' else 'cpu',
        project='outputs/yolo',
        name='platevision_yolo',
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        augment=True,
        lr0=0.01,
        weight_decay=0.0005,
        exist_ok=True,
    )

    best = 'outputs/yolo/platevision_yolo/weights/best.pt'
    print(f"\n  ✓ Entraînement terminé")
    print(f"  ✓ Meilleur modèle : {best}")

    # Évaluation
    try:
        metrics = model.val(data=yaml_path, split='test')
        print(f"\n  {'MÉTRIQUES':^40}")
        print(f"  {'─'*40}")
        print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    except Exception as e:
        print(f"  ⚠ Évaluation : {e}")

    return best


# ════════════════════════════════════════════════
# TEST ET VISUALISATION
# ════════════════════════════════════════════════

def tester_yolo(model_path, df_test, n=6):
    """
    Teste YOLO sur n images et affiche les détections.
    Vert = détection YOLO / Rouge pointillé = ground truth CSV
    """
    print("\n" + "─" * 55)
    print("  YOLO — TEST SUR IMAGES")
    print("─" * 55)

    if not os.path.exists(model_path):
        print(f"  ⚠ Modèle introuvable : {model_path}")
        return

    from ultralytics import YOLO
    model   = YOLO(model_path)
    exemples = df_test.head(n)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'YOLO — Détections vs Ground Truth\n'
        'PlateVision MINT/DGI Cameroun',
        fontsize=13, fontweight='bold'
    )

    for ax, (_, row) in zip(axes.flat, exemples.iterrows()):
        img = cv2.imread(row['image_full_path'])
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results_y = model(row['image_full_path'], verbose=False)
        ax.imshow(img_rgb)

        # Bounding boxes YOLO détectées
        for res in results_y:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_det        = float(box.conf[0])
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='lime',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 5,
                    f"Plaque {conf_det:.2f}",
                    color='lime', fontsize=8,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='black', alpha=0.6)
                )

        # Ground truth (CSV)
        rect_gt = patches.Rectangle(
            (row['xmin'], row['ymin']),
            row['xmax']-row['xmin'],
            row['ymax']-row['ymin'],
            linewidth=2, edgecolor='red',
            facecolor='none', linestyle='--'
        )
        ax.add_patch(rect_gt)
        ax.set_title(
            os.path.basename(row['image_full_path'])[:25],
            fontsize=8
        )
        ax.axis('off')

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor='lime', label='Détection YOLO'),
            Patch(facecolor='red',  label='Ground Truth'),
        ],
        loc='lower center', ncol=2, fontsize=10
    )

    plt.tight_layout()
    p = os.path.join(PATH_FIGURES, 'yolo_detections.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure : {p}")