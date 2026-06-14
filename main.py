"""
PlateVision — Main
Orchestre les 3 modules : OCR → YOLO → Clustering
MINT/DGI Cameroun — UCAC-ICAM
"""

import torch
import easyocr

from modules.module_a.ocr_module import (
    charger_tous_splits,
    run_ocr,
    visualiser_ocr,
)
from modules.module_a.yolo_module import (
    preparer_dataset_yolo,
    entrainer_yolo,
    tester_yolo,
)
from modules.module_a.clustering_module import (
    extraire_features,
    choisir_k,
    entrainer_kmeans,
    interpreter_et_visualiser,
)
from sklearn.preprocessing import normalize

# ── Détection GPU ──
import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU non disponible ! "
        "Vérifie l'installation PyTorch CUDA."
    )
DEVICE = 'cuda'
print(f"GPU confirmé : {torch.cuda.get_device_name(0)}")
print("=" * 55)
print("  PlateVision — Pipeline Complet")
print(f"  Device : {DEVICE.upper()}")
if DEVICE == 'cuda':
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
print("=" * 55)

# ════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ════════════════════════════════════════════════
df_train, df_test, df_valid = charger_tous_splits()

# ════════════════════════════════════════════════
# MODULE 1 — OCR
# ════════════════════════════════════════════════
print("\n\n" + "█" * 55)
print("  MODULE 1 — OCR")
print("█" * 55)

print("\n  Initialisation EasyOCR sur", DEVICE.upper(), "...")
reader = easyocr.Reader(
    ['en'], gpu=(DEVICE == 'cuda'), verbose=False
)
print("  ✓ EasyOCR prêt")

# ⚠ max_images=50 pour tester
# Change en None pour traiter tout le dataset
df_ocr = run_ocr(
    df_train, df_test, df_valid,
    reader,
    max_images=None
)
visualiser_ocr(df_ocr)

# ════════════════════════════════════════════════
# MODULE 2 — YOLO
# ════════════════════════════════════════════════
print("\n\n" + "█" * 55)
print("  MODULE 2 — YOLO")
print("█" * 55)

yaml_path  = preparer_dataset_yolo(df_train, df_test, df_valid)
model_path = entrainer_yolo(
    yaml_path,
    device=DEVICE,
    epochs=30,
    imgsz=640
)
if model_path:
    tester_yolo(model_path, df_test)

# ════════════════════════════════════════════════
# MODULE 3 — CLUSTERING
# ════════════════════════════════════════════════
print("\n\n" + "█" * 55)
print("  MODULE 3 — CLUSTERING")
print("█" * 55)

if not df_ocr.empty:
    X, meta = extraire_features(df_ocr)

    if len(X) >= 10:
        X_norm = normalize(X, norm='l2')
        k_opt  = choisir_k(X_norm, k_max=8)
        km, labels = entrainer_kmeans(X_norm, k_opt)
        interp = interpreter_et_visualiser(
            X_norm, labels, meta, k_opt
        )

print("\n\n" + "✅ " * 18)
print("  PIPELINE COMPLET TERMINÉ !")
print("  OCR      → data/processed/ocr_results/")
print("  YOLO     → outputs/yolo/platevision_yolo/")
print("  Clusters → data/processed/embeddings/")
print("  Figures  → outputs/figures/")
print("✅ " * 18)