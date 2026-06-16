"""
Module B — Étape 1 (révisée) : Features de qualité/lisibilité par PLAQUE
PlateVision / MINT-DGI Cameroun

Pourquoi cette refonte (§4.2 cahier des charges) :
  La version précédente clusterisait des CARACTÈRES individuels (28×28, embeddings
  CNN 256D issus de CharEmbeddingCNN) puis renommait les clusters "conforme /
  illisible / expirée" par simple convention d'index — sans aucune validation que
  ce regroupement de FORMES DE CARACTÈRES corresponde à un quelconque statut de
  plaque. La seule métrique de validation disponible (ARI/NMI vs label_char)
  mesure si le clustering retrouve l'identité du caractère, pas la conformité.

  Cette version clusterise directement au niveau PLAQUE, sur des features
  objectivement mesurables dans l'image : netteté, contraste, complétude de la
  segmentation, confiance OCR. Ces features captent réellement une notion de
  QUALITÉ/LISIBILITÉ visuelle — ce que la vision peut honnêtement évaluer.
  Un statut légal (plaque expirée, falsifiée) reste un fait administratif que
  la vision seule ne peut pas valider : Module C continue donc à raisonner sur
  des États (cluster_qualité × confiance × alerte), mais la procédure attachée
  au cluster le plus dégradé est désormais formulée comme une escalade pour
  vérification humaine, pas comme une détection de fraude.

Sources réutilisées (déjà calculées par data/prepare_datasets.py, Phases 3-4) :
  - data/processed/plate_crops/roboflow_*.jpg : crops 200×60 déjà recadrés sur
    la plaque via les annotations réelles Roboflow.
  - data/processed/ocr_results.json : confiance/texte OCR par crop.
  Mendeley est exclu (cf. fix YOLO précédent : pas d'annotation réelle, ses
  "crops" sont des scènes entières squashées, donc des features de qualité
  calculées dessus seraient elles aussi non significatives).
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.prepare_datasets import _segment_chars_from_plate

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["blur_score", "contrast", "n_chars_detected", "ocr_confidence"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES CROPS ROBOFLOW + OCR
# ══════════════════════════════════════════════════════════════════════════════

def load_roboflow_ocr_results(data_dir: Path = Path("data/processed")) -> pd.DataFrame:
    """
    Charge ocr_results.json et filtre sur source == 'roboflow' — seule source
    avec un recadrage réel de la plaque (annotation Roboflow), donc la seule
    où des features de netteté/contraste/segmentation ont un sens.
    """
    data_dir = Path(data_dir)
    ocr_path = data_dir / "ocr_results.json"
    if not ocr_path.exists():
        raise RuntimeError(
            f"{ocr_path} introuvable. Exécute d'abord : "
            "python data/prepare_datasets.py --from-phase 3"
        )

    with open(ocr_path, encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df = df[df["source"] == "roboflow"].reset_index(drop=True)
    logger.info("OCR Roboflow chargé : %d plaques", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURES DE QUALITÉ PAR PLAQUE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_quality_features(img: np.ndarray) -> tuple[float, float, int]:
    """
    Calcule (blur_score, contrast, n_chars_detected) pour un crop de plaque.

    blur_score : variance du Laplacien — plus haut = plus net (moins flou).
    contrast   : écart-type des niveaux de gris — plus haut = mieux contrasté
                 (plaque propre/éclairée vs plaque délavée/sale).
    n_chars_detected : nombre de caractères segmentables (cf. Module A1) —
                 une plaque très dégradée ou occultée en segmente peu/aucun.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast   = float(gray.std())
    chars      = _segment_chars_from_plate(img)
    return blur_score, contrast, len(chars)


def extract_plate_quality_features(
    data_dir: Path = Path("data/processed"),
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Construit la matrice de features qualité (N, 4) + le DataFrame de
    traçabilité associé, pour toutes les plaques Roboflow.

    Colonnes embeddings (dans cet ordre, voir FEATURE_NAMES) :
      blur_score, contrast, n_chars_detected, ocr_confidence
    """
    data_dir = Path(data_dir)
    ocr_df   = load_roboflow_ocr_results(data_dir)

    rows: list[dict] = []
    feats: list[list[float]] = []

    for _, rec in tqdm(ocr_df.iterrows(), total=len(ocr_df), desc="Features qualité plaque"):
        crop_path = Path(rec["crop_path"])
        img = cv2.imread(str(crop_path))
        if img is None:
            continue

        blur_score, contrast, n_chars = _compute_quality_features(img)
        ocr_conf = float(rec.get("ocr_confidence") or 0.0)

        feats.append([blur_score, contrast, float(n_chars), ocr_conf])
        rows.append({
            "crop_path":        str(crop_path),
            "source":           rec.get("source"),
            "ocr_text":         rec.get("plate_text") or "",
            "ocr_confidence":   ocr_conf,
            "blur_score":       blur_score,
            "contrast":         contrast,
            "n_chars_detected": n_chars,
        })

    embeddings = np.asarray(feats, dtype=np.float32)
    metadata   = pd.DataFrame(rows)

    logger.info(
        "Features qualité extraites : %d plaques × %dD (%s)",
        embeddings.shape[0], embeddings.shape[1], ", ".join(FEATURE_NAMES),
    )
    return embeddings, metadata


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAUVEGARDE — même contrat d'E/S que l'ancien extract_embeddings.py
# ══════════════════════════════════════════════════════════════════════════════

def save_embeddings_and_metadata(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: Path = Path("data/processed"),
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", embeddings)
    metadata.to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8")

    logger.info("✓ embeddings.npy sauvegardé : shape=%s", embeddings.shape)
    logger.info("✓ metadata.csv sauvegardé  : %d lignes", len(metadata))
    print(f"✓ embeddings.npy sauvegardé : shape={embeddings.shape}")
    print(f"✓ metadata.csv sauvegardé  : {len(metadata)} lignes")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE COMPLET (remplace run_embedding_pipeline — Étape B1)
# ══════════════════════════════════════════════════════════════════════════════

def run_plate_quality_pipeline(data_dir: Path = Path("data/processed")) -> None:
    print("\n" + "═" * 60)
    print("  PlateVision — Module B Étape 1 : features qualité plaque")
    print("═" * 60)

    embeddings, metadata = extract_plate_quality_features(data_dir)
    save_embeddings_and_metadata(embeddings, metadata, data_dir)

    print("\n[RÉSUMÉ]")
    print(f"  Plaques traitées : {len(metadata)}")
    print(f"  Features         : {', '.join(FEATURE_NAMES)}")
    print(f"  → data/processed/embeddings.npy + metadata.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_plate_quality_pipeline()
