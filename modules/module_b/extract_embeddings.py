"""
Module B — Étape 1 : Extraction des embeddings CNN (CharEmbeddingCNN, couche FC1, 256D)
PlateVision / MINT-DGI Cameroun

Différence fondamentale avec le Module A :
  - Module A (Naïves Bayes) : features manuelles 120D définies par un expert
    humain (histogramme HSV, projections, densité...). Interprétables mais
    limitées par les choix de l'ingénieur.
  - Module B (K-Means + CNN) : embeddings 256D APPRIS par le réseau de neurones.
    Le CNN a optimisé ces représentations pour distinguer les 36 classes de
    caractères. Il capture des patterns visuels complexes non formulables
    manuellement. L'espace d'embedding est dense et continu — adapté au
    clustering géométrique par K-Means.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping label entier → caractère
# 0-9  → '0'-'9'   |   10-35 → 'A'-'Z'
def _int_to_char(i: int) -> str:
    return chr(ord("0") + i) if i < 10 else chr(ord("A") + i - 10)

INT_TO_CHAR: dict[int, str] = {i: _int_to_char(i) for i in range(36)}


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES PAR SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def load_split_data(
    data_dir: Path,
    splits: list[str] = ["train", "val", "test"],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Charge les images de caractères et leurs labels depuis data/processed/.

    Cherche pour chaque split :
      chars_{split}.npy          → images (N_i, 28, 28) float32 [0,1]
      chars_labels_{split}.npy   → labels entiers (N_i,)

    Les splits absents sont ignorés avec un warning.
    Retourne (X_all, y_all, split_tags) où split_tags trace l'origine de
    chaque image.
    """
    data_dir = Path(data_dir)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    tags: list[str] = []

    for split in splits:
        img_path   = data_dir / f"chars_{split}.npy"
        label_path = data_dir / f"chars_labels_{split}.npy"

        if not img_path.exists():
            logger.warning("Split '%s' introuvable : %s — ignoré.", split, img_path)
            continue
        if not label_path.exists():
            logger.warning("Labels '%s' introuvables : %s — ignoré.", split, label_path)
            continue

        X_i = np.load(img_path).astype(np.float32)
        y_i = np.load(label_path).astype(np.int64)
        X_parts.append(X_i)
        y_parts.append(y_i)
        tags.extend([split] * len(X_i))
        logger.info("Split '%s' chargé : %d images", split, len(X_i))

    if not X_parts:
        raise RuntimeError(
            "Données de caractères absentes. "
            "Exécute d'abord : python main.py --module A2"
        )

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    logger.info("Total chargé : %d images — shape %s", len(X_all), X_all.shape)
    return X_all, y_all, tags


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONSTRUCTION DES MÉTADONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def build_metadata(
    data_dir: Path,
    split_tags: list[str],
    y_labels: np.ndarray,
    ocr_results_path: "Path | None" = None,
) -> pd.DataFrame:
    """
    Construit un DataFrame de traçabilité associant chaque embedding à ses
    métadonnées : split d'origine, label numérique/caractère, info OCR.

    Colonnes produites :
      index, split, label_int, label_char, ocr_text, ocr_conf, conformite
    """
    n = len(y_labels)
    df = pd.DataFrame({
        "index":      np.arange(n, dtype=np.int32),
        "split":      split_tags,
        "label_int":  y_labels.astype(np.int32),
        "label_char": [INT_TO_CHAR.get(int(l), "?") for l in y_labels],
        "ocr_text":   [""] * n,
        "ocr_conf":   [float("nan")] * n,
        "conformite": ["unknown"] * n,
    })

    if ocr_results_path is not None:
        ocr_path = Path(ocr_results_path)
        if ocr_path.exists():
            try:
                ocr_df = pd.read_csv(ocr_path)
                # Join sur l'index si la colonne 'index' existe, sinon ignoré
                if "index" in ocr_df.columns:
                    ocr_df = ocr_df.set_index("index")
                    for col in ("ocr_text", "ocr_conf", "conformite"):
                        if col in ocr_df.columns:
                            df[col] = df["index"].map(ocr_df[col]).fillna(df[col])
                logger.info("Métadonnées OCR jointes depuis %s", ocr_path)
            except Exception as exc:
                logger.warning("Erreur lecture OCR CSV (%s) — ignoré.", exc)
        else:
            logger.warning("ocr_results_path fourni mais absent : %s", ocr_path)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXTRACTION DES EMBEDDINGS VIA HOOK FC1
# ══════════════════════════════════════════════════════════════════════════════

def extract_embeddings_from_model(
    model_path: Path,
    X: np.ndarray,
    batch_size: int = 256,
    device: str = "auto",
) -> np.ndarray:
    """
    Extrait les embeddings 256D de la couche FC1 du CharEmbeddingCNN.

    Utilise un forward hook sur model.fc1 pour capturer la sortie après
    ReLU (avant Dropout et avant fc2). Traite X par batches pour éviter
    les débordements mémoire GPU.

    Retourne un array (N, 256) float32.
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from modules.module_b.cnn_embeddings import CharEmbeddingCNN

    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(
            "CNN non entraîné. Exécute d'abord : python main.py --module B0"
        )

    # ── Sélection device ──────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info("Device : %s", dev)

    # ── Chargement modèle ─────────────────────────────────────────────────────
    state = torch.load(model_path, map_location=dev)
    model = CharEmbeddingCNN()
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(dev)
    model.eval()

    # ── Hook sur fc1 — capture sortie APRÈS ReLU ──────────────────────────────
    captured: list[np.ndarray] = []

    def _hook(module, _input, output):
        # Applique ReLU pour obtenir la sortie post-activation
        activated = F.relu(output).detach().cpu().numpy()
        captured.append(activated)

    handle = model.fc1.register_forward_hook(_hook)

    # ── Normalisation et reshape ───────────────────────────────────────────────
    X_proc = X.astype(np.float32)
    if X_proc.max() > 1.0:
        X_proc /= 255.0
    if X_proc.ndim == 3:                      # (N, 28, 28) → (N, 1, 28, 28)
        X_proc = X_proc[:, np.newaxis, :, :]

    # ── Extraction par batches ─────────────────────────────────────────────────
    n = len(X_proc)
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Extraction embeddings"):
            batch = torch.tensor(X_proc[start : start + batch_size], device=dev)
            model(batch)          # forward déclenche le hook

    handle.remove()

    embeddings = np.concatenate(captured, axis=0).astype(np.float32)
    logger.info("Embeddings extraits : shape %s", embeddings.shape)
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════

def save_embeddings_and_metadata(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: Path = Path("data/processed"),
) -> None:
    """
    Sauvegarde embeddings.npy et metadata.csv dans out_dir.
    Crée le répertoire si absent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path  = out_dir / "embeddings.npy"
    meta_path = out_dir / "metadata.csv"

    np.save(emb_path, embeddings)
    metadata.to_csv(meta_path, index=False, encoding="utf-8")

    logger.info("✓ embeddings.npy sauvegardé : shape=%s", embeddings.shape)
    logger.info("✓ metadata.csv sauvegardé  : %d lignes", len(metadata))
    print(f"✓ embeddings.npy sauvegardé : shape={embeddings.shape}")
    print(f"✓ metadata.csv sauvegardé  : {len(metadata)} lignes")


# ══════════════════════════════════════════════════════════════════════════════
# 5. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_embeddings(embeddings: np.ndarray) -> None:
    """
    Vérifie la cohérence des embeddings : absence de NaN, de vecteurs nuls.
    Affiche les statistiques globales (min, max, mean, std).
    """
    if np.isnan(embeddings).any():
        raise ValueError(
            "Embeddings contiennent des NaN — vérifier le CNN"
        )

    norms = np.linalg.norm(embeddings, axis=1)
    zero_count = int((norms == 0).sum())
    zero_ratio = zero_count / len(embeddings)
    if zero_ratio > 0.05:
        logger.warning(
            "%.1f%% des embeddings sont des vecteurs nuls (%d/%d).",
            zero_ratio * 100, zero_count, len(embeddings),
        )

    print(
        f"  Validation embeddings — min={embeddings.min():.4f}  "
        f"max={embeddings.max():.4f}  "
        f"mean={embeddings.mean():.4f}  "
        f"std={embeddings.std():.4f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_embedding_pipeline(
    data_dir: Path = Path("data/processed"),
    model_path: Path = Path("models/char_cnn.pth"),
    out_dir: Path = Path("data/processed"),
    splits: list[str] = ["train", "val", "test"],
    ocr_results_path: "Path | None" = None,
    batch_size: int = 256,
    device: str = "auto",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Orchestre l'extraction complète des embeddings CNN pour le Module B.

    Différence fondamentale avec le Module A :
      - Module A (Naïves Bayes) : features manuelles 120D définies par un
        expert humain (histogramme HSV, projections, densité...).
        Interprétables mais limitées par les choix de l'ingénieur.
      - Module B (K-Means + CNN) : embeddings 256D APPRIS par le réseau de
        neurones. Le CNN a optimisé ces représentations pour distinguer les
        36 classes de caractères. Il capture des patterns visuels complexes
        non formulables manuellement. L'espace d'embedding est dense et
        continu — adapté au clustering géométrique par K-Means.

    Étapes :
      1. load_split_data()
      2. build_metadata()
      3. extract_embeddings_from_model()
      4. validate_embeddings()
      5. save_embeddings_and_metadata()

    Retourne (embeddings, metadata).
    """
    # ── Vérifications préalables ──────────────────────────────────────────────
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(
            "CNN non entraîné. Exécute d'abord : python main.py --module B0"
        )

    data_dir = Path(data_dir)
    char_files = list(data_dir.glob("chars_*.npy"))
    if not char_files:
        raise RuntimeError(
            "Données de caractères absentes. "
            "Exécute d'abord : python main.py --module A2"
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    X, y, split_tags = load_split_data(data_dir, splits)
    metadata         = build_metadata(data_dir, split_tags, y, ocr_results_path)
    embeddings       = extract_embeddings_from_model(model_path, X, batch_size, device)
    validate_embeddings(embeddings)
    save_embeddings_and_metadata(embeddings, metadata, Path(out_dir))

    print("\n=== Module B — Étape 1 terminée ===")
    print(f"Embeddings : {embeddings.shape}")
    print(f"Métadonnées : {len(metadata)} lignes")
    print("Prêt pour le clustering K-Means (Étape 2)")

    return embeddings, metadata


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module B Étape 1 — Extraction embeddings CNN"
    )
    parser.add_argument("--data-dir",    type=Path, default=Path("data/processed"))
    parser.add_argument("--model-path",  type=Path, default=Path("models/char_cnn.pth"))
    parser.add_argument("--out-dir",     type=Path, default=Path("data/processed"))
    parser.add_argument("--splits",      nargs="+", default=["train", "val", "test"])
    parser.add_argument("--ocr-results", type=Path, default=None)
    parser.add_argument("--batch-size",  type=int,  default=256)
    parser.add_argument("--device",      type=str,  default="auto")
    args = parser.parse_args()

    run_embedding_pipeline(
        data_dir         = args.data_dir,
        model_path       = args.model_path,
        out_dir          = args.out_dir,
        splits           = args.splits,
        ocr_results_path = args.ocr_results,
        batch_size       = args.batch_size,
        device           = args.device,
    )
