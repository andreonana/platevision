"""
Module B — Étape B0 : CNN pour embeddings de caractères (CharEmbeddingCNN)
PlateVision / MINT-DGI Cameroun

Architecture CNN 28×28 → FC1 256D → 36 classes (0-9, A-Z).
La couche FC1 (self.fc1) sert d'embedding layer : extract_embeddings.py
y accroche un forward hook pour extraire les vecteurs 256D.
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Mapping label entier → caractère (0-9 → '0'-'9', 10-35 → 'A'-'Z')
INT_TO_CHAR: dict[int, str] = {
    **{i: str(i) for i in range(10)},
    **{10 + i: chr(ord("A") + i) for i in range(26)},
}
CHAR_TO_INT: dict[str, int] = {v: k for k, v in INT_TO_CHAR.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ARCHITECTURE CNN
# ══════════════════════════════════════════════════════════════════════════════

class CharEmbeddingCNN(nn.Module):
    """
    CNN léger pour classification de caractères alphanumériques 28×28 px.

    Blocs convolutifs :
      conv1 : 1→32  ch, 3×3 pad=1 → BN → ReLU → MaxPool(2)  → (32,14,14)
      conv2 : 32→64 ch, 3×3 pad=1 → BN → ReLU → MaxPool(2)  → (64, 7, 7)
      conv3 : 64→128ch, 3×3 pad=1 → BN → ReLU → MaxPool(2)  → (128,3,3)

    Têtes :
      Flatten → 1152
      fc1  : Linear(1152, 256)  — embedding layer (hook ici)
      drop : Dropout(0.3)
      fc2  : Linear(256, 36)    — logits, pas de softmax
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 128 × 3 × 3 = 1152
        # fc1 = couche embedding : le hook extract_embeddings.py s'accroche ici pour extraire les 256D
        self.fc1  = nn.Linear(1152, 256)
        # Dropout(0.3) uniquement pendant l'entraînement — désactivé en mode eval (extraction d'embeddings)
        self.drop = nn.Dropout(0.3)
        self.fc2  = nn.Linear(256, 36)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)          # (N, 1152)
        embedding = F.relu(self.fc1(x))     # (N, 256) — embedding 256D
        x = self.drop(embedding)
        x = self.fc2(x)                     # (N, 36)  — logits
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def load_char_data(
    data_dir: Path = Path("data/processed"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge les données de caractères en priorité depuis les fichiers .npy,
    sinon depuis data/processed/characters/{classe}/*.png.

    Retourne (X_train, y_train, X_val, y_val, X_test, y_test)
    X : float32 [0,1], shape (N, 28, 28).
    y : int32, labels 0-35.
    """
    import cv2
    from sklearn.model_selection import train_test_split

    data_dir = Path(data_dir)

    # ── Priorité 1 : fichiers .npy ────────────────────────────────────────────
    npy_paths = {
        "train": (data_dir / "chars_train.npy",        data_dir / "chars_labels_train.npy"),
        "val":   (data_dir / "chars_val.npy",          data_dir / "chars_labels_val.npy"),
        "test":  (data_dir / "chars_test.npy",         data_dir / "chars_labels_test.npy"),
    }
    if all(p.exists() for pair in npy_paths.values() for p in pair):
        logger.info("Chargement depuis fichiers .npy")
        splits = {}
        for split, (xp, yp) in npy_paths.items():
            X = np.load(xp).astype(np.float32)
            if X.max() > 1.0:
                X /= 255.0
            splits[split] = (X, np.load(yp).astype(np.int32))
        _log_split_info(splits["train"][1], splits["val"][1], splits["test"][1])
        return (*splits["train"], *splits["val"], *splits["test"])

    # ── Priorité 2 : dossier characters/ ─────────────────────────────────────
    chars_dir = data_dir / "characters"
    if not chars_dir.exists():
        raise RuntimeError(
            f"Aucune donnée trouvée dans {data_dir}. "
            "Lancez d'abord : python data/prepare_datasets.py"
        )

    logger.info("Chargement depuis %s", chars_dir)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for cls_name in sorted(cls.name for cls in chars_dir.iterdir() if cls.is_dir()):
        if cls_name not in CHAR_TO_INT:
            continue
        label = CHAR_TO_INT[cls_name]
        cls_dir = chars_dir / cls_name
        for img_file in sorted(cls_dir.iterdir()):
            if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(label)

    if not images:
        raise RuntimeError(f"Aucune image valide trouvée dans {chars_dir}")

    X_all = np.stack(images)           # (N, 28, 28)
    y_all = np.array(labels, dtype=np.int32)
    logger.info("Total images chargées : %d sur %d classes", len(X_all), len(np.unique(y_all)))

    # Split 80 / 10 / 10
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=0.10, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.111, random_state=42, stratify=y_tmp  # 0.111 ≈ 10% du total
    )

    _log_split_info(y_train, y_val, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _log_split_info(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> None:
    logger.info(
        "Split — train:%d  val:%d  test:%d  | classes:%d",
        len(y_train), len(y_val), len(y_test),
        len(np.unique(np.concatenate([y_train, y_val, y_test]))),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    # (N, 28, 28) → (N, 1, 28, 28)
    X_t = torch.tensor(X[:, np.newaxis, :, :], dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    model_path: Path = Path("models/char_cnn.pth"),
    device: str = "auto",
) -> "CharEmbeddingCNN":
    """
    Entraîne CharEmbeddingCNN et sauvegarde le meilleur modèle (val accuracy).
    Retourne le modèle avec les meilleurs poids chargés.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info("Entraînement sur device : %s", dev)

    model = CharEmbeddingCNN().to(dev)
    # Adam(lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau(mode=max, patience=5, factor=0.5) — divisé par 2 si val_acc stagne 5 epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   batch_size, shuffle=False)

    best_acc   = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                logits = model(xb)
                val_loss    += criterion(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(X_val)
        val_acc   = val_correct / len(X_val) * 100

        scheduler.step(val_acc)

        tqdm.write(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%"
            + (" ✓ best" if val_acc > best_acc else "")
        )

        if val_acc > best_acc:
            best_acc   = val_acc
            best_epoch = epoch
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "best_val_acc": best_acc,
                 "epoch": epoch},
                model_path,
            )

    logger.info("Meilleur modèle : epoch %d, val_acc=%.2f%%", best_epoch, best_acc)
    logger.info("Poids sauvegardés → %s", model_path)

    # Recharge les meilleurs poids
    ckpt = torch.load(model_path, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_cnn(
    model: "CharEmbeddingCNN",
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "auto",
    figures_dir: Path = Path("reports/rapport_technique/figures"),
) -> dict:
    """
    Évalue le modèle sur X_test.
    Sauvegarde la matrice de confusion dans figures_dir/cnn_confusion_matrix.png.
    Retourne {"accuracy": float, "classification_report": str, "confusion_matrix": ndarray}.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model.eval()
    model.to(dev)

    X_t = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)
    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            batch = X_t[i : i + 256].to(dev)
            preds.extend(model(batch).argmax(1).cpu().tolist())

    y_pred = np.array(preds, dtype=np.int32)
    acc    = accuracy_score(y_test, y_pred)

    target_names = [INT_TO_CHAR[i] for i in range(36)]
    report = classification_report(
        y_test, y_pred,
        labels=list(range(36)),
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(36)))

    logger.info("Test accuracy : %.2f%%", acc * 100)

    # ── Matrice de confusion ───────────────────────────────────────────────
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(36), yticks=range(36),
        xticklabels=target_names, yticklabels=target_names,
        xlabel="Prédit", ylabel="Réel",
        title=f"Matrice de confusion — CharEmbeddingCNN (acc={acc*100:.1f}%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()
    cm_path = figures_dir / "cnn_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Matrice de confusion → %s", cm_path)

    return {"accuracy": float(acc), "classification_report": report, "confusion_matrix": cm}


# ══════════════════════════════════════════════════════════════════════════════
# 5. EXTRACTION D'EMBEDDINGS (version standalone)
# ══════════════════════════════════════════════════════════════════════════════

def extract_embeddings(
    model: "CharEmbeddingCNN",
    X: np.ndarray,
    batch_size: int = 256,
    device: str = "auto",
) -> np.ndarray:
    """
    Extrait les embeddings 256D via hook sur model.fc1.
    X : (N, 28, 28) float32 [0,1].
    Retourne (N, 256) float32.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model.eval()
    model.to(dev)

    captured: list[np.ndarray] = []

    def _hook(_module, _input, output):
        captured.append(F.relu(output).detach().cpu().numpy())

    handle = model.fc1.register_forward_hook(_hook)

    X_t = torch.tensor(X[:, np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(X_t), batch_size), desc="Extraction embeddings", leave=False):
            model(X_t[i : i + batch_size].to(dev))

    handle.remove()

    embeddings = np.concatenate(captured, axis=0).astype(np.float32)
    logger.info("Embeddings extraits : %s", embeddings.shape)
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════

def save_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_dir: Path = Path("data/processed"),
    split_tags: "list[str] | None" = None,
) -> None:
    """
    Sauvegarde embeddings.npy, embeddings_labels.npy et metadata.csv.
    split_tags : liste de longueur N avec 'train'/'val'/'test' par sample.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path   = out_dir / "embeddings.npy"
    label_path = out_dir / "embeddings_labels.npy"
    meta_path  = out_dir / "metadata.csv"

    np.save(emb_path,   embeddings)
    np.save(label_path, labels)

    label_names = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]
    meta = pd.DataFrame({
        "label_int":  labels.astype(np.int32),
        "label_char": [label_names[i] if i < len(label_names) else "?" for i in labels],
        "split":      split_tags if split_tags is not None else ["unknown"] * len(labels),
    })
    meta.to_csv(meta_path, index=False, encoding="utf-8")

    logger.info("embeddings.npy sauvegardé : shape=%s → %s", embeddings.shape, emb_path)
    logger.info("embeddings_labels.npy    : shape=%s → %s", labels.shape, label_path)
    logger.info("metadata.csv sauvegardé  : %d lignes → %s", len(meta), meta_path)


# ══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_cnn_embedding_pipeline(
    data_dir: Path = Path("data/processed"),
    model_path: Path = Path("models/char_cnn.pth"),
    out_dir: Path = Path("data/processed"),
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    force_retrain: bool = False,
    device: str = "auto",
) -> np.ndarray:
    """
    Pipeline B0 complet :
      1. Charge les données depuis data_dir
      2. Entraîne le CNN (ou charge si déjà entraîné)
      3. Évalue sur le jeu de test
      4. Extrait les embeddings sur train+val+test
      5. Sauvegarde embeddings.npy + embeddings_labels.npy
    Retourne les embeddings (N, 256).
    """
    data_dir   = Path(data_dir)
    model_path = Path(model_path)
    out_dir    = Path(out_dir)

    if device == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_str = device
    dev = torch.device(dev_str)

    # ── 1. Chargement des données ─────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = load_char_data(data_dir)

    # ── 2. Modèle ─────────────────────────────────────────────────────────────
    if model_path.exists() and not force_retrain:
        logger.info("Modèle existant chargé — skip entraînement")
        ckpt  = torch.load(model_path, map_location=dev)
        model = CharEmbeddingCNN().to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        best_acc = float(ckpt.get("best_val_acc", 0.0))
    else:
        model    = train_cnn(X_train, y_train, X_val, y_val,
                             epochs=epochs, batch_size=batch_size, lr=lr,
                             model_path=model_path, device=dev_str)
        best_acc = float(torch.load(model_path, map_location="cpu")["best_val_acc"])

    # ── 3. Évaluation ─────────────────────────────────────────────────────────
    metrics = evaluate_cnn(model, X_test, y_test, device=dev_str)
    print(metrics["classification_report"])

    # ── 4. Extraction sur tout le dataset ─────────────────────────────────────
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    split_tags = (
        ["train"] * len(X_train)
        + ["val"]   * len(X_val)
        + ["test"]  * len(X_test)
    )
    embeddings = extract_embeddings(model, X_all, batch_size=batch_size, device=dev_str)

    # ── 5. Sauvegarde ─────────────────────────────────────────────────────────
    save_embeddings(embeddings, y_all, out_dir, split_tags=split_tags)

    print("\n=== Module B — Étape B0 terminée ===")
    print(f"Modèle : {model_path}")
    print(f"Val accuracy : {best_acc:.2f}%")
    print(f"Embeddings : {embeddings.shape} → {out_dir}/embeddings.npy")
    print("Prêt pour : python -m modules.module_b.extract_embeddings")

    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module B Étape B0 — Entraînement CNN CharEmbeddingCNN"
    )
    parser.add_argument("--data-dir",      type=Path,  default=Path("data/processed"))
    parser.add_argument("--model-path",    type=Path,  default=Path("models/char_cnn.pth"))
    parser.add_argument("--out-dir",       type=Path,  default=Path("data/processed"))
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--device",        type=str,   default="auto")
    args = parser.parse_args()

    run_cnn_embedding_pipeline(
        data_dir      = args.data_dir,
        model_path    = args.model_path,
        out_dir       = args.out_dir,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        force_retrain = args.force_retrain,
        device        = args.device,
    )
