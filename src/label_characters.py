# ── src/label_characters.py (version assouplie) ────────────────────────
import glob                                   # lister les fichiers
import os                                      # gérer dossiers
import cv2                                     # images
import easyocr                                 # OCR

CROPS_DIR = "data/processed/plate_crops"       # plaques découpées
OUT_DIR = "data/processed/characters_labeled"  # sortie : caractères étiquetés
MAX_PLATES = 3000                              # on augmente le nombre de plaques traitées
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" # caractères autorisés

os.makedirs(OUT_DIR, exist_ok=True)            # créer le dossier de sortie
reader = easyocr.Reader(['en'], gpu=False)     # OCR sur CPU

fichiers = glob.glob(f"{CROPS_DIR}/*.jpg") + glob.glob(f"{CROPS_DIR}/*.png")  # plaques
print(f"{len(fichiers)} plaques disponibles, on en traite {min(len(fichiers), MAX_PLATES)}")

nb_plaques_ok = 0                              # compteur plaques exploitées
nb_caracteres = 0                              # compteur caractères

for i, chemin in enumerate(fichiers[:MAX_PLATES]):
    if i % 200 == 0:                           # progression tous les 200
        print(f"  ... {i} plaques traitees")

    img = cv2.imread(chemin)                   # lire la plaque
    if img is None:
        continue

    textes = reader.readtext(img, detail=0, allowlist=ALLOW)  # lire le texte
    texte = "".join(textes).upper()            # majuscules
    texte = "".join(c for c in texte if c in ALLOW)  # garder A-Z, 0-9
    if len(texte) < 4 or len(texte) > 9:       # garder les longueurs plausibles de plaque
        continue

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gris
    # binarisation adaptative : plus robuste aux éclairages variables
    binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    H, W = gris.shape                          # hauteur, largeur de la plaque
    contours = cv2.findContours(binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    boites = []                                # boîtes candidates
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)       # rectangle de la forme
        # filtres plus tolérants : hauteur entre 35% et 95%, largeur raisonnable
        if 0.35 * H < h < 0.95 * H and 0.02 * W < w < 0.4 * W:
            boites.append((x, y, w, h))
    boites = sorted(boites, key=lambda b: b[0])  # gauche -> droite

    # APPARIEMENT TOLÉRANT : on prend le plus petit nombre commun.
    # Si on a découpé autant ou plus de boîtes que de lettres lues, on apparie
    # les premières boîtes aux premières lettres (au lieu de tout rejeter).
    n = min(len(boites), len(texte))           # nombre de paires sûres
    if n < 4:                                  # il faut au moins 4 caractères fiables
        continue

    for (x, y, w, h), lettre in zip(boites[:n], texte[:n]):  # apparier
        crop = gris[y:y+h, x:x+w]              # découper le caractère
        crop = cv2.resize(crop, (28, 28))      # 28x28
        dossier = os.path.join(OUT_DIR, lettre)  # dossier de la lettre
        os.makedirs(dossier, exist_ok=True)
        cv2.imwrite(os.path.join(dossier, f"{nb_caracteres}.png"), crop)  # sauver
        nb_caracteres += 1

    nb_plaques_ok += 1

print(f"\nTermine : {nb_plaques_ok} plaques exploitees, {nb_caracteres} caracteres etiquetes")
for lettre in sorted(os.listdir(OUT_DIR)):     # répartition par lettre
    n = len(os.listdir(os.path.join(OUT_DIR, lettre)))
    print(f"  {lettre} : {n}")