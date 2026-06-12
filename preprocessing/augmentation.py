"""
Augmentation de données — PlateVision / MINT-DGI Cameroun
Simule les conditions réelles des postes de contrôle routier camerounais.

Conditions simulées :
  - Éclairage tropical (surexposition / sous-exposition)
  - Flou de mouvement (véhicules en déplacement lent)
  - Dégradation physique (boue, rayures, occultation partielle)
  - Angles de caméra obliques (caméras fixes en hauteur)
  - Variations jour / nuit

Justification des choix :
  Les datasets disponibles (Roboflow Nigeria, Mendeley Ghana) ne contiennent
  aucune plaque camerounaise. L'augmentation compense ce biais de domaine en
  reproduisant les conditions documentées dans le diagnostic DGI 2022 :
  éclairage équatorial extrême, routes non goudronnées (boue), caméras fixes
  sur poteaux à ~4m de hauteur, et présence fréquente de barrages nocturnes.
"""

import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS UNITAIRES — CARACTÈRES 28×28
# ══════════════════════════════════════════════════════════════════════════════

def rotate_char(img: np.ndarray, angle_range: float = 8.0) -> np.ndarray:
    """
    Rotation aléatoire ±8°.

    POURQUOI : Les plaques camerounaises sont souvent vissées de travers ou
    abîmées, et l'étape de segmentation des caractères introduit elle-même
    une légère inclinaison résiduelle (±3–8° observés sur le terrain DGI).
    INTER_NEAREST préserve les bords nets des caractères (pas de flou induit).
    """
    angle = random.uniform(-angle_range, angle_range)
    M = cv2.getRotationMatrix2D((13.5, 13.5), angle, 1.0)
    return cv2.warpAffine(img, M, (28, 28),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REPLICATE)


def add_gaussian_noise(img: np.ndarray,
                       sigma_range: tuple = (1, 5)) -> np.ndarray:
    """
    Bruit gaussien σ ∈ sigma_range.

    POURQUOI : Les caméras IP bas de gamme déployées aux postes de contrôle
    MINT produisent un bruit de capteur (grain numérique) mesurable surtout
    la nuit ou par temps couvert. σ ∈ [1, 5] couvre la plage observée lors
    des tests en conditions réelles (rapport PSTNAC phase 1, 2023).
    """
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def translate_char(img: np.ndarray, max_shift: int = 2) -> np.ndarray:
    """
    Translation aléatoire ±2 px en x et y.

    POURQUOI : La segmentation des caractères (Phase 5) introduit une
    incertitude de localisation de ±1–2 px selon le contraste de la plaque.
    Cette transform force le modèle à être robuste au centrage imparfait,
    condition courante sur plaques camerounaises à fond jaune délavé.
    """
    tx = random.uniform(-max_shift, max_shift)
    ty = random.uniform(-max_shift, max_shift)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (28, 28), borderMode=cv2.BORDER_REPLICATE)


def erode_or_dilate(img: np.ndarray) -> np.ndarray:
    """
    Érosion ou dilatation morphologique 2×2 (50/50), 1 itération.

    POURQUOI :
      - Érosion  → simule l'effacement progressif de la peinture des
        caractères (usure UV tropicale, frottement de branchages sur routes
        forestières camerounaises).
      - Dilatation → simule l'épaississement par accumulation de boue
        latéritique rouge sur les caractères (saison des pluies, routes
        non bitumées vers postes intérieurs MINT).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    if random.random() < 0.5:
        return cv2.erode(img, kernel, iterations=1)
    return cv2.dilate(img, kernel, iterations=1)


def affine_distort_char(img: np.ndarray,
                         max_distort: float = 1.5) -> np.ndarray:
    """
    Distorsion affine légère via déplacement aléatoire de 3 coins (±1.5 px).

    POURQUOI : Les caméras MINT sont installées en hauteur (~4 m) sur des
    poteaux, regardant vers le bas-avant à 30–45°. Cette perspective oblique
    déforme légèrement les caractères de façon affine (parallélogramme au lieu
    de rectangle), ce que cette transform reproduit fidèlement.
    """
    src = np.float32([[0, 0], [27, 0], [0, 27]])
    def _j():
        return random.uniform(-max_distort, max_distort)
    dst = np.float32([
        [_j(), _j()],
        [27 + _j(), _j()],
        [_j(), 27 + _j()],
    ])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (28, 28), borderMode=cv2.BORDER_REPLICATE)


def add_char_occlusion(img: np.ndarray,
                        max_cover: float = 0.25) -> np.ndarray:
    """
    Rectangle d'occlusion aléatoire ≤ 25 % de la surface, rempli [0, 128].

    POURQUOI : Sur les plaques camerounaises, des adhésifs publicitaires,
    des taches de boue concentrées ou des chocs physiques (grêle, cailloux)
    masquent partiellement 1 à 2 caractères. La valeur [0, 128] représente
    les occlusions sombres (boue, ombre portée), les plus fréquentes.
    """
    out = img.copy()
    H, W = out.shape
    max_pixels = int(H * W * max_cover)  # ≤ 196 px sur 28×28
    w = random.randint(1, W)
    h = random.randint(1, max(1, max_pixels // w))
    x = random.randint(0, max(0, W - w))
    y = random.randint(0, max(0, H - h))
    out[y:y + h, x:x + w] = random.randint(0, 128)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS UNITAIRES — CROPS DE PLAQUES 200×60
# ══════════════════════════════════════════════════════════════════════════════

def adjust_brightness(img: np.ndarray,
                      factor_range: tuple = (0.5, 1.6)) -> np.ndarray:
    """
    Facteur multiplicatif sur canal V (HSV) ∈ [0.5, 1.6].

    POURQUOI : Plage élargie volontairement pour couvrir les deux extrêmes
    rencontrés aux postes camerounais :
      → factor > 1.3 : surexposition solaire tropicale (midi équatorial,
        soleil zénithal, pas d'ombre portée sur la plaque).
      → factor < 0.7 : barrages nocturnes avec éclairage de fortune
        (lampe torche, phare d'un seul côté, zone sans électricité).
    """
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(img: np.ndarray,
                    factor_range: tuple = (0.7, 1.4)) -> np.ndarray:
    """
    Contraste sur canal V : V_new = clip(f × (V − 128) + 128, 0, 255).

    POURQUOI : Simule la disparité de qualité optique entre les caméras des
    postes MINT (modèles hétérogènes, objectifs encrassés, dômes vieillis).
    Un faible contraste (f < 0.9) imite une caméra dont l'objectif est voilé
    par la poussière rouge latéritique, problème documenté au Grand-Nord.
    """
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(factor * (hsv[:, :, 2] - 128.0) + 128.0, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def simulate_motion_blur(img: np.ndarray,
                          kernel_range: tuple = (3, 9),
                          direction: str = "auto") -> np.ndarray:
    """
    Flou de mouvement horizontal ou diagonal.

    POURQUOI : Aux barrages MINT, les véhicules ralentissent mais ne
    s'immobilisent pas toujours complètement avant que la caméra capture
    l'image (diagnostic DGI 2022 : ~18 % des captures présentent un flou
    de mouvement mesurable). Le flou horizontal simule le passage en ligne
    droite ; le flou diagonal simule un véhicule oblique dans le champ.

    Implémentation :
      - k impair tiré dans kernel_range
      - horizontal : kernel[k//2, :] = 1/k
      - diagonal   : np.eye(k) / k
    """
    lo, hi = kernel_range
    # Sélection d'une valeur impaire dans l'intervalle
    odd_vals = list(range(lo if lo % 2 == 1 else lo + 1, hi + 1, 2))
    k = random.choice(odd_vals) if odd_vals else (lo if lo % 2 == 1 else lo + 1)

    if direction == "auto":
        direction = random.choice(["horizontal", "diagonal"])

    if direction == "horizontal":
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
    else:  # diagonal
        kernel = np.eye(k, dtype=np.float32) / k

    return cv2.filter2D(img, -1, kernel)


def simulate_night_conditions(img: np.ndarray) -> np.ndarray:
    """
    Simule les conditions nocturnes aux barrages MINT/DGI.

    Étapes :
      1. V × [0.3, 0.6]   — obscurité générale (absence d'éclairage public)
      2. S × [0.6, 0.9]   — désaturation typique d'un capteur CCD en IR
      3. Bruit σ ∈ [5, 15] — grain de caméra infrarouge en basse lumière

    POURQUOI : Les postes intérieurs (barrières forestières, axe Yaoundé-
    Bertoua, routes vers l'Adamaoua) fonctionnent souvent sans groupe
    électrogène fiable. Les images sont sous-exposées, bruitées, et les
    couleurs partiellement délavées par le mode gain-auto de la caméra.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.3, 0.6), 0, 255)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.6, 0.9), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    sigma = random.uniform(5.0, 15.0)
    noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def simulate_plate_degradation(img: np.ndarray) -> np.ndarray:
    """
    Simule la dégradation physique des plaques camerounaises.

    1–2 effets choisis aléatoirement parmi :
      a) BOUE (taches latéritiques) : 2–5 ellipses brun-kaki, axes 3–15 px
         POURQUOI : boue rouge latéritique projetée depuis routes en terre
         (Adamaoua, Extrême-Nord, Est). Couleur BGR ≈ (30–60, 50–110, 60–120).
      b) RAYURES HORIZONTALES : 1–3 lignes grises fines
         POURQUOI : frottement de broussailles lors de dépassements sur routes
         étroites ; les griffures horizontales sont les plus fréquentes (source
         terrain DGI, 2023).
      c) OCCULTATION BORD : bande noire ≤ 15 % largeur sur bord gauche ou droit
         POURQUOI : support de plaque décalé, fixation cassée, ou plaque repliée
         par un choc léger masquant 1 caractère extrême.
      d) DÉCOLORATION CENTRALE : S × [0.3, 0.7] sur zone rectangulaire centrale
         POURQUOI : chaleur tropicale + UV dégradent le fond réfléchissant de la
         plaque en son centre, réduisant la saturation de couleur (effet constaté
         sur plaques de + de 5 ans au Cameroun).
    """
    out = img.copy()
    H, W = out.shape[:2]

    effects = random.sample(["mud", "scratches", "edge", "decolor"],
                             k=random.randint(1, 2))

    for effect in effects:
        if effect == "mud":
            for _ in range(random.randint(2, 5)):
                cx = random.randint(0, W - 1)
                cy = random.randint(0, H - 1)
                ax = random.randint(3, 15)
                ay = random.randint(3, 15)
                color = (
                    random.randint(20, 60),   # B — teinte marron
                    random.randint(50, 110),  # G
                    random.randint(60, 120),  # R — latérite rouge-orangée
                )
                cv2.ellipse(out, (cx, cy), (ax, ay), 0, 0, 360, color, -1)

        elif effect == "scratches":
            for _ in range(random.randint(1, 3)):
                y_line = random.randint(0, H - 1)
                v = random.randint(100, 200)
                cv2.line(out, (0, y_line), (W - 1, y_line), (v, v, v), 1)

        elif effect == "edge":
            max_band = max(1, W * 15 // 100)
            band_w = random.randint(1, max_band)
            if random.random() < 0.5:
                out[:, :band_w] = 0
            else:
                out[:, W - band_w:] = 0

        elif effect == "decolor":
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            cy, cx = H // 2, W // 2
            zh = random.randint(H // 4, 3 * H // 4)
            zw = random.randint(W // 4, 3 * W // 4)
            y0, y1 = max(0, cy - zh // 2), min(H, cy + zh // 2)
            x0, x1 = max(0, cx - zw // 2), min(W, cx + zw // 2)
            hsv[y0:y1, x0:x1, 1] = np.clip(
                hsv[y0:y1, x0:x1, 1] * random.uniform(0.3, 0.7), 0, 255
            )
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out


def rotate_plate(img: np.ndarray, angle_range: float = 4.0) -> np.ndarray:
    """
    Rotation légère ±4°.

    POURQUOI : Deux sources d'inclinaison sur le terrain :
      1. Le véhicule n'est pas parfaitement parallèle à la caméra
         (files d'attente obliques aux barrières MINT).
      2. L'installation de la caméra elle-même peut être légèrement désaxée
         (maintenance irrégulière des postes isolés).
    """
    angle = random.uniform(-angle_range, angle_range)
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)


def perspective_plate(img: np.ndarray,
                       max_shift: float = 0.04) -> np.ndarray:
    """
    Transformation perspective : coins décalés de ±4 % des dimensions.

    POURQUOI : Configuration standard des postes MINT — caméra sur poteau
    à ~4 m de hauteur regardant vers le bas-avant à 30–45°. La plaque, vue
    sous cet angle, présente une déformation trapézoïdale réelle. Cette
    transform l'apprend au modèle sans nécessiter de vraies images en
    perspective (dont nous ne disposons pas dans les datasets).
    cv2.getPerspectiveTransform → warpPerspective, border REPLICATE.
    """
    H, W = img.shape[:2]

    def _sw():
        return random.uniform(-max_shift * W, max_shift * W)
    def _sh():
        return random.uniform(-max_shift * H, max_shift * H)

    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([
        [_sw(), _sh()],
        [W + _sw(), _sh()],
        [W + _sw(), H + _sh()],
        [_sw(), H + _sh()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)


def flip_plate(img: np.ndarray, probability: float = 0.3) -> np.ndarray:
    """
    Flip horizontal avec probabilité `probability` (défaut 30 %).

    POURQUOI : Sur certains postes MINT, des caméras sont installées des
    deux côtés de la barrière (sens entrant et sens sortant). Une même
    plaque peut donc être vue de face ou en miroir selon le sens de
    circulation. 30 % est conservative (moins de 50 % = sens dominant).
    """
    if random.random() < probability:
        return cv2.flip(img, 1)
    return img.copy()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINES D'AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment_char(img: np.ndarray,
                 n: int = 1,
                 seed: int = None) -> list:
    """
    Génère n versions augmentées d'un caractère 28×28 (niveaux de gris uint8).

    Probabilités par transform (appliquées séquentiellement) :
      rotate_char()          70 %  — inclinaison résiduelle post-segmentation
      add_gaussian_noise()   70 %  — grain caméra (surtout nocturne)
      translate_char()       60 %  — imprécision de localisation
      erode_or_dilate()      50 %  — dégradation peinture / boue
      affine_distort_char()  50 %  — angle oblique caméra haute
      add_char_occlusion()   30 %  — occlusion partielle (boue, sticker)

    seed : si fourni, réinitialise random + numpy avant la boucle
           pour garantir la reproductibilité.
    Retourne : liste de n np.ndarray uint8 (28, 28).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results = []
    for _ in range(n):
        out = img.copy()
        if random.random() < 0.70:
            out = rotate_char(out)
        if random.random() < 0.70:
            out = add_gaussian_noise(out)
        if random.random() < 0.60:
            out = translate_char(out)
        if random.random() < 0.50:
            out = erode_or_dilate(out)
        if random.random() < 0.50:
            out = affine_distort_char(out)
        if random.random() < 0.30:
            out = add_char_occlusion(out)
        results.append(out.astype(np.uint8))
    return results


def augment_plate_crop(img: np.ndarray,
                        n: int = 1,
                        seed: int = None) -> list:
    """
    Génère n versions augmentées d'un crop de plaque 200×60 BGR uint8.
    Simule les conditions réelles des postes de contrôle MINT/DGI camerounais.

    Probabilités par transform :
      adjust_brightness()          70 %  — éclairage tropical / nocturne
      adjust_contrast()            60 %  — hétérogénéité des capteurs
      simulate_motion_blur()       40 %  — véhicule en mouvement lent
      simulate_night_conditions()  25 %  — barrages nocturnes
      simulate_plate_degradation() 35 %  — plaques dégradées terrain
      rotate_plate()               50 %  — inclinaison véhicule / caméra
      perspective_plate()          45 %  — angle caméra haute fixe
      flip_plate()                 30 %  — caméras deux côtés (interne)

    seed : si fourni, réinitialise random + numpy avant la boucle.
    Retourne : liste de n np.ndarray uint8 (60, 200, 3).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results = []
    for _ in range(n):
        out = img.copy()
        if random.random() < 0.70:
            out = adjust_brightness(out)
        if random.random() < 0.60:
            out = adjust_contrast(out)
        if random.random() < 0.40:
            out = simulate_motion_blur(out)
        if random.random() < 0.25:
            out = simulate_night_conditions(out)
        if random.random() < 0.35:
            out = simulate_plate_degradation(out)
        if random.random() < 0.50:
            out = rotate_plate(out)
        if random.random() < 0.45:
            out = perspective_plate(out)
        # flip_plate gère sa propre probabilité (30 %)
        out = flip_plate(out)
        results.append(out.astype(np.uint8))
    return results


def augment_dataset_chars(chars_dir,
                           target_per_class: int = 300,
                           seed: int = 42) -> dict:
    """
    Équilibre le dataset de caractères pour les 36 classes VALID_CHARS.

    chars_dir : Path vers data/processed/characters/
                Structure attendue : chars_dir/{label}/img_XXXXXX.png

    Pour chaque classe avec N < target_per_class :
      - n_needed = target_per_class − N
      - Tire aléatoirement parmi les images existantes (avec remplacement)
      - Appelle augment_char(n=1) pour chaque image nécessaire
      - Sauvegarde : même dossier, suffixe _aug_{k:04d}.png

    Retourne :
      {
        "n_classes"    : int,
        "n_added"      : int,
        "distribution" : {"A": int, "B": int, ...}  ← counts après augmentation
      }
    Lève FileNotFoundError si chars_dir n'existe pas.
    """
    chars_dir = Path(chars_dir)
    if not chars_dir.exists():
        raise FileNotFoundError(f"Dossier de caractères introuvable : {chars_dir}")

    py_rng = random.Random(seed)
    n_added = 0
    distribution: dict = {}

    label_dirs = sorted([d for d in chars_dir.iterdir() if d.is_dir()])

    for label_dir in tqdm(label_dirs, desc="Augmentation classes"):
        label = label_dir.name
        png_paths = sorted(label_dir.glob("*.png"))
        N = len(png_paths)
        distribution[label] = N

        if N == 0 or N >= target_per_class:
            continue

        n_needed = target_per_class - N
        k = 0
        while k < n_needed:
            src_path = py_rng.choice(png_paths)
            img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            aug_imgs = augment_char(img, n=1, seed=seed + k)
            aug_name = src_path.stem + f"_aug_{k:04d}.png"
            cv2.imwrite(str(label_dir / aug_name), aug_imgs[0])
            k += 1
            n_added += 1

        distribution[label] = target_per_class

    return {
        "n_classes":    len(label_dirs),
        "n_added":      n_added,
        "distribution": distribution,
    }
