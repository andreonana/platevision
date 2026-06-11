#!/usr/bin/env bash
# =============================================================================
# PlateVision — Script d'initialisation du dépôt Git
# Compatible : Linux, macOS, WSL Windows
#
# Usage :
#   chmod +x setup_repo.sh
#   ./setup_repo.sh
#
# Ce script crée la structure complète du projet, initialise Git,
# configure les branches et prépare le premier commit.
# =============================================================================

set -euo pipefail

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

info()    { echo -e "${BLUE}[INFO]${NC}    $1"; }
success() { echo -e "${GREEN}[OK]${NC}      $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC}    $1"; }
error()   { echo -e "${RED}[ERREUR]${NC}  $1"; exit 1; }

echo ""
echo "============================================================"
echo "   PlateVision — Initialisation du dépôt Git"
echo "   MINT / DGI Cameroun — UCAC-ICAM / ULC-ICAM"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# Vérification des prérequis
# -----------------------------------------------------------------------------
info "Vérification des prérequis..."

command -v git >/dev/null 2>&1 || error "Git n'est pas installé. Installer git et relancer."
command -v python3 >/dev/null 2>&1 || error "Python 3 n'est pas installé."

GIT_VERSION=$(git --version | awk '{print $3}')
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
info "Git version : $GIT_VERSION"
info "Python version : $PYTHON_VERSION"
success "Prérequis vérifiés"

# -----------------------------------------------------------------------------
# Création de la structure des dossiers
# -----------------------------------------------------------------------------
info "Création de la structure des dossiers..."

DIRS=(
    "data/raw"
    "data/processed"
    "data/annotations"
    "modules/module_a"
    "modules/module_b"
    "modules/module_c"
    "modules/module_d"
    "preprocessing"
    "models/weights"
    "models/configs"
    "notebooks"
    "reports/DSD"
    "reports/gestion_projet"
    "reports/rapport_technique"
    "tests"
    ".github/ISSUE_TEMPLATE"
)

for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
    # Créer un .gitkeep pour les dossiers normalement vides
    case "$dir" in
        data/raw|data/processed|data/annotations|models/weights|reports/DSD|reports/gestion_projet|reports/rapport_technique)
            touch "$dir/.gitkeep"
            ;;
    esac
done

success "Structure des dossiers créée"

# -----------------------------------------------------------------------------
# Initialisation Git
# -----------------------------------------------------------------------------
info "Initialisation du dépôt Git..."

if [ -d ".git" ]; then
    warning "Dépôt Git déjà initialisé. Ignoré."
else
    git init
    success "Dépôt Git initialisé"
fi

# Configuration locale (ne pas écraser la config globale)
if ! git config user.name >/dev/null 2>&1; then
    warning "git user.name non configuré. Configurer avec :"
    echo "    git config --global user.name 'Votre Nom'"
fi
if ! git config user.email >/dev/null 2>&1; then
    warning "git user.email non configuré. Configurer avec :"
    echo "    git config --global user.email 'votre@email.com'"
fi

# -----------------------------------------------------------------------------
# Création des fichiers minimaux (si absents)
# -----------------------------------------------------------------------------
info "Création des fichiers de base..."

# .gitignore minimal si absent
if [ ! -f ".gitignore" ]; then
cat > .gitignore << 'GITIGNORE'
venv/
.venv/
.env
data/raw/
models/weights/
__pycache__/
*.py[cod]
*.pt
*.npy
runs/
.ipynb_checkpoints/
.DS_Store
Thumbs.db
GITIGNORE
    success ".gitignore créé"
fi

# .env.example si absent
if [ ! -f ".env.example" ]; then
cat > .env.example << 'ENVEXAMPLE'
DATA_RAW_PATH=data/raw/
DATA_PROCESSED_PATH=data/processed/
YOLO_WEIGHTS_PATH=models/weights/yolov8_platevision.pt
YOLO_CONFIDENCE_THRESHOLD=0.5
DEVICE=cpu
RANDOM_SEED=42
ENVEXAMPLE
    success ".env.example créé"
fi

# README minimal si absent
if [ ! -f "README.md" ]; then
cat > README.md << 'README'
# PlateVision

Système de Reconnaissance Automatique de Plaques d'Immatriculation — MINT/DGI Cameroun

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
README
    success "README.md créé"
fi

# requirements.txt minimal si absent
if [ ! -f "requirements.txt" ]; then
cat > requirements.txt << 'REQ'
ultralytics==8.3.40
easyocr==1.7.2
opencv-python==4.10.0.84
scikit-learn==1.5.2
torch==2.4.1
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.2
pytest==7.4.4
python-dotenv==1.0.1
REQ
    success "requirements.txt créé"
fi

# -----------------------------------------------------------------------------
# Premier commit sur main
# -----------------------------------------------------------------------------
info "Staging des fichiers initiaux..."

git add \
    .gitignore \
    .env.example \
    README.md \
    2>/dev/null || true

# Ajouter les autres fichiers s'ils existent
for f in requirements.txt CONTRIBUTING.md main.py; do
    [ -f "$f" ] && git add "$f"
done

# Ajouter les dossiers structurés
for dir in modules preprocessing tests data/README_data.md models/configs notebooks reports .github; do
    [ -e "$dir" ] && git add "$dir" 2>/dev/null || true
done

# Vérifier qu'il y a des fichiers à committer
if git diff --cached --quiet; then
    warning "Aucun fichier stagé. Le dépôt était peut-être déjà initialisé."
else
    git commit -m "chore(init): initialisation du dépôt PlateVision

Structure complète du projet : modules A, B, C, D
Prétraitement, tests, notebooks, configuration Git

Co-Authored-By: PlateVision Team <platevision@ucac-icam.cm>"
    success "Premier commit effectué sur main"
fi

# -----------------------------------------------------------------------------
# Création des branches de l'équipe
# -----------------------------------------------------------------------------
info "Création des branches de l'équipe..."

BRANCHES=("develop" "feature/module-a" "feature/module-b" "feature/module-c" "feature/module-d")

for branch in "${BRANCHES[@]}"; do
    if git show-ref --verify --quiet "refs/heads/$branch"; then
        warning "Branche '$branch' existe déjà. Ignorée."
    else
        git branch "$branch"
        success "Branche créée : $branch"
    fi
done

# Revenir sur develop comme branche de travail par défaut
git checkout develop 2>/dev/null || git checkout -b develop

success "Branches créées. Branche courante : develop"

# -----------------------------------------------------------------------------
# Résumé et instructions
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo -e "${GREEN}   Initialisation terminée avec succès !${NC}"
echo "============================================================"
echo ""
echo "Branches disponibles :"
git branch -a
echo ""
echo "Structure créée :"
if command -v tree >/dev/null 2>&1; then
    tree -L 2 --dirsfirst -I '__pycache__|*.pyc|.git'
else
    find . -maxdepth 2 -not -path './.git/*' | sort | head -40
fi

echo ""
echo "============================================================"
echo "   PROCHAINES ÉTAPES"
echo "============================================================"
echo ""
echo "1. Connecter au dépôt GitHub distant :"
echo "   git remote add origin https://github.com/andreonana/platevision.git"
echo "   git push -u origin main"
echo "   git push origin develop"
echo "   git push origin feature/module-a feature/module-b feature/module-c feature/module-d"
echo ""
echo "2. Créer l'environnement virtuel Python :"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate  # Linux/macOS/WSL"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Copier les variables d'environnement :"
echo "   cp .env.example .env"
echo "   # Éditer .env avec vos chemins locaux"
echo ""
echo "4. Chaque étudiant checkout sa branche :"
echo "   Étudiant 1 & 2 : git checkout feature/module-a"
echo "   Étudiant 3      : git checkout feature/module-b"
echo "   Étudiant 4      : git checkout feature/module-c"
echo "   Étudiant 5      : git checkout feature/module-d"
echo ""
echo "5. Configurer les protections de branches sur GitHub :"
echo "   Settings → Branches → Add branch ruleset"
echo "   (voir CONTRIBUTING.md pour les détails)"
echo ""
echo "   Bon courage pour le projet PlateVision ! 🚗"
echo "============================================================"
