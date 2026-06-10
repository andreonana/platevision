#!/usr/bin/env bash
# ==============================================================================
# PlateVision — Script d'initialisation du dépôt Git
# Compatible : Linux, macOS, WSL (Windows Subsystem for Linux)
#
# Usage :
#   chmod +x setup_repo.sh
#   ./setup_repo.sh
# ==============================================================================

set -euo pipefail

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         PlateVision — Initialisation du Dépôt           ║"
echo "║         MINT/DGI Cameroun — UCAC-ICAM / ULC-ICAM       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ==============================================================================
# PARTIE 1 — Création de la structure des dossiers
# ==============================================================================
log_info "Création de la structure des dossiers..."

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
    "reports/figures"
    "tests"
    ".github/ISSUE_TEMPLATE"
    ".github/PULL_REQUEST_TEMPLATE"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        log_success "Créé : $dir/"
    else
        log_warn "Existe déjà : $dir/"
    fi
done

# Fichiers .gitkeep pour versionner les dossiers vides
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/annotations/.gitkeep
touch models/weights/.gitkeep
touch reports/DSD/.gitkeep
touch reports/gestion_projet/.gitkeep
touch reports/rapport_technique/.gitkeep
touch reports/figures/.gitkeep

log_success "Structure des dossiers créée."

# ==============================================================================
# PARTIE 2 — Fichiers __init__.py pour les modules Python
# ==============================================================================
log_info "Création des fichiers __init__.py..."

PYTHON_PACKAGES=(
    "modules"
    "modules/module_a"
    "modules/module_b"
    "modules/module_c"
    "modules/module_d"
    "preprocessing"
    "tests"
)

for pkg in "${PYTHON_PACKAGES[@]}"; do
    init_file="$pkg/__init__.py"
    if [ ! -f "$init_file" ]; then
        echo '"""PlateVision — Package Python."""' > "$init_file"
        log_success "Créé : $init_file"
    fi
done

# ==============================================================================
# PARTIE 3 — Initialisation Git
# ==============================================================================
log_info "Vérification du dépôt Git..."

if [ ! -d ".git" ]; then
    git init
    log_success "Dépôt Git initialisé."
else
    log_warn "Dépôt Git déjà initialisé."
fi

# Configuration Git minimale si non configurée
if [ -z "$(git config user.name 2>/dev/null)" ]; then
    log_warn "Configuration Git user.name non définie."
    read -p "  Entrez votre nom complet : " git_name
    git config user.name "$git_name"
fi

if [ -z "$(git config user.email 2>/dev/null)" ]; then
    log_warn "Configuration Git user.email non définie."
    read -p "  Entrez votre email : " git_email
    git config user.email "$git_email"
fi

# ==============================================================================
# PARTIE 4 — Premier Commit Initial sur main
# ==============================================================================
log_info "Préparation du commit initial..."

# Vérifier qu'il existe des fichiers à commiter
if [ -z "$(git status --porcelain)" ]; then
    log_warn "Aucune modification à commiter."
else
    git add \
        README.md \
        requirements.txt \
        .gitignore \
        .env.example \
        CONTRIBUTING.md \
        main.py \
        data/README_data.md \
        modules/ \
        preprocessing/ \
        models/configs/ \
        tests/ \
        .github/ \
        2>/dev/null || true

    # Ajouter les .gitkeep
    git add data/raw/.gitkeep data/processed/.gitkeep data/annotations/.gitkeep \
        models/weights/.gitkeep reports/ 2>/dev/null || true

    git commit -m "[ALL] feat(init): initialisation du dépôt PlateVision

Structure complète du projet PlateVision — Reconnaissance automatique
de plaques d'immatriculation pour le MINT et la DGI du Cameroun.

Modules : A (Naïves Bayes + YOLO/OCR), B (Clustering), C (MDP), D (Éthique)
Équipe : 5 étudiants UCAC-ICAM / ULC-ICAM
Durée : 4 jours intensifs + soutenance"

    log_success "Commit initial créé sur main."
fi

# ==============================================================================
# PARTIE 5 — Création des Branches de l'Équipe
# ==============================================================================
log_info "Création des branches de l'équipe..."

# Branche develop (intégration commune)
if ! git show-ref --verify --quiet refs/heads/develop; then
    git checkout -b develop
    git checkout main 2>/dev/null || git checkout -
    log_success "Branche 'develop' créée."
else
    log_warn "Branche 'develop' existe déjà."
fi

# Branches de fonctionnalités (créées depuis develop)
FEATURE_BRANCHES=(
    "feature/module-a"
    "feature/module-b"
    "feature/module-c"
    "feature/module-d"
)

for branch in "${FEATURE_BRANCHES[@]}"; do
    if ! git show-ref --verify --quiet "refs/heads/$branch"; then
        git checkout develop 2>/dev/null
        git checkout -b "$branch"
        log_success "Branche '$branch' créée depuis develop."
    else
        log_warn "Branche '$branch' existe déjà."
    fi
done

# Retour sur develop
git checkout develop 2>/dev/null
log_success "Toutes les branches créées."

# ==============================================================================
# PARTIE 6 — Résumé et Instructions
# ==============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    RÉSUMÉ                               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Branches créées :"
git branch --list
echo ""
echo "Structure du dépôt :"
find . -not -path './.git/*' -not -name '*.pyc' -maxdepth 3 | sort | head -50
echo ""

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         PROCHAINES ÉTAPES — Connexion GitHub            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "1. Créer le dépôt sur GitHub (si ce n'est pas fait) :"
echo "   → https://github.com/new"
echo "   → Nom suggéré : platevision"
echo "   → Visibilité : Private (recommandé pour un projet académique)"
echo ""
echo "2. Connecter votre dépôt local au remote GitHub :"
echo "   git remote add origin https://github.com/VOTRE_ORG/platevision.git"
echo ""
echo "3. Pousser toutes les branches :"
echo "   git push -u origin main"
echo "   git push -u origin develop"
echo "   git push -u origin feature/module-a"
echo "   git push -u origin feature/module-b"
echo "   git push -u origin feature/module-c"
echo "   git push -u origin feature/module-d"
echo ""
echo "4. Configurer la protection des branches sur GitHub :"
echo "   Settings → Branches → Branch protection rules"
echo "   Voir CONTRIBUTING.md section 6 pour les détails."
echo ""
echo "5. Chaque étudiant clone et configure :"
echo "   git clone https://github.com/VOTRE_ORG/platevision.git"
echo "   cd platevision"
echo "   git checkout feature/module-x"
echo "   python -m venv venv && source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo "   cp .env.example .env"
echo ""
echo "Bonne chance pour PlateVision ! 🚗"
