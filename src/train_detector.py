# Importe la python src/train_detector.pyclasse YOLO depuis la bibliothèque ultralytics (elle contient
# tout le nécessaire pour charger, entraîner et utiliser un modèle YOLOv8)
from ultralytics import YOLO

# On définit une fonction main() : bonne pratique pour que le code ne
# s'exécute que lorsqu'on lance le fichier, pas quand on l'importe ailleurs
def main():
    # Charge le modèle YOLOv8n DÉJÀ pré-entraîné sur le dataset COCO.
    # "yolov8n.pt" = la version "nano" (la plus légère et rapide).
    # C'est le transfer learning : on part d'un modèle qui sait déjà
    # reconnaître des formes, au lieu de repartir de zéro.
    model = YOLO("yolov8n.pt")

    # Lance l'entraînement (fine-tuning) du modèle sur NOS plaques.
    model.train(
        # Chemin vers le fichier de configuration généré par l'usine :
        # il indique à YOLO où sont les images train/val/test et les classes.
        data="models/configs/yolov8_platevision.yaml",

        # Nombre d'epochs = combien de fois le modèle voit tout le dataset.
        # On met 3 pour un test rapide (on passera à 50 pour le vrai entraînement).
        epochs=3,

        # Taille à laquelle toutes les images sont redimensionnées : 640x640.
        # C'est la taille standard attendue par YOLOv8.
        imgsz=640,

        # Batch = nombre d'images traitées en même temps avant chaque mise à jour.
        # On met 8 (petit) car on est sur CPU : un batch trop grand saturerait la mémoire.
        batch=8,

        # Patience = arrêt anticipé : si le modèle ne s'améliore plus pendant
        # 10 epochs d'affilée, l'entraînement s'arrête pour ne pas perdre de temps.
        patience=10,

        # Dossier racine où YOLO sauvegarde les résultats de l'entraînement.
        project="runs",

        # Nom du sous-dossier de cet entraînement précis (dans runs/).
        # Les poids finaux seront dans runs/yolo_plaque/weights/best.pt
        name="yolo_plaque"
    )

# Cette condition signifie : "exécute main() seulement si on lance
# directement ce fichier" (et pas s'il est importé par un autre script).
if __name__ == "__main__":
    main()