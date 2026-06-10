"""
PlateVision — Tests unitaires : Pipeline YOLO + OCR (Module A2).
"""

import numpy as np
import pytest
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDetectPlates:
    """Tests pour la détection de plaques avec YOLO."""

    def test_detect_plates_returns_list(self):
        from modules.module_a.yolo_ocr_pipeline import detect_plates

        # Création d'un modèle YOLO simulé
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detect_plates(mock_model, image)

        assert isinstance(detections, list)

    def test_detect_plates_empty_on_blank_image(self):
        from modules.module_a.yolo_ocr_pipeline import detect_plates

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detect_plates(mock_model, image)

        assert detections == []


class TestRecognizeText:
    """Tests pour la reconnaissance OCR des plaques."""

    def test_recognize_text_returns_string(self):
        from modules.module_a.yolo_ocr_pipeline import recognize_text

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "AB123CD", 0.95)
        ]

        image_crop = np.zeros((30, 100, 3), dtype=np.uint8)
        result = recognize_text(mock_reader, image_crop)

        assert isinstance(result, str)
        assert result == "AB123CD"

    def test_recognize_text_filters_low_confidence(self):
        from modules.module_a.yolo_ocr_pipeline import recognize_text

        mock_reader = MagicMock()
        # Retourne un résultat avec confiance faible (< seuil 0.3)
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "NOISE", 0.1)
        ]

        image_crop = np.zeros((30, 100, 3), dtype=np.uint8)
        result = recognize_text(mock_reader, image_crop)

        # Le texte avec confiance faible doit être ignoré
        assert result == ""

    def test_recognize_text_uppercase(self):
        from modules.module_a.yolo_ocr_pipeline import recognize_text

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "ab123cd", 0.9)
        ]

        image_crop = np.zeros((30, 100, 3), dtype=np.uint8)
        result = recognize_text(mock_reader, image_crop)

        assert result == result.upper()


class TestProcessImage:
    """Tests d'intégration pour le traitement d'une image complète."""

    def test_process_image_returns_dict_structure(self, tmp_path):
        from modules.module_a.yolo_ocr_pipeline import process_image

        # Créer une image de test
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(image_path), image)

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []

        result = process_image(mock_model, mock_reader, image_path, tmp_path / "out")

        assert "image" in result
        assert "plates" in result
        assert isinstance(result["plates"], list)

    def test_process_image_nonexistent_file(self, tmp_path):
        from modules.module_a.yolo_ocr_pipeline import process_image

        mock_model = MagicMock()
        mock_reader = MagicMock()

        result = process_image(
            mock_model, mock_reader,
            tmp_path / "nonexistent.jpg",
            tmp_path / "out",
        )

        assert result["plates"] == []
