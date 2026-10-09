import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


IMG_SIZE = (128, 128)


class InputDomainGuard:
    """Two-stage input-domain gate for the malaria classifier.

    Stage 1: cheap image-structure checks.
    Stage 2: one-class model in the trained MobileNetV2 embedding space,
             fitted only on genuine malaria cell images when the dataset is
             available locally.

    This reduces obvious out-of-domain inputs but is not a clinically
    validated OOD detector and cannot guarantee rejection of every possible
    non-cell image.
    """

    def __init__(self, reference_model, max_reference_images: int = 80):
        self.reference_model = reference_model
        self.max_reference_images = max_reference_images
        self.embedding_model = self._make_embedding_model(reference_model)
        self.detector = None
        self.reference_count = 0
        self.reference_source = None
        self._fit_if_possible()

    @staticmethod
    def _make_embedding_model(model):
        dense_layers = [
            layer for layer in model.layers
            if isinstance(layer, tf.keras.layers.Dense)
            and getattr(layer, "units", None) == 128
        ]
        if not dense_layers:
            raise ValueError("Could not locate the 128-unit feature layer.")
        feature_layer = dense_layers[-1]
        return tf.keras.Model(model.input, feature_layer.output)

    @staticmethod
    def _find_dataset_root() -> Optional[Path]:
        candidates = [
            Path("data/raw/clean_cell_images"),
            Path("data/raw/cell_images"),
            Path("data/cell_images"),
        ]
        for root in candidates:
            if (root / "Parasitized").is_dir() and (root / "Uninfected").is_dir():
                return root

        # Kaggle extraction sometimes creates an extra nested cell_images
        # directory. Search only under data/ so we do not accidentally scan
        # the virtual environment or unrelated project files.
        for base in (Path("data/raw"), Path("data")):
            if not base.exists():
                continue
            for parasitized in base.rglob("Parasitized"):
                if (parasitized.parent / "Uninfected").is_dir():
                    return parasitized.parent
        return None

    def _collect_reference_paths(self) -> List[Path]:
        root = self._find_dataset_root()
        if root is None:
            return []

        per_class = max(10, self.max_reference_images // 2)
        paths = []
        for class_name in ("Parasitized", "Uninfected"):
            files = sorted(
                p for p in (root / class_name).iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            if not files:
                continue
            # Deterministic spread across the directory instead of only taking
            # the first files.
            idx = np.linspace(0, len(files) - 1, min(per_class, len(files)), dtype=int)
            paths.extend(files[i] for i in idx)
        self.reference_source = str(root)
        return paths

    @staticmethod
    def _preprocess(image: Image.Image):
        image = image.convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(image, dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        return arr

    def _make_embeddings(self, paths: List[Path]) -> np.ndarray:
        images = []
        valid = []
        for path in paths:
            try:
                with Image.open(path) as im:
                    images.append(self._preprocess(im))
                valid.append(path)
            except Exception:
                continue

        if not images:
            return np.empty((0, 128), dtype=np.float32)

        batch = np.stack(images, axis=0)
        emb = self.embedding_model(batch, training=False).numpy()
        emb = emb.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return emb / norms

    def _fit_if_possible(self):
        paths = self._collect_reference_paths()
        if len(paths) < 20:
            return

        embeddings = self._make_embeddings(paths)
        if len(embeddings) < 20:
            return

        # Conservative one-class boundary. The model learns the compact
        # feature distribution of genuine malaria-cell images.
        self.detector = make_pipeline(
            StandardScaler(),
            OneClassSVM(kernel="rbf", gamma="scale", nu=0.08),
        )
        self.detector.fit(embeddings)
        self.reference_count = len(embeddings)

    @staticmethod
    def _cheap_structure_check(image: Image.Image) -> Tuple[bool, str, Dict[str, float]]:
        image = image.convert("RGB")
        w, h = image.size

        if w < 100 or h < 100:
            return False, "Image resolution is too low for cell analysis.", {}

        aspect = w / float(h)
        if aspect < 0.72 or aspect > 1.38:
            return False, (
                "This does not look like an individual blood-smear cell crop. "
                "Please upload a microscopy cell image."
            ), {"aspect": aspect}

        small = cv2.resize(np.asarray(image), (192, 192), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

        brightness = float(gray.mean())
        contrast = float(gray.std())
        texture = float(cv2.Laplacian(gray, cv2.CV_64F).std())
        saturation = hsv[:, :, 1]

        white_fraction = float(np.mean(
            (small[:, :, 0] > 245)
            & (small[:, :, 1] > 245)
            & (small[:, :, 2] > 245)
        ))

        # Long straight lines are common in charts, tables and screenshots.
        edges = cv2.Canny(gray, 70, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=45,
            minLineLength=45, maxLineGap=4
        )
        straight = 0
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                length = float(np.hypot(x2 - x1, y2 - y1))
                angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
                if length >= 45 and (angle <= 7 or abs(angle - 90) <= 7):
                    straight += 1

        # Look for a large, compact, non-background component. This is only
        # a supporting signal; the embedding detector is the main gate.
        sat_mask = ((saturation > 22) & (gray > 20)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_compact = False
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.04 * 192 * 192 or area > 0.85 * 192 * 192:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity >= 0.18:
                large_compact = True
                break

        features = {
            "aspect": aspect,
            "brightness": brightness,
            "contrast": contrast,
            "texture": texture,
            "white_fraction": white_fraction,
            "straight_lines": float(straight),
            "large_compact_component": float(large_compact),
        }

        if brightness < 8 or brightness > 248:
            return False, "Image exposure is unsuitable for microscopy analysis.", features
        if contrast < 8 or texture < 4:
            return False, "The image has insufficient visual structure for cell analysis.", features
        if white_fraction > 0.55 and straight >= 5:
            return False, "This image appears to be a chart, document, or screenshot.", features
        if white_fraction > 0.72:
            return False, "This image contains too much plain background for a cell crop.", features

        return True, "", features

    def validate(self, image: Image.Image) -> Tuple[bool, str, Dict[str, object]]:
        ok, message, features = self._cheap_structure_check(image)
        if not ok:
            return False, message, {"stage": "structure", "features": features}

        # If genuine reference images are available, require the image to be
        # inside their learned embedding distribution.
        if self.detector is not None:
            # Build the input embedding directly to avoid touching the file system.
            try:
                x = np.expand_dims(self._preprocess(image), axis=0)
                xemb = self.embedding_model(x, training=False).numpy().astype(np.float32)
                xemb /= np.linalg.norm(xemb, axis=1, keepdims=True) + 1e-8
                pred = int(self.detector.predict(xemb)[0])
                score = float(self.detector.decision_function(xemb)[0])
            except Exception as exc:
                return False, (
                    "Input-domain validation could not be completed. "
                    "Please try another microscopy image."
                ), {"stage": "embedding", "error": str(exc)}

            if pred != 1:
                return False, (
                    "The uploaded image is outside the learned microscopy-cell "
                    "image domain. The malaria classifier was not run."
                ), {
                    "stage": "embedding",
                    "score": score,
                    "reference_count": self.reference_count,
                    "reference_source": self.reference_source,
                }

            return True, "", {
                "stage": "accepted",
                "embedding_score": score,
                "reference_count": self.reference_count,
                "reference_source": self.reference_source,
                "features": features,
            }

        # No real reference dataset available: do not pretend the fallback is
        # robust. It is intentionally conservative.
        return False, (
            "Input validation reference images are not available yet. "
            "Place the malaria dataset in data/raw/clean_cell_images/ "
            "with Parasitized and Uninfected folders, then try again."
        ), {
            "stage": "no_reference_dataset",
            "features": features,
        }
