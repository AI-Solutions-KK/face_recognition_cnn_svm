"""
Inference helper for Streamlit demo.

Pipeline:
- MTCNN + InceptionResnetV1 (facenet-pytorch) for face embeddings
- SVM classifier (svc_model_retrained.pkl) for prediction
- Extra helpers for training & prediction reports.
"""

from pathlib import Path
import pickle
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1

from .config import (
    CLF_FILE,
    CENTROIDS_FILE,
    CLASSES_FILE,
    DATA_ROOT,
    IMAGE_EXTENSIONS,
)

# -------------------------------------------------------------------
# Global objects (loaded once)
# -------------------------------------------------------------------
mtcnn = None
resnet = None
clf = None
label_encoder = None
normalizer = None
centroid_matrix = None
classes_order = None
_models_loaded = False


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
def load_models():
    """
    Initialize MTCNN + ResNet backbone and load SVM + centroid artifacts.
    Safe to call multiple times; actual loading happens once.
    """
    global mtcnn, resnet, clf, label_encoder, normalizer
    global centroid_matrix, classes_order, _models_loaded

    if _models_loaded:
        return

    device = "cpu"

    # ---- Face detector + embedding model ----
    mtcnn_local = MTCNN(keep_all=False, device=device)
    resnet_local = InceptionResnetV1(pretrained="vggface2").eval()

    # ---- Load SVM classifier + label encoder + normalizer ----
    with open(CLF_FILE, "rb") as f:
        obj = pickle.load(f)

    clf_local = obj["clf"]
    le_local = obj["le"]
    norm_local = obj["norm"]

    # ---- Load centroid artifacts (optional) ----
    try:
        centroid_matrix_local = np.load(CENTROIDS_FILE)
        classes_order_local = np.load(CLASSES_FILE)
    except FileNotFoundError:
        centroid_matrix_local = None
        classes_order_local = None

    # Assign to globals only after successful load
    mtcnn = mtcnn_local
    resnet = resnet_local
    clf = clf_local
    label_encoder = le_local
    normalizer = norm_local
    centroid_matrix = centroid_matrix_local
    classes_order = classes_order_local

    _models_loaded = True
    print("Models loaded successfully from:", CLF_FILE)


# -------------------------------------------------------------------
# Utility: dataset inspection
# -------------------------------------------------------------------
def list_dataset_images():
    """Return list of all image paths inside DATA_ROOT."""
    images = []
    for person_folder in DATA_ROOT.iterdir():
        if person_folder.is_dir():
            for img_file in person_folder.iterdir():
                if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(str(img_file))
    return images


def compute_class_distribution() -> Tuple[Dict[str, int], int]:
    """
    Compute number of images per class (folder) in DATA_ROOT.

    Returns:
        (counts_dict, total_images)
    """
    counts: Dict[str, int] = {}
    total = 0
    for person_folder in DATA_ROOT.iterdir():
        if not person_folder.is_dir():
            continue
        n = sum(
            1
            for img_file in person_folder.iterdir()
            if img_file.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[person_folder.name] = n
        total += n
    return counts, total


# -------------------------------------------------------------------
# Core: extract embedding from an image path
# -------------------------------------------------------------------
def _extract_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Given an image path, detect face, align with MTCNN, and get a 512-d embedding.
    Returns:
        emb (np.ndarray of shape (512,)) or None if no face detected.
    """
    # Ensure models loaded
    load_models()

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect + align face
    face = mtcnn(img_rgb)
    if face is None:
        return None

    # If single face, shape [C,H,W] -> add batch dim
    if face.dim() == 3:
        face = face.unsqueeze(0)

    with torch.no_grad():
        emb = resnet(face).cpu().numpy().reshape(-1)  # (512,)

    return emb


# -------------------------------------------------------------------
# Prediction using SVM  (single image)
# -------------------------------------------------------------------
def predict_image(image_path: str, top_k: int = 3):
    """
    Predict label + confidence for a given image path.

    Returns dict:
        {
            "predicted_label": str or None,
            "confidence": float or None,
            "top_k": [(label, prob), ...] or None,
            "error": optional error message
        }
    """
    global clf, label_encoder, normalizer

    # Ensure models are loaded (covers any Streamlit reload edge cases)
    load_models()

    emb = _extract_embedding(image_path)
    if emb is None:
        return {
            "predicted_label": None,
            "confidence": None,
            "top_k": None,
            "error": "No face detected in the image.",
        }

    # Shape to (1, 512)
    emb = emb.reshape(1, -1)

    # Apply the same normalizer used in training (Normalizer('l2'))
    if normalizer is not None:
        emb_norm = normalizer.transform(emb)
    else:
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # Predict probabilities
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(emb_norm)[0]
    else:
        scores = clf.decision_function(emb_norm)[0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

    # Get top-k indices
    top_k = min(top_k, len(probs))
    idx_sorted = np.argsort(probs)[::-1][:top_k]

    top_labels = label_encoder.inverse_transform(idx_sorted)
    top_probs = probs[idx_sorted]

    predicted_label = top_labels[0]
    confidence = float(top_probs[0])

    top_k_list = [
        (str(lbl), float(p)) for lbl, p in zip(top_labels, top_probs)
    ]

    return {
        "predicted_label": str(predicted_label),
        "confidence": confidence,
        "top_k": top_k_list,
    }


# -------------------------------------------------------------------
# Batch evaluation for Prediction Report (optional)
# -------------------------------------------------------------------
def evaluate_dataset(
    images_per_class: int = 5,
    max_images: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Run prediction on a subset of the dataset for reporting.

    Args:
        images_per_class: max images per class (folder) to evaluate.
        max_images: optional global cap on total images (None = no cap).

    Returns:
        df: DataFrame with columns [image_path, true_label, predicted_label, confidence, correct]
        accuracy: float in [0,1] or None if df empty.
    """
    records = []
    total_seen = 0

    for person_folder in sorted(DATA_ROOT.iterdir()):
        if not person_folder.is_dir():
            continue

        true_label = person_folder.name
        images = [
            img_file
            for img_file in sorted(person_folder.iterdir())
            if img_file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if images_per_class is not None:
            images = images[:images_per_class]

        for img_path in images:
            res = predict_image(str(img_path), top_k=3)

            records.append(
                {
                    "image_path": str(img_path),
                    "true_label": true_label,
                    "predicted_label": res.get("predicted_label"),
                    "confidence": res.get("confidence"),
                    "error": res.get("error"),
                    "correct": (
                        res.get("predicted_label") == true_label
                        if res.get("predicted_label") is not None
                        else False
                    ),
                }
            )

            total_seen += 1
            if max_images is not None and total_seen >= max_images:
                break

        if max_images is not None and total_seen >= max_images:
            break

    if not records:
        return pd.DataFrame(), None

    df = pd.DataFrame.from_records(records)
    if "correct" in df.columns and len(df) > 0:
        accuracy = float(df["correct"].mean())
    else:
        accuracy = None

    return df, accuracy
