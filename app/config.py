from pathlib import Path

# Base paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset root – same as notebook
DATA_ROOT = PROJECT_ROOT / "data" / "pins_face_recognition"

# Embeddings + model cache folder
CACHE_DIR = PROJECT_ROOT / "embeddings_cache"

# Model files (from your notebook)
CLF_FILE = CACHE_DIR / "svc_model_retrained.pkl"
CENTROIDS_FILE = CACHE_DIR / "centroids.npy"
CLASSES_FILE = CACHE_DIR / "classes.npy"

# Optional embedding files (not mandatory for deployment)
EMB_FILE = CACHE_DIR / "X_emb_augmented.npy"
LBL_FILE = CACHE_DIR / "y_lbl_augmented.npy"

# Image extensions allowed for the gallery
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
