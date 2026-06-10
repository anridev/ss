import os
from pathlib import Path

# Model paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = str(BASE_DIR)
BINARY_MODEL_PATH = os.path.join(MODEL_DIR, "binary_model.pkl")
CATEGORY_MODEL_PATH = os.path.join(MODEL_DIR, "category_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# Vectorizer settings (TF-IDF word + char, lite)
WORD_MAX_FEATURES = 10000
CHAR_MAX_FEATURES = 5000
MIN_DF = 3
SUBLINEAR_TF = True

# Prediction settings
SPAM_THRESHOLD = 0.65
SHORT_TEXT_WORD_COUNT = 2
SHORT_TEXT_THRESHOLD = 0.9
VERY_SHORT_TEXT_WORD_COUNT = 1
VERY_SHORT_TEXT_THRESHOLD = 0.96
LONG_TEXT_WORD_THRESHOLD = 80
CHUNK_MAX_WORDS = 40
MAX_CHUNKS = 24
BLOCKED_URL_DOMAINS = {
    "suspicious-free-prize-now.biz",
}

# Precision/recall tuning during training
TARGET_MIN_PRECISION = 0.98
