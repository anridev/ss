import json
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

# ONNX Conversion
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent  # Project root
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent)) # Add SpamShield to path for utils

import config
from utils import preprocess_text


DATASETS_ROOT = Path(os.getenv("DATASETS_ROOT", str(BASE_DIR / "Datasets")))
DATASET_CONFIG_PATH = Path(os.getenv("DATASET_CONFIG_PATH", str(DATASETS_ROOT / "0.4.json")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR)))
TARGET_MIN_PRECISION = float(os.getenv("TARGET_MIN_PRECISION", str(config.TARGET_MIN_PRECISION)))
SPAM_THRESHOLD = float(os.getenv("SPAM_THRESHOLD", str(config.SPAM_THRESHOLD)))


def load_all_data(dataset_config_path: Path):
    records = []
    seen = set()
    if not dataset_config_path.exists():
        print(f"Config file not found: {dataset_config_path}")
        return records

    with dataset_config_path.open("r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    for relative_path_str in config_data.get("files", []):
        if "*" in relative_path_str:
            paths = list(DATASETS_ROOT.glob(relative_path_str))
        else:
            paths = [DATASETS_ROOT / relative_path_str]
            
        for path in paths:
            if not path.exists():
                print(f"Warning: File not found {path}")
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = preprocess_text(data.get("text", ""))
                        if not text:
                            continue
                        if text in seen:
                            continue
                        seen.add(text)
                        label = int(data.get("label", 0))
                        category = data.get("category")
                        if not category:
                            category = "spam" if label == 1 else "normal"
                        records.append({"text": text, "label": label, "category": str(category)})
                    except json.JSONDecodeError:
                        continue
    return records


def pick_threshold(y_true, y_prob, target_min_precision):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_threshold = SPAM_THRESHOLD
    best_score = -1.0

    for idx, threshold in enumerate(thresholds):
        p = precision[idx + 1]
        r = recall[idx + 1]
        if p < target_min_precision:
            continue
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_score:
            best_score = f1
            best_threshold = float(threshold)

    if best_score >= 0:
        return round(best_threshold, 4)

    f1_scores = []
    for idx, threshold in enumerate(thresholds):
        p = precision[idx + 1]
        r = recall[idx + 1]
        f1 = 0.0 if (p + r) == 0 else (2 * p * r / (p + r))
        f1_scores.append((f1, threshold))

    if not f1_scores:
        return SPAM_THRESHOLD

    return round(float(max(f1_scores, key=lambda x: x[0])[1]), 4)


def main():
    print("Starting training for model 0.4...", flush=True)
    records = load_all_data(DATASET_CONFIG_PATH)
    if not records:
        raise SystemExit(f"No records found using config {DATASET_CONFIG_PATH}")

    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    categories = [r["category"] for r in records]

    print(f"Dataset size: {len(records)}")
    print(f"Label distribution: {Counter(labels)}")

    X_train_texts, X_val_texts, y_train, y_val, y_train_cat, y_val_cat = train_test_split(
        texts,
        labels,
        categories,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # Use FeatureUnion to combine Word and Char TF-IDF into one pipeline step
    union = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=config.WORD_MAX_FEATURES,
            min_df=config.MIN_DF,
            sublinear_tf=config.SUBLINEAR_TF,
            dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=config.CHAR_MAX_FEATURES,
            min_df=config.MIN_DF,
            sublinear_tf=config.SUBLINEAR_TF,
            dtype=np.float32,
        ))
    ])

    binary_pipeline = Pipeline([
        ("union", union),
        ("classifier", LogisticRegression(
            solver="saga",
            max_iter=500,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
            verbose=1,
        ))
    ])

    print("Fitting binary model pipeline...")
    binary_pipeline.fit(X_train_texts, y_train)
    
    y_prob = binary_pipeline.predict_proba(X_val_texts)[:, 1]
    threshold = pick_threshold(y_val, y_prob, TARGET_MIN_PRECISION)
    y_pred = (y_prob >= threshold).astype(int)

    print("\n--- TF-IDF Model Results (Full) ---")
    print(f"Threshold: {threshold}")
    print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_val, y_pred, zero_division=0):.4f}")

    print("\nTraining spam category classifier...")
    train_spam_mask = np.array(y_train) == 1
    val_spam_mask = np.array(y_val) == 1

    if train_spam_mask.any():
        spam_X_train_texts = [X_train_texts[i] for i, m in enumerate(train_spam_mask) if m]
        spam_y_train_cat = np.array(y_train_cat)[train_spam_mask]
        
        category_pipeline = Pipeline([
            ("union", union), # Reuse the fitted union
            ("classifier", LogisticRegression(
                max_iter=20000,
                tol=1e-2,
                class_weight="balanced",
                solver="saga",
                random_state=42,
                verbose=1,
            ))
        ])
        category_pipeline.fit(spam_X_train_texts, spam_y_train_cat)
    else:
        category_pipeline = binary_pipeline

    if val_spam_mask.any():
        spam_X_val_texts = [X_val_texts[i] for i, m in enumerate(val_spam_mask) if m]
        spam_y_val_cat = np.array(y_val_cat)[val_spam_mask]
        y_pred_cat = category_pipeline.predict(spam_X_val_texts)
        cat_accuracy = accuracy_score(spam_y_val_cat, y_pred_cat)
        cat_f1 = f1_score(spam_y_val_cat, y_pred_cat, average="weighted", zero_division=0)
        print("\n--- Spam Category Results (spam-only validation set) ---")
        print(f"Accuracy: {cat_accuracy:.4f}")
        print(f"Weighted F1: {cat_f1:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save scikit-learn models
    joblib.dump(union, MODEL_DIR / "vectorizer.pkl", compress=3)
    joblib.dump(binary_pipeline.named_steps["classifier"], MODEL_DIR / "binary_model.pkl", compress=3)
    joblib.dump(category_pipeline.named_steps["classifier"], MODEL_DIR / "category_model.pkl", compress=3)

    # Export to ONNX for production inference (Lossless conversion)
    if ONNX_AVAILABLE:
        print("\nExporting models to ONNX...")
        initial_type = [('input', StringTensorType([None, 1]))]
        
        # Convert binary model
        onnx_binary = convert_sklearn(binary_pipeline, initial_types=initial_type, 
                                     target_opset=12, options={'zipmap': False})
        with open(MODEL_DIR / "binary_model.onnx", "wb") as f:
            f.write(onnx_binary.SerializeToString())
            
        # Convert category model
        onnx_category = convert_sklearn(category_pipeline, initial_types=initial_type, 
                                       target_opset=12, options={'zipmap': False})
        with open(MODEL_DIR / "category_model.onnx", "wb") as f:
            f.write(onnx_category.SerializeToString())
        print("ONNX models saved successfully.")
    else:
        print("\nWarning: skl2onnx not found. Skipping ONNX export.")

    metadata = {
        "spam_threshold": threshold,
        "short_text_word_count": config.SHORT_TEXT_WORD_COUNT,
        "short_text_threshold": config.SHORT_TEXT_THRESHOLD,
        "very_short_text_word_count": config.VERY_SHORT_TEXT_WORD_COUNT,
        "very_short_text_threshold": config.VERY_SHORT_TEXT_THRESHOLD,
        "target_min_precision": TARGET_MIN_PRECISION,
        "vectorizer": "tfidf_word_char",
        "word_max_features": config.WORD_MAX_FEATURES,
        "char_max_features": config.CHAR_MAX_FEATURES,
        "min_df": config.MIN_DF,
    }
    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
