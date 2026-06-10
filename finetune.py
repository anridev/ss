#! /usr/bin/env python3
"""Fine-tune SpamShield (Approach 1: full retrain) — binary spam/ham only.

Retrains the binary (spam vs. ham) classifier on the original corpus PLUS your
own new labeled data, then re-exports binary_model.onnx so infer.py picks up the
changes automatically. (No category model is trained.)

To avoid leakage and memorization, the new data is split into train/test
*before* it is combined with the corpus. The held-out test split never enters
training and is written to a file (default: test_split.jsonl) so infer.py can
evaluate on it. New examples are emphasized via sample_weight (a real
re-weighting of the loss) rather than by duplicating rows.

Usage:
    # Retrain on the original dataset only (reproduces the released model)
    ./finetune.py

    # Fine-tune by mixing in your own labeled examples (multiple files allowed)
    ./finetune.py --new-data new_phishing.jsonl generated_phishing.jsonl

    # Emphasize new examples in the loss (sample weight, not duplication)
    ./finetune.py --new-data new_phishing.jsonl --new-data-weight 10

    # Control the held-out test fraction and where it is written
    ./finetune.py --new-data new_phishing.jsonl --new-test-size 0.2 --eval-out test_split.jsonl

New data must be JSONL (one object per line):
    {"text": "...", "label": 0|1}
`label`: 0 = ham, 1 = spam. Any extra fields (e.g. "category") are ignored.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
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
from sklearn.pipeline import FeatureUnion, Pipeline

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from utils import preprocess_text

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
except ImportError:
    print("ERROR: skl2onnx is required for ONNX export. Install it with:\n"
          "  pip install skl2onnx onnx", file=sys.stderr)
    raise


def load_jsonl_records(paths, seen):
    """Read JSONL files into deduplicated records.

    Each record keeps both the preprocessed `text` (used for training, matching
    what the ONNX vectorizer sees) and the original `raw` text (written to the
    test-split file so infer.py preprocesses it exactly once, like real input).
    """
    records = []
    for path in paths:
        if not path.exists():
            print(f"Warning: file not found {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw = data.get("text", "")
                text = preprocess_text(raw)
                if not text or text in seen:
                    continue
                seen.add(text)
                label = int(data.get("label", 0))
                records.append({"text": text, "raw": raw, "label": label})
    return records


def load_corpus(datasets_root, config_path, seen):
    """Load the original dataset using the {"files": [...]} glob config."""
    if not config_path.exists():
        raise SystemExit(f"Dataset config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    all_paths = []
    for pattern in cfg.get("files", []):
        if "*" in pattern:
            all_paths.extend(sorted(datasets_root.glob(pattern)))
        else:
            all_paths.append(datasets_root / pattern)
    return load_jsonl_records(all_paths, seen)


def collect_new_paths(entries):
    """Expand a list of files/directories into a flat list of .jsonl paths."""
    paths = []
    for entry in entries:
        p = Path(entry)
        if p.is_dir():
            paths.extend(sorted(p.glob("**/*.jsonl")))
        else:
            paths.append(p)
    return paths


def pick_threshold(y_true, y_prob, target_min_precision, fallback):
    """Pick the F1-optimal threshold subject to a minimum-precision constraint."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_threshold, best_score = fallback, -1.0
    for idx, threshold in enumerate(thresholds):
        p, r = precision[idx + 1], recall[idx + 1]
        if p < target_min_precision or (p + r) == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_score:
            best_score, best_threshold = f1, float(threshold)
    if best_score >= 0:
        return round(best_threshold, 4)
    # No threshold meets the precision target -> fall back to global best F1.
    scored = [
        (0.0 if (precision[i + 1] + recall[i + 1]) == 0
         else 2 * precision[i + 1] * recall[i + 1] / (precision[i + 1] + recall[i + 1]), t)
        for i, t in enumerate(thresholds)
    ]
    return round(float(max(scored, key=lambda x: x[0])[1]), 4) if scored else fallback


def build_union():
    """Word + char TF-IDF FeatureUnion, matching the released 0.4 config."""
    return FeatureUnion([
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
        )),
    ])


def export_onnx(pipeline, out_path):
    initial_type = [("input", StringTensorType([None, 1]))]
    # zipmap=True makes output_probability a sequence of {class: prob} dicts,
    # matching the released models so infer.py's `binary_out[1][0].get(1, 0.0)` works.
    onnx_model = convert_sklearn(
        pipeline, initial_types=initial_type, target_opset=12, options={"zipmap": True}
    )
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def evaluate_holdout(binary_pipeline, threshold, records, title):
    """Evaluate on held-out records that were never part of training."""
    if not records:
        return
    texts = [r["text"] for r in records]
    y_true = np.array([r["label"] for r in records])
    prob = binary_pipeline.predict_proba(texts)[:, 1]
    y_pred = (prob >= threshold).astype(int)

    print(f"\n--- {title} ({len(records)} held-out examples) ---")
    print(f"Spam detection (recall on label==1): "
          f"{recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    spam_mask = y_true == 1
    missed = int((spam_mask & (y_pred == 0)).sum())
    if missed:
        print(f"Missed spam ({missed}): showing up to 5")
        shown = 0
        for r, p, pr in zip(records, y_pred, prob):
            if r["label"] == 1 and p == 0:
                print(f"  [{pr:.3f}] {r['raw'][:90]}")
                shown += 1
                if shown >= 5:
                    break


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SpamShield (full retrain + ONNX export)")
    parser.add_argument("--datasets-root", default=str(BASE_DIR / "Datasets"),
                        help="Folder containing the language dataset folders")
    parser.add_argument("--config", default=str(BASE_DIR / "Datasets" / "0.4.json"),
                        help='Dataset config JSON with a {"files": [...]} glob list')
    parser.add_argument("--new-data", nargs="+", default=None,
                        help="One or more .jsonl files / directories with new labeled data")
    parser.add_argument("--new-data-weight", type=float, default=1.0,
                        help="sample_weight applied to new-data training rows (default: 1.0)")
    parser.add_argument("--new-test-size", type=float, default=0.2,
                        help="Fraction of new data held out for evaluation (default: 0.2)")
    parser.add_argument("--eval-out", default=str(BASE_DIR / "test_split.jsonl"),
                        help="Where to write the held-out new-data test split (for infer.py)")
    parser.add_argument("--model-dir", default=str(BASE_DIR),
                        help="Where to write the .pkl / .onnx / metadata.json (default: this folder)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Internal validation fraction for threshold selection")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="Train ONLY on --new-data (NOT recommended; causes forgetting)")
    args = parser.parse_args()

    seen = set()

    corpus = []
    if not args.skip_corpus:
        print("Loading original corpus...", flush=True)
        corpus = load_corpus(Path(args.datasets_root), Path(args.config), seen)
        print(f"  corpus records: {len(corpus)}")

    # --- Split new data into train/test BEFORE combining (prevents leakage) ---
    new_train, new_test = [], []
    if args.new_data:
        print(f"Loading new data from {args.new_data} ...", flush=True)
        new_records = load_jsonl_records(collect_new_paths(args.new_data), seen)
        print(f"  new records: {len(new_records)}")
        if 0.0 < args.new_test_size < 1.0 and len(new_records) >= 5:
            strat = [r["label"] for r in new_records]
            strat = strat if len(set(strat)) > 1 else None
            new_train, new_test = train_test_split(
                new_records, test_size=args.new_test_size, random_state=42, stratify=strat,
            )
        else:
            new_train = new_records
        print(f"  new train: {len(new_train)} | held-out test: {len(new_test)} "
              f"(sample_weight x{args.new_data_weight})")
        # Persist the held-out test split (raw text) for infer.py to evaluate.
        eval_path = Path(args.eval_out)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with eval_path.open("w", encoding="utf-8") as f:
            for r in new_test:
                f.write(json.dumps(
                    {"text": r["raw"], "label": r["label"]},
                    ensure_ascii=False) + "\n")
        print(f"  wrote held-out test split -> {eval_path}")

    # --- Combine corpus + new-train; build parallel sample_weight vector ---
    records = corpus + new_train
    weights = [1.0] * len(corpus) + [args.new_data_weight] * len(new_train)
    if not records:
        raise SystemExit("No training records found. Provide a corpus and/or --new-data.")

    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    print(f"Total training records: {len(records)}")
    print(f"Label distribution: {Counter(labels)}")

    X_tr_txt, X_val_txt, y_tr, y_val, w_tr, w_val = train_test_split(
        texts, labels, weights,
        test_size=args.test_size, random_state=42, stratify=labels,
    )
    w_tr = np.asarray(w_tr, dtype=np.float64)

    union = build_union()
    binary_pipeline = Pipeline([
        ("union", union),
        ("classifier", LogisticRegression(
            solver="saga", max_iter=500, tol=1e-3,
            class_weight="balanced", random_state=42, verbose=1,
        )),
    ])

    print("\nFitting binary model...", flush=True)
    binary_pipeline.fit(X_tr_txt, y_tr, classifier__sample_weight=w_tr)

    y_prob = binary_pipeline.predict_proba(X_val_txt)[:, 1]
    threshold = pick_threshold(y_val, y_prob, config.TARGET_MIN_PRECISION, config.SPAM_THRESHOLD)
    y_pred = (y_prob >= threshold).astype(int)

    print("\n--- Binary model (validation) ---")
    print(f"Threshold: {threshold}")
    print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_val, y_pred, zero_division=0):.4f}")

    # --- Honest evaluation on the held-out new-data test split ---
    evaluate_holdout(binary_pipeline, threshold, new_test, "Held-out new-data evaluation")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving sklearn artifacts (.pkl)...", flush=True)
    joblib.dump(union, model_dir / "vectorizer.pkl", compress=3)
    joblib.dump(binary_pipeline.named_steps["classifier"], model_dir / "binary_model.pkl", compress=3)

    print("Exporting ONNX model...", flush=True)
    export_onnx(binary_pipeline, model_dir / "binary_model.onnx")

    metadata = {
        "spam_threshold": threshold,
        "short_text_word_count": config.SHORT_TEXT_WORD_COUNT,
        "short_text_threshold": config.SHORT_TEXT_THRESHOLD,
        "very_short_text_word_count": config.VERY_SHORT_TEXT_WORD_COUNT,
        "very_short_text_threshold": config.VERY_SHORT_TEXT_THRESHOLD,
        "target_min_precision": config.TARGET_MIN_PRECISION,
        "vectorizer": "tfidf_word_char",
        "word_max_features": config.WORD_MAX_FEATURES,
        "char_max_features": config.CHAR_MAX_FEATURES,
        "min_df": config.MIN_DF,
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Binary model written to {model_dir}")
    print("Run `infer.py` with no arguments to evaluate the held-out test split.")


if __name__ == "__main__":
    main()
