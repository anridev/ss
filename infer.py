#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import onnxruntime as ort
import numpy as np

from utils import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_SPLIT = BASE_DIR / "test_split.jsonl"

# Load model
binary_sess = ort.InferenceSession('binary_model.onnx')


def classify(message):
    """Return (is_spam, confidence) for a single message."""
    # Apply the same preprocessing used during training to avoid train/serve skew.
    x = np.array([[preprocess_text(message)]], dtype=object)
    binary_out = binary_sess.run(None, {'input': x})
    is_spam = bool(binary_out[0][0] == 1)
    confidence = float(binary_out[1][0].get(1, 0.0))
    return is_spam, confidence


def show_result(message):
    """Classify a message and print the result."""
    is_spam, confidence = classify(message)
    print("Classification result:")
    print(f"  Text       : {message}")
    print(f"  Spam       : {is_spam}")
    print(f"  Confidence : {confidence:.4f}")


def benchmark(message, runs=1000, warmup=50):
    """Measure per-message inference latency."""
    x = np.array([[preprocess_text(message)]], dtype=object)
    # Warm-up to stabilize timings (first runs pay one-off init costs)
    for _ in range(warmup):
        binary_sess.run(None, {'input': x})

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        binary_sess.run(None, {'input': x})
        timings.append((time.perf_counter() - start) * 1000.0)

    timings = np.array(timings)
    print(f"\nInference time over {runs} runs:")
    print(f"  mean   : {timings.mean():.3f} ms")
    print(f"  median : {np.median(timings):.3f} ms")
    print(f"  min    : {timings.min():.3f} ms")
    print(f"  max    : {timings.max():.3f} ms")
    print(f"  p95    : {np.percentile(timings, 95):.3f} ms")
    print(f"  throughput: {1000.0 / timings.mean():.0f} msgs/sec")


def evaluate_test_split(path):
    """Evaluate the held-out test split written by finetune.py."""
    path = Path(path)
    if not path.exists():
        print(f"No test split found at {path}.")
        print("Generate one by fine-tuning with new data, e.g.:")
        print("  .venv/bin/python finetune.py --new-data new_phishing.jsonl generated_phishing.jsonl")
        return

    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print(f"Test split {path} is empty.")
        return

    total = len(rows)
    correct = spam_correct = spam_total = 0
    misses = []
    for r in rows:
        label = int(r.get("label", 0))
        is_spam, confidence = classify(r["text"])
        pred = 1 if is_spam else 0
        if pred == label:
            correct += 1
        if label == 1:
            spam_total += 1
            if pred == 1:
                spam_correct += 1
            else:
                misses.append((confidence, r["text"]))

    print(f"Evaluating {total} held-out examples from {path.name}")
    print(f"  Accuracy            : {correct / total:.4f} ({correct}/{total})")
    if spam_total:
        print(f"  Spam detection rate : {spam_correct / spam_total:.4f} ({spam_correct}/{spam_total})")
    if misses:
        print(f"  Missed spam ({len(misses)}), lowest-confidence first:")
        for conf, txt in sorted(misses)[:5]:
            print(f"    [{conf:.3f}] {txt[:90]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpamShield ONNX spam classifier")
    parser.add_argument("message", nargs="?", default=None,
                        help="Message to classify. If omitted, evaluates the held-out test split.")
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_SPLIT),
                        help=f"Test split to evaluate when no message is given (default: {DEFAULT_TEST_SPLIT.name})")
    parser.add_argument("--runs", type=int, default=1000,
                        help="Number of benchmark iterations (default: 1000)")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Only classify, skip the latency benchmark")
    args = parser.parse_args()

    if args.message is None:
        evaluate_test_split(args.test_file)
    else:
        show_result(args.message)
        if not args.no_benchmark:
            benchmark(args.message, runs=args.runs)
