# SpamShield

Multilingual SMS/text **binary** spam detection (spam vs. ham), based on
[M-Arjun/SpamShield](https://huggingface.co/M-Arjun/SpamShield). It uses a single
ONNX model with TF-IDF vectorization baked into the graph (so it accepts raw
strings):

- **`binary_model.onnx`** — spam vs. ham

Under the hood it is a scikit-learn pipeline: a word + char TF-IDF
`FeatureUnion` feeding a `LogisticRegression`, exported to ONNX via `skl2onnx`.

> The upstream model also ships a `category_model.onnx` (phishing/crypto/etc.).
> This project is binary-only — `finetune.py` and `infer.py` do not train or use
> the category model.

## Layout

```
spamShield/
├── infer.py                  # Classify a message, or evaluate the test split (no args)
├── finetune.py               # Retrain / fine-tune and re-export ONNX models
├── gen_phishing.py           # Generate synthetic phishing examples
├── gen_extortion.py          # Generate synthetic extortion/sextortion examples
├── train.py                  # Original training script (reference)
├── utils.py                  # preprocess_text() — shared text normalization
├── config.py                 # Feature sizes, thresholds
├── metadata.json             # Spam threshold + vectorizer settings
├── binary_model.onnx         # Spam/ham model
├── vectorizer.pkl            # Fitted TF-IDF FeatureUnion (used by finetune.py)
├── new_phishing.jsonl        # Hand-labeled phishing examples (sample new data)
├── generated_phishing.jsonl  # Synthetic phishing from gen_phishing.py
├── generated_extortion.jsonl # Synthetic extortion from gen_extortion.py
├── test_split.jsonl          # Held-out eval split (created by finetune.py)
├── requirements-train.txt
└── Datasets/                 # Training corpus (downloaded from HF) + 0.4.json config
```

## Setup

Homebrew's system Python is externally managed (PEP 668), so use the project
virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt onnxruntime numpy
```

For inference only, `onnxruntime` and `numpy` are enough.

## Prediction

`infer.py` classifies a message and (optionally) benchmarks latency. It applies
the same `preprocess_text()` normalization used during training.

```bash
# Classify a message (also runs the latency benchmark)
.venv/bin/python infer.py "Congratulations! You won a free iPhone, claim now!"

# Classify only, skip the benchmark
.venv/bin/python infer.py "Hey, are we still on for lunch tomorrow?" --no-benchmark

# Control benchmark iterations
.venv/bin/python infer.py "Free Bitcoin airdrop, claim your crypto now" --runs 200
```

Example output:

```
Classification result:
  Text       : Congratulations! You won a free iPhone, claim now!
  Spam       : True
  Confidence : 0.9969

Inference time over 1000 runs:
  mean   : 0.087 ms
  median : 0.084 ms
  ...
  throughput: 11448 msgs/sec
```

Use it as a library:

```python
from infer import classify

is_spam, confidence = classify("Click here to verify your account")
# (True, 0.99...)
```

> Note: `infer.py` imports `utils`, so run it from this directory.

## Training / Fine-tuning

`finetune.py` does a full retrain of the binary spam/ham classifier on the
original corpus plus (optionally) your own labeled data, then re-exports
`binary_model.onnx` in place so `infer.py` picks up the changes automatically.

To avoid leakage and memorization, it:

- splits your new data into train/test **before** combining it with the corpus,
- writes the held-out test split to `test_split.jsonl` (never used for training),
- emphasizes new examples via `sample_weight` (a real loss re-weighting), **not**
  by duplicating rows.

### 1. Get the dataset

The dataset is gated — accept the terms once at
[M-Arjun/SpamShield-Datasets](https://huggingface.co/datasets/M-Arjun/SpamShield-Datasets),
then download:

```bash
hf download M-Arjun/SpamShield-Datasets --repo-type dataset --local-dir Datasets \
  --exclude "combined.parquet" --exclude "filter_datasets.py"
```

`Datasets/0.4.json` (`{"files": ["*/*.jsonl"]}`) tells the trainer which files to load.

### 2. (Optional) Generate synthetic spam examples

`gen_phishing.py` and `gen_extortion.py` produce diverse synthetic messages
(varied brands, amounts, URLs, threats, contact handles, phrasings) so the model
learns generalizable signals instead of memorizing fixed strings:

```bash
.venv/bin/python gen_phishing.py  --n 300   # -> generated_phishing.jsonl
.venv/bin/python gen_extortion.py --n 150   # -> generated_extortion.jsonl
```

### 3. Train

```bash
# Reproduce the released model (retrain on the original corpus only)
.venv/bin/python finetune.py

# Fine-tune by mixing your own labeled data into the full corpus
# (multiple files allowed), emphasizing new examples 10x in the loss
.venv/bin/python finetune.py \
  --new-data new_phishing.jsonl generated_phishing.jsonl generated_extortion.jsonl \
  --new-data-weight 10
```

Useful flags: `--new-data-weight` (loss weight for new rows), `--new-test-size`
(held-out fraction, default 0.2), `--eval-out` (test-split path), `--model-dir`,
`--test-size` (internal validation), `--datasets-root`, `--config`, and
`--skip-corpus` (train only on `--new-data` — not recommended, causes forgetting).

### New-data format

JSONL, one object per line:

```json
{"text": "Your package could not be delivered, confirm details here", "label": 1}
{"text": "Lunch at 1pm works for me", "label": 0}
```

- `label`: `0` = ham, `1` = spam
- Any extra fields (e.g. a `category`) are ignored.

### 4. Use & evaluate the retrained model

```bash
# Classify a message
.venv/bin/python infer.py "your message here"

# No arguments -> evaluate the held-out test split (test_split.jsonl)
.venv/bin/python infer.py
```

Running `infer.py` with no message reports accuracy and spam-detection rate on
the held-out examples that were never seen during training.

## Results

Latest fine-tune: full corpus (142,269 records) + new data (40 hand-labeled
phishing, 300 synthetic phishing, 150 synthetic extortion), `--new-data-weight 10`.

Validation (held-out slice of the combined training set):

| Metric | Value |
| --- | --- |
| Accuracy | 0.959 |
| Precision | 0.980 |
| Recall | 0.941 |
| F1 | 0.960 |
| Threshold | 0.605 |

Held-out new-data split (95 examples never seen in training): **100%** spam
detection.

Generalization spot-check on messages **not** in the training data:

| Message | Predicted | Confidence |
| --- | --- | --- |
| Novel sextortion (webcam + BTC demand) | spam | 0.978 |
| Novel data-leak threat (email + SAR demand) | spam | 0.870 |
| Legitimate ham ("send me the meeting notes") | ham | 0.010 |
| Terse extortion, no explicit demand/deadline | ham (missed) | 0.135 |

The model generalizes well to new phishing/extortion phrasings without raising
false positives on ham. The remaining miss is very short, low-signal extortion
text that lacks an explicit threat/demand/contact structure; adding short-form
extortion examples to `gen_extortion.py` and retraining would help close it.

## Notes & limitations

- The ONNX model is exported with `zipmap=True` so `output_probability` is a
  `{class: prob}` dict (what `infer.py` expects). Keep this if you re-export.
- `metadata.json` holds the tuned `spam_threshold` (≈0.49) plus stricter
  thresholds for short texts; `infer.py` currently uses a plain `label == 1`
  decision rather than these thresholds.
- This is a classical TF-IDF + logistic-regression model. It performs best on
  promotional / phishing / giveaway-style spam and English text; out-of-domain
  messages (e.g. extortion threats, heavy obfuscation) may be missed.
- `skl2onnx` is sensitive to the `scikit-learn` version used at export time;
  keep them compatible (see `requirements-train.txt`).

## License

Model and datasets are MIT / CC-BY-4.0 respectively (see the upstream
[model card](https://huggingface.co/M-Arjun/SpamShield)).
