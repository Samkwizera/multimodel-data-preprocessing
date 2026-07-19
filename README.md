# Formative 2: Multimodal Data Preprocessing

**Course:** Machine Learning — African Leadership University  
**Group:** 21

---

## What This Project Does

This project builds a **User Identity and Product Recommendation System**. Before a customer can receive a product recommendation, they must pass two biometric checks:

1. **Facial recognition** — the system confirms the person's face matches a known group member
2. **Voice verification** — the system confirms the voice sample matches the same identity

Only after both checks pass does the system run the product recommendation model and display a predicted product. If either check fails, access is denied.

```
Start
  → [1] Facial Recognition Model
        ├── FAIL → Access Denied
        └── PASS → [2] Run Product Recommendation Model
                        → [3] Voice Validation Model
                              ├── FAIL → Access Denied
                              └── PASS → Display Predicted Product
```

---

## Team Members and Contributions

| Name | Task |
|---|---|
| Samuel Kwizera Ihimbazwe | Task 1 — Data merge, EDA, and product recommendation model |
| Kyla Nyaboke Ochweri | Task 2 — Face image collection, augmentation, and feature extraction |
| Ajak Bul Zacharia Chol | Task 3 — Audio collection, augmentation, and feature extraction |
| Berissa Muyizere | Task 4 — Model training, system integration, and CLI demo |

---

## Repository Contents

```
multimodel-data-preprocessing-main/
│
├── data/
│   ├── raw/
│   │   ├── audio/                        # Voice recordings — 2 per member (8 total)
│   │   │   ├── member1_approve.ogg       # "Yes, approve"
│   │   │   ├── member1_confirm.ogg       # "Confirm transaction"
│   │   │   ├── member2_approve.ogg
│   │   │   ├── member2_confirm.ogg
│   │   │   ├── member3_approve.ogg
│   │   │   ├── member3_confirm.ogg
│   │   │   ├── member4_approve.ogg
│   │   │   └── member4_confirm.ogg
│   │   ├── images/                       # Face images — 3 per member (12 total)
│   │   │   ├── member1_neutral.jpeg      # neutral, smiling, surprised per member
│   │   │   ├── member1_smile.jpeg
│   │   │   ├── member1_surprised.jpeg
│   │   │   ├── member2_neutral.jpeg  ...
│   │   │   ├── member3_neutral.jpeg  ...
│   │   │   └── member4_neutral.png   ...
│   │   ├── unauthorized/
│   │   │   ├── unknown_face.png          # Synthetic unknown face for demo
│   │   │   └── unauthorized_voice.wav    # Unauthorized voice sample for demo
│   │   ├── customer_social_profiles.csv  # Raw social media data (155 rows)
│   │   └── customer_transactions.csv     # Raw transaction data (150 rows)
│   │
│   └── processed/
│       ├── merged_dataset.csv            # Task 1 output — merged + feature-engineered
│       ├── image_features.csv            # Task 2 output — grayscale histogram features
│       └── audio_features.csv            # Task 3 output — MFCC + spectral features
│
├── models/
│   ├── face_model.pkl                    # Trained facial recognition model
│   ├── voice_model.pkl                   # Trained voiceprint verification model
│   └── product_model.pkl                 # Trained product recommendation model
│
├── notebooks/
│   ├── eda.ipynb                         # Task 1 — EDA and merge walkthrough
│   ├── image_processing.ipynb            # Task 2 — image pipeline with outputs
│   └── audio_processing.ipynb            # Task 3 — audio pipeline with outputs
│
├── src/
│   ├── preprocessing.py                  # Task 1 — merge and feature engineering
│   ├── product_model.py                  # Task 1 — product recommendation model
│   ├── face_model.py                     # Task 4 — facial recognition pipeline
│   ├── voice_model.py                    # Task 4 — voiceprint verification pipeline
│   ├── train_models.py                   # Task 4 — trains all three models at once
│   ├── main_app.py                       # Task 4 — CLI system demo
│   ├── make_unauthorized_samples.py      # Task 4 — generates unauthorized demo files
│   └── __init__.py
│
├── reports/
│   ├── task1_findings.md                 # Detailed write-up: merge logic and EDA
│   ├── task2_findings.md                 # Detailed write-up: image pipeline
│   ├── task3_findings.md                 # Detailed write-up: audio pipeline
│   └── task4_findings.md                 # Detailed write-up: models and CLI
│
├── requirements.txt
└── README.md
```

---

## Setup Instructions

**Requirements:** Python 3.10 or higher

```bash
# Clone the repository
git clone https://github.com/Samkwizera/multimodel-data-preprocessing.git
cd multimodel-data-preprocessing

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Train all models

```bash
python -m src.train_models
```

This runs the full pipeline in order:
1. Merges the tabular datasets and trains the product recommendation model
2. Extracts image features and trains the facial recognition model
3. Extracts audio features and trains the voiceprint verification model
4. Generates the unauthorized demo samples

### Step 2 — Run the system demo

**Authorized transaction** (recognized face + matching voice → product displayed):

```bash
python -m src.main_app \
  --face data/raw/images/member4_neutral.png \
  --voice data/raw/audio/member4_approve.ogg
```

**Unauthorized demo** (unknown face + unauthorized voice → access denied at each gate):

```bash
python -m src.main_app --unauthorized-demo
```

**Interactive mode** (prompts for file paths at runtime):

```bash
python -m src.main_app --interactive
```

---

## Datasets

### Task 1 — Tabular Data (`merged_dataset.csv`)

Two raw datasets are merged into one clean dataset for the product recommendation model:

| Dataset | Rows | Description |
|---|---|---|
| `customer_social_profiles.csv` | 155 | Customer engagement scores, social platforms, review sentiment |
| `customer_transactions.csv` | 150 | Transaction amounts, product categories, customer ratings |

**Why a direct join doesn't work:** Both tables have multiple rows per customer. A direct join inflates 150 transactions into 219 rows by creating impossible transaction duplicates. The social table is first collapsed to one row per customer (mean engagement, modal platform, modal sentiment), then joined onto the transaction table — preserving all 150 real transactions.

**Feature engineering added:**

| Feature | Description |
|---|---|
| `purchase_month` | Month extracted from purchase date |
| `purchase_dayofweek` | Day of week extracted from purchase date |
| `customer_txn_count` | How many transactions this customer has made |
| `customer_avg_amount` | This customer's average spend |
| `amount_vs_customer_avg` | Ratio of this transaction to the customer's average |
| `has_social_profile` | Flag for customers with no social data (vs. filled median) |
| `rating_missing` | Flag for imputed customer ratings |

**Final dataset:** 150 rows × 18 columns, zero nulls, one row per transaction.

### Task 2 — Image Data (`image_features.csv`)

Each of the 4 group members submitted 3 face images (neutral, smiling, surprised), giving 12 images total stored in `data/raw/images/`.

**Augmentations applied per image:**
- Rotation (±15°)
- Horizontal flip
- Grayscale conversion

**Features extracted:** Normalised grayscale histogram (256 bins) per image.

**Output:** `data/processed/image_features.csv` — 12 rows × 258 columns (`filename`, `member_id`, `histogram_bin_0` … `histogram_bin_255`).

### Task 3 — Audio Data (`audio_features.csv`)

Each of the 4 group members recorded 2 voice samples, giving 8 recordings total stored in `data/raw/audio/`:

| Phrase | File pattern |
|---|---|
| *"Yes, approve"* | `memberN_approve.ogg` |
| *"Confirm transaction"* | `memberN_confirm.ogg` |

All files were resampled to 22 050 Hz mono at load time (members 1 & 4 recorded at 16 kHz; members 2 & 3 at 48 kHz).

**Augmentations applied per sample (5 per sample):**

| Augmentation | Technique |
|---|---|
| Time stretch (fast) | Resampled to 1.15× speed |
| Time stretch (slow) | Resampled to 0.85× speed |
| Pitch shift up | Resample shorter, then restore original length |
| Pitch shift down | Resample longer, then restore original length |
| Background noise | Additive Gaussian noise (σ = 0.005) |

**Features extracted per sample:**

| Feature | Columns | Description |
|---|---|---|
| MFCCs | `mfcc_mean_0` – `mfcc_std_12` (26 cols) | Mean and std of 13 mel-frequency cepstral coefficients |
| Spectral roll-off | `spectral_rolloff_mean`, `_std` | Frequency below which 85% of energy lies |
| Spectral centroid | `spectral_centroid_mean` | Perceptual brightness of the voice |
| Energy (RMS) | `energy_mean`, `energy_std` | Loudness and variation across frames |
| Zero-crossing rate | `zcr_mean` | Noisiness and fricative content |
| Duration | `duration_sec` | Phrase length in seconds |

**Output:** `data/processed/audio_features.csv` — 48 rows × 36 columns (4 members × 2 phrases × 6 variants, 33 feature columns + `member_id`, `source_file`, `augmentation`).

---

## Models

### Facial Recognition Model (`src/face_model.py` → `models/face_model.pkl`)

- **Algorithm:** Random Forest (200 trees, balanced class weights)
- **Features:** RGB colour histograms (16 bins × 3 channels) + 16×16 grayscale pixel embedding = 304 features
- **Training data:** 12 original images × 6 augmentation variants = 72 rows
- **Evaluation:** Leave-one-out on the 12 original images only (no augmentation data leakage)
- **Threshold logic:** Prediction requires confidence ≥ 0.70 AND margin over second-best ≥ 0.20; otherwise returns "unknown" and denies access

| Metric | Score |
|---|---|
| LOO Accuracy | 1.000 |
| Macro F1 | 1.000 |
| Log Loss | 0.359 |

### Voiceprint Verification Model (`src/voice_model.py` → `models/voice_model.pkl`)

- **Algorithm:** Random Forest (200 trees, balanced class weights)
- **Features:** 33 acoustic features from `audio_features.csv`
- **Training data:** 8 original recordings × 6 augmentation variants = 48 rows
- **Evaluation:** Leave-one-out on the 8 original recordings only
- **Identity check:** Voice prediction must match the face identity from Step 1; mismatches deny access even if the voice is recognized

| Metric | Score |
|---|---|
| LOO Accuracy | 0.375 |
| Macro F1 | 0.325 |
| Log Loss | 1.108 |

> The modest LOO score reflects having only 2 short phrases per speaker. On inference, known members achieve high confidence (member4: 0.985). The CLI's face–voice identity match check and confidence thresholds make the system reject impostors even with the limited training data.

### Product Recommendation Model (`src/product_model.py` → `models/product_model.pkl`)

- **Algorithm:** Random Forest
- **Features:** All numeric and categorical features from `merged_dataset.csv`, excluding `transaction_id`, `customer_id`, `purchase_date`, and the target `product_category`
- **Evaluation:** GroupKFold cross-validation (grouped by `customer_id` to prevent data leakage through spend aggregates)
- **Target classes:** Electronics, Sports, Clothing, Groceries, Books

| Metric | Score |
|---|---|
| Grouped CV Accuracy | ~0.22 |
| Majority-class baseline | 0.233 |

> The model performs at the majority-class baseline, which is expected: statistical testing (ANOVA and chi-square) finds no significant relationship between any available feature and product category. This is a property of the dataset, not the pipeline — the same code would find real signal if the data contained any.

---

## System Demonstration

The CLI in `src/main_app.py` implements the full authentication and recommendation flow:

```
[1/3] Facial recognition
  raw prediction : member4
  confidence     : 1.000  (margin=1.000)
  recognized as  : member4

[2/3] Voiceprint verification
  raw prediction : member4
  confidence     : 0.985  (margin=0.970)
  approved as    : member4

[3/3] Product recommendation
  customer_id    : 151
  prediction     : Sports
  actual (label) : Sports
  top classes    :
    - Sports: 0.640
    - Clothing: 0.160
    - Books: 0.120

>>> TRANSACTION APPROVED — recommendation displayed.
```

**Unauthorized demo** tests two denial paths:
- Unknown face → denied at Step 1
- Authorized face + unauthorized voice → denied at Step 2

---

## Demo Video

> https://youtu.be/v71tJky0QRU

---

## Dependencies

```
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
pillow>=10.0
numpy>=1.24
scipy>=1.11
soundfile>=0.12
matplotlib>=3.7
opencv-python>=4.0
```
