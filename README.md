# Multimodal Data Preprocessing — Formative 2

A multimodal authentication and product recommendation system built for the ALU Machine Learning course. The system authenticates a user through facial recognition and voice verification before serving a personalised product recommendation — matching the flow in the assignment diagram.

---

## System Flow

```
Start
  → Facial Recognition Model
      → FAIL: Access Denied
      → PASS: Run Product Recommendation Model
                → Voice Validation Model
                      → FAIL: Access Denied
                      → PASS: Display Predicted Product
```

Access is only granted when **both** biometric checks pass and the face identity matches the voice identity.

---

## Project Structure

```
multimodel-data-preprocessing-main/
│
├── data/
│   ├── raw/
│   │   ├── audio/                    # Voice recordings (2 per member)
│   │   ├── images/                   # Face images (3 per member)
│   │   ├── unauthorized/             # Unknown face + unauthorized voice for demo
│   │   ├── customer_social_profiles.csv
│   │   └── customer_transactions.csv
│   └── processed/
│       ├── merged_dataset.csv        # Task 1: merged + feature-engineered dataset
│       ├── image_features.csv        # Task 2: face feature embeddings
│       └── audio_features.csv        # Task 3: MFCC + spectral audio features
│
├── models/
│   ├── face_model.pkl                # Trained facial recognition model
│   ├── voice_model.pkl               # Trained voiceprint verification model
│   └── product_model.pkl             # Trained product recommendation model
│
├── notebooks/
│   ├── eda.ipynb                     # Task 1: exploratory data analysis
│   ├── image_processing.ipynb        # Task 2: face image pipeline
│   └── audio_processing.ipynb        # Task 3: audio preprocessing pipeline
│
├── src/
│   ├── preprocessing.py              # Task 1: merge + feature engineering
│   ├── product_model.py              # Task 1: product recommendation model
│   ├── face_model.py                 # Task 4: facial recognition model
│   ├── voice_model.py                # Task 4: voiceprint verification model
│   ├── train_models.py               # Task 4: trains all three models
│   ├── make_unauthorized_samples.py  # Task 4: generates unauthorized demo inputs
│   ├── main_app.py                   # Task 4: CLI system demo
│   └── __init__.py
│
├── reports/
│   ├── task1_findings.md
│   ├── task3_findings.md
│   └── task4_findings.md
│
├── requirements.txt
└── README.md
```

---

## Tasks and Contributors

| Task | Description | Member |
|---|---|---|
| **Task 1** | Data merge, EDA, product recommendation model | Samuel Kwizera Ihimbazwe |
| **Task 2** | Face image collection, augmentation, feature extraction | Member 2 |
| **Task 3** | Audio collection, augmentation, feature extraction | Tyrus (Member 4) |
| **Task 4** | Model training, system integration, CLI demo | Member 4 |

---

## Setup

**Requirements:** Python 3.10+

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd multimodel-data-preprocessing-main

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### Train all models

```bash
python -m src.train_models
```

This runs the full pipeline in order:
1. Merges the tabular datasets and trains the product model
2. Extracts image features and trains the face model
3. Extracts audio features and trains the voice model
4. Generates unauthorized demo samples

### Run the CLI demo

**Authorized transaction** (supply a known face + matching voice):

```bash
python -m src.main_app \
  --face data/raw/images/member4_smile.png \
  --voice data/raw/audio/member4_approve.ogg
```

**Unauthorized demo** (unknown face + unauthorized voice):

```bash
python -m src.main_app --unauthorized-demo
```

**Interactive mode** (prompted for file paths at runtime):

```bash
python -m src.main_app --interactive
```

---

## Data

### Tabular (Task 1)

Two datasets are merged on customer ID:

| Dataset | Rows | Key columns |
|---|---|---|
| `customer_social_profiles.csv` | 155 | `customer_id_new`, `engagement_score`, `social_media_platform`, `review_sentiment` |
| `customer_transactions.csv` | 150 | `customer_id_legacy`, `purchase_amount`, `product_category`, `customer_rating` |

The merge collapses the social table to one row per customer before joining, keeping all 150 transactions intact. The final merged dataset is 150 rows × 18 columns with no nulls.

### Image (Task 2)

3 images per member (neutral, smiling, surprised) → augmented → histogram features saved to `image_features.csv`.

### Audio (Task 3)

2 recordings per member:

| Phrase | File pattern |
|---|---|
| *"Yes, approve"* | `memberN_approve.ogg` |
| *"Confirm transaction"* | `memberN_confirm.ogg` |

5 augmentations per sample (time stretch fast/slow, pitch shift up/down, background noise) → 48 rows total in `audio_features.csv`.

---

## Models

| Model | Algorithm | Features | Evaluation |
|---|---|---|---|
| **Facial Recognition** | Random Forest | Histogram embeddings from `image_features.csv` | LOO on 12 originals: Acc 1.00, F1 1.00 |
| **Voiceprint Verification** | Random Forest | MFCCs, spectral roll-off, energy, ZCR from `audio_features.csv` | LOO on 8 originals: Acc 0.375, F1 0.325 |
| **Product Recommendation** | Random Forest | Merged tabular features | Grouped CV: ~0.22 (near majority-class baseline) |

> The voice model's modest LOO score reflects having only 2 short phrases per speaker. Inference on known members is confident and the CLI enforces that face and voice identity must match, so the system still rejects impostors.

> The product model performs near the majority-class baseline (23.3%), which is consistent with the EDA finding that none of the available features are statistically significant predictors of product category — a property of the dataset, not the pipeline.

---

## Audio Features (Task 3 — Tyrus)

Feature extraction is implemented from scratch using `numpy` and `scipy` (no librosa):

| Feature group | Columns | Description |
|---|---|---|
| MFCCs | `mfcc_mean_0` – `mfcc_std_12` | Mean and std of 13 mel-frequency cepstral coefficients |
| Spectral roll-off | `spectral_rolloff_mean`, `_std` | Frequency below which 85% of energy lies |
| Spectral centroid | `spectral_centroid_mean` | Perceptual brightness of the voice |
| Energy (RMS) | `energy_mean`, `energy_std` | Loudness and variation across frames |
| Zero-crossing rate | `zcr_mean` | Noisiness and fricative content |
| Duration | `duration_sec` | Phrase length in seconds |

**33 feature columns, 48 rows** (4 members × 2 phrases × 6 augmentation variants).

---

## Demo Video

> _Add link after recording:_
> `demo_video_link:`

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
```