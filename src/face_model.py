"""Facial recognition model for authorized group members.

Extracts color-histogram + downsampled pixel features from face images,
trains a Random Forest identity classifier, evaluates it, and saves the
artifact used by the CLI demo.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/raw/images"
FEATURES_OUT = ROOT / "data/processed/image_features.csv"
MODEL_OUT = ROOT / "models/face_model.pkl"

# closed-set classifiers always pick *someone*; require high confidence and a
# clear margin over the runner-up so unknown faces are rejected
CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.20
HIST_BINS = 16
RESIZE = (64, 64)


def _member_id(path: Path) -> str:
    # member4_neutral.png and member4_smile.png both map to member4
    match = re.match(r"(member\d+)", path.stem.replace("-", "_"), flags=re.I)
    if not match:
        raise ValueError(f"cannot parse member id from {path.name}")
    return match.group(1).lower()


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def extract_features(img: Image.Image) -> dict[str, float]:
    """Histogram + small embedding (flattened grayscale pixels)."""
    resized = img.resize(RESIZE)
    arr = np.asarray(resized, dtype=np.float32)

    feats: dict[str, float] = {}
    for i, channel in enumerate("rgb"):
        hist, _ = np.histogram(arr[:, :, i], bins=HIST_BINS, range=(0, 256), density=True)
        for b, v in enumerate(hist):
            feats[f"hist_{channel}_{b}"] = float(v)

    gray = np.asarray(ImageOps.grayscale(resized), dtype=np.float32) / 255.0
    # keep the embedding compact for a tiny dataset
    small = np.asarray(ImageOps.grayscale(resized).resize((16, 16)), dtype=np.float32) / 255.0
    for i, v in enumerate(small.flatten()):
        feats[f"embed_{i}"] = float(v)

    feats["mean_intensity"] = float(gray.mean())
    feats["std_intensity"] = float(gray.std())
    return feats


def augmentations(img: Image.Image) -> list[tuple[str, Image.Image]]:
    """Produce labeled variants so each member has enough training rows."""
    variants = [
        ("original", img),
        ("rotate15", img.rotate(15, expand=False, fillcolor=(0, 0, 0))),
        ("rotate_neg15", img.rotate(-15, expand=False, fillcolor=(0, 0, 0))),
        ("flip", ImageOps.mirror(img)),
        ("grayscale", ImageOps.colorize(ImageOps.grayscale(img), black="black", white="white")),
        ("bright", ImageEnhance.Brightness(img).enhance(1.3)),
    ]
    return variants


def build_feature_table(image_dir: Path = IMAGE_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        member = _member_id(path)
        img = load_image(path)
        for aug_name, aug_img in augmentations(img):
            feats = extract_features(aug_img)
            feats["member_id"] = member
            feats["source_file"] = path.name
            feats["augmentation"] = aug_name
            rows.append(feats)
    df = pd.DataFrame(rows)
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_OUT, index=False)
    print(f"wrote {FEATURES_OUT} ({len(df)} rows, {df['member_id'].nunique()} members)")
    return df


def _xy(df: pd.DataFrame):
    y = df["member_id"]
    X = df.drop(columns=["member_id", "source_file", "augmentation"])
    return X, y


def evaluate_loo(model_factory, X: pd.DataFrame, y: pd.Series):
    """Leave-one-out is the fair eval when each identity only has a few photos."""
    loo = LeaveOneOut()
    y_true, y_pred, y_proba = [], [], []
    classes = sorted(y.unique())
    for train_idx, test_idx in loo.split(X):
        clf = model_factory()
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = clf.predict_proba(X.iloc[test_idx])[0]
        # align probability columns to the global class order
        full = np.zeros(len(classes))
        for i, c in enumerate(clf.classes_):
            full[classes.index(c)] = proba[i]
        y_true.append(y.iloc[test_idx].iloc[0])
        y_pred.append(classes[int(np.argmax(full))])
        y_proba.append(full)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    loss = log_loss(y_true, y_proba, labels=classes)
    print(f"face model (LOO)  acc={acc:.3f}  macro F1={f1:.3f}  log loss={loss:.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc, f1, loss


def train(image_dir: Path = IMAGE_DIR) -> dict:
    df = build_feature_table(image_dir)
    # train on originals + augmentations; evaluate LOO on originals only so
    # the metric reflects identity recognition rather than recognizing an
    # augmentation of the same photo
    originals = df[df["augmentation"] == "original"].reset_index(drop=True)
    X_all, y_all = _xy(df)
    X_orig, y_orig = _xy(originals)

    def factory():
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, class_weight="balanced"
        )

    print(f"training rows: {len(df)} (incl. augmentations); LOO on {len(originals)} originals\n")
    evaluate_loo(factory, X_orig, y_orig)

    model = factory()
    model.fit(X_all, y_all)
    payload = {
        "model": model,
        "columns": list(X_all.columns),
        "classes": list(model.classes_),
        "threshold": CONFIDENCE_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_OUT)
    print(f"wrote {MODEL_OUT}")
    return payload


def load_model(path: Path = MODEL_OUT) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"face model not found at {path}; run: python -m src.face_model")
    return joblib.load(path)


def predict_identity(image_path: str | Path, model_bundle: dict | None = None) -> dict:
    """Return predicted member id, confidence, and whether access is granted."""
    bundle = model_bundle or load_model()
    img = load_image(Path(image_path))
    feats = extract_features(img)
    X = pd.DataFrame([feats])[bundle["columns"]]
    proba = bundle["model"].predict_proba(X)[0]
    order = np.argsort(proba)[::-1]
    idx = int(order[0])
    member = bundle["model"].classes_[idx]
    confidence = float(proba[idx])
    second = float(proba[order[1]]) if len(order) > 1 else 0.0
    margin = confidence - second
    recognized = (
        confidence >= bundle.get("threshold", CONFIDENCE_THRESHOLD)
        and margin >= bundle.get("margin_threshold", MARGIN_THRESHOLD)
    )
    return {
        "member_id": member if recognized else "unknown",
        "raw_prediction": member,
        "confidence": confidence,
        "margin": margin,
        "recognized": recognized,
        "probabilities": dict(zip(bundle["model"].classes_, map(float, proba))),
    }


if __name__ == "__main__":
    train()
