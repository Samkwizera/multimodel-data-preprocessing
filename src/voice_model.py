"""Voiceprint verification model for authorized group members.

Extracts MFCC-style / spectral roll-off / energy features from audio samples,
trains a Random Forest speaker classifier, evaluates it, and saves the
artifact used by the CLI demo.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data/raw/audio"
FEATURES_OUT = ROOT / "data/processed/audio_features.csv"
MODEL_OUT = ROOT / "models/voice_model.pkl"

CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.15
SR = 22050
N_MFCC = 13
N_FFT = 2048
HOP = 512
N_MELS = 26


def _member_id(path: Path) -> str:
    match = re.match(r"(member\d+)", path.stem.replace("-", "_"), flags=re.I)
    if not match:
        raise ValueError(f"cannot parse member id from {path.name}")
    return match.group(1).lower()


def load_audio(path: Path, sr: int = SR) -> tuple[np.ndarray, int]:
    y, file_sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr and len(y) > 0:
        n_out = max(1, int(round(len(y) * sr / file_sr)))
        y = resample(y, n_out).astype(np.float32)
    return y, sr


def _frame_signal(y: np.ndarray, frame_length: int = N_FFT, hop: int = HOP) -> np.ndarray:
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    n_frames = 1 + (len(y) - frame_length) // hop
    frames = np.stack(
        [y[i * hop : i * hop + frame_length] for i in range(n_frames)],
        axis=0,
    )
    window = get_window("hann", frame_length, fftbins=True).astype(np.float32)
    return frames * window


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS) -> np.ndarray:
    freqs = rfftfreq(n_fft, d=1.0 / sr)
    mels = np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_mels + 2)
    hz_points = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (right - j) / (right - center)
    return fb


def _mfcc(y: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    frames = _frame_signal(y)
    spectra = np.abs(rfft(frames, axis=1)) ** 2
    fb = _mel_filterbank(sr, N_FFT)
    mel = np.maximum(spectra @ fb.T, 1e-10)
    log_mel = np.log(mel)
    # DCT-II for MFCCs
    n = log_mel.shape[1]
    n_ = np.arange(n)
    k = np.arange(n_mfcc)[:, None]
    dct = np.cos(np.pi * (n_ + 0.5) * k / n)
    return (log_mel @ dct.T).T  # (n_mfcc, n_frames)


def extract_features(y: np.ndarray, sr: int = SR) -> dict[str, float]:
    if len(y) == 0:
        y = np.zeros(sr // 10, dtype=np.float32)

    mfcc = _mfcc(y, sr=sr)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    frames = _frame_signal(y)
    spectra = np.abs(rfft(frames, axis=1))
    freqs = rfftfreq(N_FFT, d=1.0 / sr)
    # spectral roll-off: frequency below which 85% of energy lies
    power = spectra ** 2
    cumsum = np.cumsum(power, axis=1)
    total = cumsum[:, -1][:, None] + 1e-10
    rolloff_idx = np.argmax(cumsum >= 0.85 * total, axis=1)
    rolloff = freqs[rolloff_idx]

    # spectral centroid
    centroid = (spectra * freqs).sum(axis=1) / (spectra.sum(axis=1) + 1e-10)

    # energy (RMS) and zero-crossing rate per frame
    rms = np.sqrt((frames ** 2).mean(axis=1))
    zcr = ((frames[:, 1:] * frames[:, :-1]) < 0).mean(axis=1)

    feats: dict[str, float] = {}
    for i, (m, s) in enumerate(zip(mfcc_mean, mfcc_std)):
        feats[f"mfcc_mean_{i}"] = float(m)
        feats[f"mfcc_std_{i}"] = float(s)
    feats["spectral_rolloff_mean"] = float(rolloff.mean())
    feats["spectral_rolloff_std"] = float(rolloff.std())
    feats["spectral_centroid_mean"] = float(centroid.mean())
    feats["energy_mean"] = float(rms.mean())
    feats["energy_std"] = float(rms.std())
    feats["zcr_mean"] = float(zcr.mean())
    feats["duration_sec"] = float(len(y) / sr)
    return feats


def augmentations(y: np.ndarray, sr: int = SR) -> list[tuple[str, np.ndarray]]:
    variants = [("original", y)]
    # time stretch via resample
    if len(y) > 10:
        fast = resample(y, int(len(y) / 1.15)).astype(np.float32)
        slow = resample(y, int(len(y) / 0.85)).astype(np.float32)
        variants.append(("stretch_fast", fast))
        variants.append(("stretch_slow", slow))
    # pitch-ish shift: resample then restore duration
    if len(y) > 10:
        shifted = resample(y, int(len(y) * 0.9)).astype(np.float32)
        shifted = resample(shifted, len(y)).astype(np.float32)
        variants.append(("pitch_up", shifted))
        shifted_down = resample(y, int(len(y) * 1.1)).astype(np.float32)
        shifted_down = resample(shifted_down, len(y)).astype(np.float32)
        variants.append(("pitch_down", shifted_down))
    # background noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.005, size=y.shape).astype(np.float32)
    variants.append(("noise", y + noise))
    return variants


def build_feature_table(audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(audio_dir.iterdir()):
        if path.suffix.lower() not in {".wav", ".ogg", ".mp3", ".flac", ".m4a"}:
            continue
        member = _member_id(path)
        y, sr = load_audio(path)
        for aug_name, aug_y in augmentations(y, sr):
            feats = extract_features(aug_y, sr)
            feats["member_id"] = member
            feats["source_file"] = path.name
            feats["augmentation"] = aug_name
            rows.append(feats)
    if not rows:
        raise FileNotFoundError(f"no audio files found in {audio_dir}")
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
    loo = LeaveOneOut()
    y_true, y_pred, y_proba = [], [], []
    classes = sorted(y.unique())
    for train_idx, test_idx in loo.split(X):
        clf = model_factory()
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = clf.predict_proba(X.iloc[test_idx])[0]
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
    print(f"voice model (LOO)  acc={acc:.3f}  macro F1={f1:.3f}  log loss={loss:.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc, f1, loss


def train(audio_dir: Path = AUDIO_DIR) -> dict:
    df = build_feature_table(audio_dir)
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
        raise FileNotFoundError(f"voice model not found at {path}; run: python -m src.voice_model")
    return joblib.load(path)


def verify_voice(audio_path: str | Path, expected_member: str | None = None,
                 model_bundle: dict | None = None) -> dict:
    """Verify speaker identity. Optionally require a match to expected_member."""
    bundle = model_bundle or load_model()
    y, sr = load_audio(Path(audio_path))
    feats = extract_features(y, sr)
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
    if expected_member is not None and member != expected_member:
        recognized = False
    return {
        "member_id": member if recognized else "unknown",
        "raw_prediction": member,
        "confidence": confidence,
        "margin": margin,
        "approved": recognized,
        "probabilities": dict(zip(bundle["model"].classes_, map(float, proba))),
    }


if __name__ == "__main__":
    train()
