#!/usr/bin/env python3
"""Multimodal CLI demo: face -> voice -> product recommendation.

Flow (Task 4 / Member 4):
  1. Input a face image  -> facial recognition
  2. If recognized       -> prompt for voice sample
  3. If voice approved   -> run product recommendation
  4. Otherwise           -> ACCESS DENIED

Also supports an unauthorized demo (unknown face + unauthorized voice).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_model import predict_identity  # noqa: E402
from src.voice_model import verify_voice  # noqa: E402

MERGED = ROOT / "data/processed/merged_dataset.csv"
PRODUCT_MODEL = ROOT / "models/product_model.pkl"
DEFAULT_FACE = ROOT / "data/raw/images/member1_neutral.jpeg"
DEFAULT_VOICE = ROOT / "data/raw/audio/member1_approve.ogg"
UNAUTHORIZED_FACE = ROOT / "data/raw/unauthorized/unknown_face.png"
UNAUTHORIZED_VOICE = ROOT / "data/raw/unauthorized/unauthorized_voice.wav"


def load_product_model(path: Path = PRODUCT_MODEL):
    if not path.exists():
        raise FileNotFoundError(
            f"product model missing at {path}. Run: python -m src.product_model"
        )
    return joblib.load(path)


def recommend_for_customer(customer_id: int | None = None) -> dict:
    """Score one customer row from the merged dataset with the product model."""
    bundle = load_product_model()
    df = pd.read_csv(MERGED)
    drop = ["transaction_id", "customer_id", "purchase_date", "product_category"]

    if customer_id is None:
        # pick a customer that has a social profile for a more interesting demo
        row = df[df["has_social_profile"] == 1].iloc[0]
    else:
        matches = df[df["customer_id"] == customer_id]
        if matches.empty:
            raise ValueError(f"customer_id {customer_id} not found in merged dataset")
        row = matches.iloc[0]

    X = pd.get_dummies(pd.DataFrame([row.drop(labels=drop)]), drop_first=True)
    # align to training columns
    for col in bundle["columns"]:
        if col not in X.columns:
            X[col] = 0
    X = X[bundle["columns"]]

    model = bundle["model"]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    top = sorted(zip(model.classes_, proba), key=lambda t: t[1], reverse=True)[:3]
    return {
        "customer_id": int(row["customer_id"]),
        "prediction": pred,
        "top_probabilities": {c: float(p) for c, p in top},
        "actual_category": row["product_category"],
    }


def run_transaction(face_path: Path, voice_path: Path, customer_id: int | None = None) -> int:
    print("=" * 60)
    print("USER IDENTITY + PRODUCT RECOMMENDATION SYSTEM")
    print("=" * 60)

    print(f"\n[1/3] Facial recognition\n  image: {face_path}")
    face = predict_identity(face_path)
    print(f"  raw prediction : {face['raw_prediction']}")
    print(f"  confidence     : {face['confidence']:.3f}  (margin={face.get('margin', 0):.3f})")
    if not face["recognized"]:
        print("\n>>> ACCESS DENIED: face not recognized.")
        return 1
    print(f"  recognized as  : {face['member_id']}")

    print(f"\n[2/3] Voiceprint verification\n  audio: {voice_path}")
    voice = verify_voice(voice_path, expected_member=face["member_id"])
    print(f"  raw prediction : {voice['raw_prediction']}")
    print(f"  confidence     : {voice['confidence']:.3f}  (margin={voice.get('margin', 0):.3f})")
    if not voice["approved"]:
        print("\n>>> ACCESS DENIED: voice not approved "
              f"(expected {face['member_id']}, got {voice['raw_prediction']}).")
        return 1
    print(f"  approved as    : {voice['member_id']}")

    print("\n[3/3] Product recommendation")
    result = recommend_for_customer(customer_id)
    print(f"  customer_id    : {result['customer_id']}")
    print(f"  prediction     : {result['prediction']}")
    print(f"  actual (label) : {result['actual_category']}")
    print("  top classes    :")
    for cls, p in result["top_probabilities"].items():
        print(f"    - {cls}: {p:.3f}")

    print("\n>>> TRANSACTION APPROVED — recommendation displayed.")
    return 0


def run_unauthorized_demo() -> int:
    print("=" * 60)
    print("UNAUTHORIZED ATTEMPT DEMO")
    print("=" * 60)

    if not UNAUTHORIZED_FACE.exists() or not UNAUTHORIZED_VOICE.exists():
        print("Unauthorized samples missing. Generate them with:")
        print("  python -m src.make_unauthorized_samples")
        return 2

    print("\n--- Attempt A: unknown face ---")
    code_a = run_transaction(UNAUTHORIZED_FACE, DEFAULT_VOICE)

    print("\n--- Attempt B: authorized face, unauthorized voice ---")
    # face of member1 but voice that is not an approved speaker sample
    code_b = run_transaction(DEFAULT_FACE, UNAUTHORIZED_VOICE)

    print("\n" + "=" * 60)
    print("Unauthorized demo finished.")
    print(f"  unknown face denied : {'yes' if code_a == 1 else 'NO (unexpected)'}")
    print(f"  bad voice denied    : {'yes' if code_b == 1 else 'NO (unexpected)'}")
    print("=" * 60)
    return 0 if code_a == 1 and code_b == 1 else 1


def interactive() -> int:
    print("Multimodal auth CLI. Press Enter to accept defaults.\n")
    face = input(f"Face image path [{DEFAULT_FACE}]: ").strip() or str(DEFAULT_FACE)
    voice = input(f"Voice sample path [{DEFAULT_VOICE}]: ").strip() or str(DEFAULT_VOICE)
    cust = input("Customer id for recommendation [auto]: ").strip()
    customer_id = int(cust) if cust else None
    return run_transaction(Path(face), Path(voice), customer_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Face + voice gate before product recommendation"
    )
    p.add_argument("--face", type=Path, help="Path to face image")
    p.add_argument("--voice", type=Path, help="Path to voice sample")
    p.add_argument("--customer-id", type=int, default=None,
                   help="Customer id from merged dataset for product prediction")
    p.add_argument("--unauthorized-demo", action="store_true",
                   help="Run unknown-face and unauthorized-voice simulations")
    p.add_argument("--interactive", action="store_true",
                   help="Prompt for face/voice paths")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.unauthorized_demo:
        return run_unauthorized_demo()

    if args.interactive or (args.face is None and args.voice is None):
        if args.face or args.voice:
            face = args.face or DEFAULT_FACE
            voice = args.voice or DEFAULT_VOICE
            return run_transaction(face, voice, args.customer_id)
        return interactive()

    face = args.face or DEFAULT_FACE
    voice = args.voice or DEFAULT_VOICE
    return run_transaction(face, voice, args.customer_id)


if __name__ == "__main__":
    raise SystemExit(main())
