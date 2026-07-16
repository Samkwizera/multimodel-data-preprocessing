"""Train all three models used by the multimodal CLI."""

from __future__ import annotations

from src import face_model, product_model, voice_model
from src.make_unauthorized_samples import main as make_unauthorized


def main():
    print("\n=== Product recommendation model ===")
    product_model.train()

    print("\n=== Facial recognition model ===")
    face_model.train()

    print("\n=== Voiceprint verification model ===")
    voice_model.train()

    print("\n=== Unauthorized demo samples ===")
    make_unauthorized()

    print("\nAll models trained. Run the CLI with:")
    print("  python -m src.main_app --face data/raw/images/member1_neutral.jpeg "
          "--voice data/raw/audio/member1_approve.ogg")
    print("  python -m src.main_app --unauthorized-demo")


if __name__ == "__main__":
    main()
