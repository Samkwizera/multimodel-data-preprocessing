"""Create synthetic unauthorized face/voice samples for the CLI demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/raw/unauthorized"
FACE_OUT = OUT_DIR / "unknown_face.png"
VOICE_OUT = OUT_DIR / "unauthorized_voice.wav"
SR = 22050


def make_unknown_face(path: Path = FACE_OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    # high-frequency synthetic pattern — deliberately unlike the selfie photos
    yy, xx = np.mgrid[0:256, 0:256]
    base = ((xx * 13 + yy * 7) % 255).astype(np.uint8)
    noise = rng.integers(0, 80, size=(256, 256), dtype=np.uint8)
    channel = np.clip(base.astype(np.int16) + noise - 40, 0, 255).astype(np.uint8)
    arr = np.stack([channel, 255 - channel, (channel // 2 + 60)], axis=-1)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 236, 236), outline=(255, 0, 0), width=4)
    draw.line((20, 20, 236, 236), fill=(255, 255, 0), width=3)
    draw.line((236, 20, 20, 236), fill=(255, 255, 0), width=3)
    img.save(path)
    print(f"wrote {path}")
    return path


def make_unauthorized_voice(path: Path = VOICE_OUT, seconds: float = 1.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    # tone burst + noise — not a spoken approval phrase from a member
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * rng.normal(size=t.shape)
    audio = (tone + noise).astype(np.float32)
    sf.write(path, audio, SR)
    print(f"wrote {path}")
    return path


def main():
    make_unknown_face()
    make_unauthorized_voice()


if __name__ == "__main__":
    main()
