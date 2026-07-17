# Task 4 - Models + System Demo

Member 4

## Scope

This task builds the three models required by the assignment and wires them into a
command-line simulation of the multimodal access flow:

1. Facial recognition
2. Voiceprint verification
3. Product recommendation (uses the merged tabular dataset from Task 1)

Access is granted only when face and voice both pass. An unauthorized demo covers
an unknown face and an unauthorized voice sample.

## Scripts

| File | Role |
|------|------|
| `src/face_model.py` | Extract image features, train/evaluate face model, save `models/face_model.pkl` |
| `src/voice_model.py` | Extract audio features, train/evaluate voice model, save `models/voice_model.pkl` |
| `src/product_model.py` | Product recommendation model on the merged dataset (from Task 1) |
| `src/main_app.py` | CLI: face → voice → product prediction, plus unauthorized demo |
| `src/train_models.py` | Train all models and generate unauthorized samples |
| `src/make_unauthorized_samples.py` | Synthetic unknown face + unauthorized voice |

Feature tables written during training:

- `data/processed/image_features.csv`
- `data/processed/audio_features.csv`

## Multimodal logic

```
input face image
  -> face model predicts member + confidence
  -> if confidence < threshold OR unknown: ACCESS DENIED
  -> else ask for voice
input voice sample
  -> voice model predicts member + confidence
  -> if confidence < threshold OR speaker != face identity: ACCESS DENIED
  -> else run product recommendation and display prediction
```

This matches the assignment flow: face gate, then voice gate, then product model.

## Evaluation

Each biometric model is evaluated with leave-one-out on original samples (not
augmentations), reporting accuracy, macro F1, and log loss. The product model
keeps the grouped holdout / metrics already implemented in `src/product_model.py`.

Observed on this repo's samples after training:

| Model | Accuracy | Macro F1 | Log loss |
|-------|----------|----------|----------|
| Face (LOO on 12 originals) | 1.000 | 1.000 | 0.359 |
| Voice (LOO on 8 originals, 4 members) | 0.375 | 0.325 | 1.108 |
| Product (see Task 1 / `product_model.py`) | grouped CV ~0.22 | — | reported in script |

Voice LOO stays modest because each speaker only has 2 short phrases, but
inference on member4 samples is strong (high confidence) and the CLI requires
face/voice identity to match. Face/voice inference also use confidence + margin
thresholds so unknown inputs are rejected even though the classifiers are
closed-set.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.train_models

# authorized transaction (member1 has both face and voice samples)
python -m src.main_app \
  --face data/raw/images/member1_neutral.jpeg \
  --voice data/raw/audio/member1_approve.ogg

# unauthorized attempts
python -m src.main_app --unauthorized-demo

# interactive prompts
python -m src.main_app --interactive
```

## Notes

- Member 4 face images are present (`member4-natural.png`, `member4_smile.png`,
  `member4_surprised.png`) and are included in face-model training.
- Member 4 voice samples are present (`member4_approve.ogg`,
  `member4_confirm.ogg`) and are included in voice-model training. An
  authorized end-to-end demo can use member4 for both modalities:
  `python -m src.main_app --face data/raw/images/member4_smile.png --voice data/raw/audio/member4_approve.ogg`
- Demo video link should be added to the group report / README after recording.

## Demo video

_Add link after recording:_

```
demo_video_link: <paste URL here>
```
