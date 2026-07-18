# Task 3 – Sound Data Collection and Processing

**Member:** Ajak Bul Zacharia Chol

---

## Overview

Task 3 covers the full audio preprocessing pipeline for the multimodal authentication system. The goal was to collect voice samples from each group member, visualise them as waveforms and spectrograms, apply data augmentation, and extract discriminative acoustic features into `audio_features.csv` for the Voiceprint Verification Model in Task 4.

All code lives in `notebooks/audio_processing.ipynb`. Feature extraction is also importable via `src/voice_model.py`.

---

## Audio Samples Collected

Each member contributed **two real voice recordings**:

| File | Phrase | Member 4 duration |
|---|---|---|
| `member4_approve.ogg` | *"Yes, approve"* | 2.62 s |
| `member4_confirm.ogg` | *"Confirm transaction"* | 4.35 s |

All four members supplied recordings at different sample rates (members 1 & 4 at 16 kHz; members 2 & 3 at 48 kHz). All files were resampled to **22 050 Hz mono float32** at load time for a consistent pipeline.

---

## Visualisations

### Waveforms
Waveforms were plotted for all four members across both phrases (8 panels). All recordings are non-silent and span 2.4–4.4 seconds. Member 4's recordings show a quiet but clear speech envelope with natural amplitude variation across syllables.

### Spectrograms
Mel spectrograms (2048-point FFT, 512-sample hop, 26 mel filters, `magma` colourmap) confirm:
- Voiced segments show harmonic banding in the 0–3 kHz range.
- Silence gaps between words appear as dark (low-energy) regions.
- Member 4's spectrograms reveal real speech formant structure, distinguishing them clearly from the other members.

---

## Augmentations Applied

Five augmentations were applied to every (member × phrase) pair, giving **6 variants per sample**:

| Augmentation | Technique | Rationale |
|---|---|---|
| `stretch_fast` | Resample to 1.15× speed | Simulates hurried or stressed speech |
| `stretch_slow` | Resample to 0.85× speed | Simulates deliberate, slow speech |
| `pitch_up` | Resample shorter → restore original length | Voice sounds slightly higher-pitched |
| `pitch_down` | Resample longer → restore original length | Voice sounds slightly lower-pitched |
| `noise` | Additive Gaussian noise (σ = 0.005) | Models background noise in real environments |

---

## Feature Extraction

Features extracted per sample using `numpy` and `scipy` only (no librosa):

| Feature | Columns | Description |
|---|---|---|
| MFCCs | `mfcc_mean_0` – `mfcc_std_12` (26 cols) | Mean and std of 13 mel-frequency cepstral coefficients |
| Spectral roll-off | `spectral_rolloff_mean`, `_std` | Frequency below which 85% of spectral energy lies |
| Spectral centroid | `spectral_centroid_mean` | Weighted mean frequency (perceptual brightness) |
| Energy (RMS) | `energy_mean`, `energy_std` | Frame-level loudness and variation |
| Zero-crossing rate | `zcr_mean` | Proportion of sign changes per frame |
| Duration | `duration_sec` | Total phrase length in seconds |

**Total: 33 acoustic feature columns per sample.**

### Implementation
- Framing: 2048-sample Hann-windowed frames, 512-sample hop
- Mel filterbank: 26 triangular filters (0 Hz to Nyquist), built from scratch
- DCT-II applied to log-mel energies → 13 MFCCs per frame
- All features are scalar statistics (mean/std over frames) for a fixed-length vector

---

## Output: `audio_features.csv`

| Property | Value |
|---|---|
| Path | `data/processed/audio_features.csv` |
| Rows | 48 (4 members × 2 phrases × 6 variants) |
| Columns | 36 (33 features + `member_id`, `source_file`, `augmentation`) |
| Members | member1, member2, member3, member4 |
| Rows per member | 12 (perfectly balanced) |

---

## Figures Saved

| File | Description |
|---|---|
| `waveforms.png` | 4 × 2 grid of raw waveforms (all members × both phrases) |
| `spectrograms.png` | 4 × 2 grid of mel spectrograms |
| `augmentation_demo.png` | 6 waveform panels for member4_approve (all augmentations) |
| `augmentation_spectrograms.png` | 6 spectrogram panels for member4_approve |
| `feature_distributions.png` | Bar chart of 6 key features by member (original samples) |
| `mfcc_heatmap.png` | MFCC fingerprint heatmap (13 coefficients × 4 members) |
