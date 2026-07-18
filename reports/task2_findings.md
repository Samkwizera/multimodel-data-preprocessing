 # Task 2 Findings: Image Data Collection and Processing

**Author:** Kyla Nyaboke Ochweri
**Task:** Face Image Collection, Augmentation, and Feature Extraction

---

## 1. Overview

This task involved collecting facial images for all four group members, applying augmentation techniques, extracting image features, and saving them for use in the facial recognition model (Task 4). Images were collected in three expressions per member — **neutral**, **smiling**, and **surprised** — giving a total of 12 raw images across the group.

---

## 2. Image Collection

Each group member submitted three images:

| Member | Neutral | Smiling | Surprised |
|---|---|---|---|
| Member 1 | member1_neutral.jpeg | member1_smile.jpeg | member1_surprised.jpeg |
| Member 2 | member2_neutral.jpeg | member2_smile.jpeg | member2_surprised.jpeg |
| Member 3 | member3_neutral.jpeg | member3_smile.jpeg | member3_surprised.jpeg |
| Member 4 | member4_neutral.png | member4_smile.png | member4_surprised.png |

Images were stored in `data/raw/images/` in JPEG and PNG format. All 12 images were successfully loaded using OpenCV with BGR-to-RGB conversion applied during loading to ensure correct colour rendering in Matplotlib.

---

## 3. Display of Sample Images

Sample images were displayed using Matplotlib (2×3 grid, first 6 images) to visually verify correct loading, colour space conversion, and image quality before proceeding to augmentation and feature extraction.

---

## 4. Image Augmentation

Three augmentation functions were implemented using OpenCV:

| Augmentation | Description |
|---|---|
| **Rotation** | Rotates the image by 15° about its centre using an affine transformation matrix |
| **Horizontal Flip** | Mirrors the image along the vertical axis (`cv2.flip`) |
| **Grayscale Conversion** | Converts RGB image to grayscale (`cv2.cvtColor`) |

Augmented versions were demonstrated visually for one sample image (alongside the original) in a 1×4 subplot display. The augmentation functions are also used within `src/face_model.py`, which applies all augmentation variants to every image during model training to increase the effective training set size.

---

## 5. Feature Extraction

A normalised grayscale histogram with **256 bins** was extracted from each image as the feature representation. The approach:

1. Convert each image to grayscale.
2. Compute a 256-bin histogram using `cv2.calcHist`.
3. Normalise the histogram with `cv2.normalize` so values are scale-invariant.
4. Flatten and store each histogram as a row in a DataFrame.

This produces a 256-dimensional feature vector per image, capturing the overall pixel intensity distribution — a computationally efficient and interpretable representation suitable for the small dataset size.

---

## 6. Output: `image_features.csv`

The extracted features were saved to `data/processed/image_features.csv`.

| Property | Value |
|---|---|
| Rows | 12 (one per image) |
| Columns | 257 (`filename` + `histogram_bin_0` … `histogram_bin_255`) |
| Feature type | Normalised grayscale histogram (256 bins) |

The `face_model.py` training script regenerates a richer version of this file at training time, adding `member_id` and `augmentation` columns and producing 72 rows (12 images × 6 augmentation variants per image).

---

## 7. Tools and Libraries

| Library | Version | Purpose |
|---|---|---|
| OpenCV (`cv2`) | ≥ 4.x | Image loading, augmentation, histogram extraction |
| NumPy | ≥ 1.24 | Array operations |
| Pandas | ≥ 2.0 | DataFrame construction and CSV export |
| Matplotlib | ≥ 3.7 | Image display and augmentation visualisation |

---

## 8. Conclusion

All 12 images were successfully collected, loaded, and displayed. Three augmentation techniques (rotation, horizontal flip, grayscale) were implemented and demonstrated. Normalised grayscale histogram features were extracted from all images and saved to `image_features.csv` for downstream use in facial recognition model training.