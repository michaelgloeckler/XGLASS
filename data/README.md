

# Data Preparation

This project relies on several publicly available medical imaging and gaze datasets.
Download the datasets from the official sources below and preprocess them
locally. The demo notebooks include example outputs.

This document describes:

1. How to obtain the required datasets
2. The expected folder structure
3. How to run the preprocessing scripts to reproduce all inputs used by XGLASS

---

## 1. Required Datasets

### 1.1 REFLACX (Radiologist Gaze + Reports)

Precomputed gaze priors are included in `data/processed/gaze_importance_maps/`; regeneration from REFLACX is optional.

REFLACX provides eye-tracking data and synchronized report transcriptions for chest X-rays.

* Source: PhysioNet
* URL: [https://physionet.org/content/reflacx/](https://physionet.org/content/reflacx/)
* Access requires:

  * A PhysioNet account
  * Acceptance of the REFLACX data use agreement

After approval, download and extract the dataset locally.

If you use the MIMIC-Eye distribution, REFLACX is included as a subset.
The preprocessing scripts assume a **REFLACX-style folder layout**, not the full MIMIC pipeline.

* MIMIC-Eye URL: [https://physionet.org/content/mimic-eye/](https://physionet.org/content/mimic-eye/)

Expected structure:

```text
data/
├── reflacx/
│   ├── main_data/
│   │   ├── <SESSION_ID>/
│   │   │   ├── chest_bounding_box.csv
│   │   │   ├── timestamps_transcription.csv
│   │   │   └── (other REFLACX files)
│   └── gaze_data/
│       └── <SESSION_ID>/
│           └── gaze.csv
```

Disease-conditioned gaze heatmaps for **14 diseases** are already provided;
the paper used **six** of these diseases. Recreating the heatmaps is optional.
If you want to recreate them from REFLACX, download the dataset in the
structure above and run:

```bash
python data/REFLACX/extract_importance_maps.py --root data/reflacx --out data/processed/gaze_importance_maps
```

This step produces **raw, unsmoothed gaze importance maps** stored as NumPy files.

---
### Download CTs and Segmentations with Python

Accept the access terms for [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) and [ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT), then download the paper-split data:

```bash
pip install -r requirements-download.txt
hf auth login
# Download all paper-split CTs and segmentations
python data/download_datasets.py
```

Manual downloads are also supported, as described below.

---

### 1.2 ReXGroundingCT (3D Segmentations)

ReXGroundingCT provides 3D segmentation masks aligned with chest CT scans and is used to derive weak slice-level supervision.

* Source: Hugging Face
* URL: [https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT)

After downloading, organize the dataset as follows:

```text
data/
├── ReXGroundingCT/
│   ├── segmentations/
│   │   └── <CT_ID>.nii.gz
│   └── dataset.json
```

The segmentation masks are **not used directly** by XGLASS.
They are only used during preprocessing to emulate weak slice-level annotations by summing finding-mask area over the in-plane dimensions for each axial slice.

---

### 1.3 CT-RATE (Chest CT Volumes)

CT-RATE provides the chest CT volumes corresponding to the ReXGroundingCT segmentations.

* Source: Hugging Face
* URL: [https://huggingface.co/datasets/ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)

Note: Only the CTs matching the segmentations of ReXGroundingCT are required (not the whole dataset).

Download the CT volumes following the official CT-RATE instructions and place them into the `cts/` directory of the ReXGroundingCT folder, ensuring filenames match the segmentation masks:

```text
data/
├── ReXGroundingCT/
│   ├── cts/
│   │   └── <CT_ID>.nii.gz
│   ├── segmentations/
│   │   └── <CT_ID>.nii.gz
│   └── dataset.json
```

#### Chest Bounding Boxes (required)

XGLASS uses chest bounding boxes for normalization and to match CTs and importance maps.
After placing the CT volumes into `ReXGroundingCT/cts/`, create chest bounding boxes using `extract_chestbboxes.py` (TotalSegmentator-based):

```bash
python data/ReXGroundingCT/extract_chestbboxes.py --root data/ReXGroundingCT
```

This will generate one JSON per CT scan:

```text
data/
├── ReXGroundingCT/
│   ├── cts/
│   │   └── <CT_ID>.nii.gz
│   ├── ChestBBoxes/
│   │   └── <CT_ID>_chest_bbox.json
│   ├── segmentations/
│   │   └── <CT_ID>.nii.gz
│   └── dataset.json
```

Notes:

* The script runs TotalSegmentator on each CT volume and unions selected thoracic structures (lungs + heart) to obtain a robust chest region.
* Bounding boxes are computed on a canonical RAS + isotropic (default: 1.25mm) grid and stored in the JSON alongside shape and spacing metadata.
* If a JSON already exists, the script skips the CT (resume-friendly).

---

## 2. Create the final projection dataset (AIP + Beer–Lambert)

After you have:

* CT volumes in `data/ReXGroundingCT/cts/`
* segmentations in `data/ReXGroundingCT/segmentations/`
* chest bounding boxes in `data/ReXGroundingCT/ChestBBoxes/`
* a manifest JSON (e.g. `filtered_dataset.json` or `dataset.json`)
* the paper split file (`splits/paper_splits.json`)

you can generate the **model-ready projection dataset** (fixed-slice-block) containing:

* `aip_wide.png`, `aip_lung.png`, `aip_soft.png`
* `bl_cxr.png` (Beer–Lambert CXR-like projection)
* `gt_data.json` (+ human-readable CSV)
* `bbox_256.json`
* `manifest.csv` (paths for loading)

### Command

```bash
python data/create_projection_dataset.py \
  --data_root data/ReXGroundingCT \
  --input_json dataset.json \
  --splits_json splits/paper_splits.json \
  --heatmap_dir data/processed/gaze_importance_maps \
  --out_dir data/processed/projection_dataset \
  --target_spacing_mm 1.25 \
  --target_z_slices 240 \
  --target_xy_dim 256
```

### Output structure

```text
data/processed/projection_dataset/
├── manifest.csv
├── data/
│   ├── train/
│   │   ├── <category>/
│   │   │   ├── <ct_stem>/
│   │   │   │   ├── aip_wide.png
│   │   │   │   ├── aip_lung.png
│   │   │   │   ├── aip_soft.png
│   │   │   │   ├── bl_cxr.png
│   │   │   │   ├── gt_data.json
│   │   │   │   ├── gt_data_human.csv
│   │   │   │   └── bbox_256.json
│   ├── val/ ...
│   └── test/ ...
└── overlays/ (optional)
```

### Notes

* The script saves **raw projections**. Any additional smoothing/augmentation is applied downstream (training time).
* If `--heatmap_dir` is provided and contains REFLACX heatmaps, the script can generate a small number of overlay PNGs for sanity checks (`--overlay_count`).

---

## Notes on Data Usage

* Obtain raw CT volumes, gaze recordings, and segmentation datasets from their official sources.
* All preprocessing must be done in accordance with the original dataset licenses.
* Check the source dataset terms before redistributing generated artifacts or examples; the code license does not replace those terms.


## Paths and reproducibility

Run commands from the repository root. Use `splits/paper_splits.json` for
projection creation, training, evaluation and the notebooks. The projection
builder accepts this repository-relative path; input dataset JSON paths can
also be resolved relative to `--data_root`. Use the matching complete manifest
for paper-cohort training and evaluation. For smaller demo datasets, use a
split file matching the selected CTs.

Install TotalSegmentator separately for chest-box generation:

```bash
pip install TotalSegmentator
```

The chest-box script uses the standard-resolution model and requests only
the lung lobes and heart. It does not enable fast mode.
