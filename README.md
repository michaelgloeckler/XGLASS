## XGLASS: X-Ray, Gaze and Language Assisted Slice Sequencing for CT Volumes

This is the official implementation of **XGLASS**, accepted at **WACV 2027**.

![Pipeline Architecture](Final_pipeline.svg)

Vision-language models (VLMs) achieve remarkable performance on 2D medical images, such as X-rays or pathology images, while 3D VLMs that directly process CT/MRI volumes lag behind due to limited 3D training data and the predominance of 2D foundation architectures. To leverage powerful 2D VLMs for analyzing medical volumes, which contain hundreds of slices, it is therefore necessary to focus only on those few slices, which are diagnostically relevant. To enable such a prioritization, we propose XGLASS, a method that transfers medical expertise from 2D X-rays to 3D CT for task-aware slice prioritization. XGLASS uses disease-conditioned importance maps derived from language-aligned radiologist gaze on X-rays and spatially aligns them with coronal CT projections, bringing gaze priors and CT content into a shared 2D representation. The fused gaze-prior/CT projection is then used to predict a disease-specific ordering of axial slices, prioritizing diagnostically relevant slices. XGLASS significantly improves budgeted retrieval over 2D and 3D-native baselines, achieving a top-10 hit rate up to 75.9% for the best-performing disease. In practice, this means that after inspecting only the top 10 slices (4.17% of a 240-slice scan), XGLASS identifies a diagnostically relevant slice in more than 75% of cases. For downstream slice-wise finding classification, an XGLASS-selected 30-slice candidate set reduces model calls by 87.5% while improving F1 by 11.3 percentage points over all-slice inference, 4.2 points over the strongest 2D selector, and remains competitive with 3D-native selectors while requiring up to 130.8× less time and 10.9× less GPU memory per finding, supporting its use as a lightweight preselector for volumetric CT analysis.

---

## Repository Structure

```text
xglass/
├── data/                   # dataset layout + preprocessing documentation
│   └── README.md           # full data preparation instructions
├── demo/                  # Jupyter notebooks for training & inference demos
├── src/xglass/             # model, training, and evaluation code
├── checkpoints/            # trained weights
├── requirements.txt
└── README.md
```

---

## Setup

### Environment

```bash
# From the repository root:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.11 and install a PyTorch build compatible with your GPU driver. Dependency versions and paper-result reproduction still need validation in a fresh environment.

---

## Data & Preprocessing

XGLASS relies on several publicly available datasets (REFLACX, ReXGroundingCT, CT-RATE).
Due to licensing and privacy restrictions, **raw datasets are not included; the demo notebooks contain illustrative outputs**.

➡️ **All dataset download instructions, folder layouts, and preprocessing steps are documented in:** `data/README.md`


That document describes how to:

* Download all required datasets from their original sources
* Extract disease-conditioned gaze heatmaps from REFLACX
* Prepare CT volumes and chest bounding boxes
* Create the final projection dataset consisting of coronal CT projections (AIP + Beer–Lambert) and slice-level relevance targets

---

### Code Structure (`src/xglass`)

The core XGLASS implementation is located in `src/xglass/`:

* `dataset.py`
  PyTorch dataset utilities for loading CT projections, slice-level supervision, and disease-conditioned gaze prior fusing.

* `model.py`
  Vision Transformer–based slice relevance model that predicts a continuous relevance
  score for each axial slice.

* `train.py`
  Training entry point. Handles configuration loading, training loops, checkpointing,
  and logging.

* `eval.py`
  Evaluation entry point. Computes ROC-AUC, Spearman correlation, and Hit@K metrics.

---
## 🧪 Demos

We provide **two demo notebooks** that illustrate the use of XGLASS for slice relevance prediction and how to train our model. The demos showcase **the inference behavior of our best-performing checkpoint** as well as the **training and data-loading pipeline** used in our experiments.

⚠️ **Data access & licensing.**
XGLASS relies on several third-party medical datasets (e.g., REFLACX, ReXGroundingCT). Due to privacy and licensing constraints, **raw datasets are not included**; the notebooks contain illustrative outputs. The demo notebooks **do not download datasets or request credentials**. They assume that all required datasets have been obtained through their official channels and prepared locally according to: `data/README.md`

---

### Demo 1 — Inference and Visualization (Best Checkpoint)

This demo illustrates **end-to-end inference using our best-performing XGLASS checkpoint**. It loads the projection dataset, fuses CT projections with disease-conditioned gaze priors, predicts a continuous relevance score for each axial slice in a CT volume, and plots a visualisation of the results.

The notebook visualizes:

1. The model input (CT projection fused with the gaze prior),
2. The slice-level relevance target used during training,
3. The axial CT slice corresponding to the predicted top-ranked index.

This demo reflects the inference pipeline used in the paper and provides a direct view of how XGLASS prioritizes diagnostically relevant slices.

---

### Demo 2 — Training and Data Loading

The second demo demonstrates **how to set up and train the XGLASS model**. It instantiates the data module and model using the same configuration as in our experiments, verifying correct loading of the projection dataset, gaze importance maps, and split definitions.

This notebook is intended to clarify the expected data format and configuration prior to launching full training runs.

---

## Training

Train the main XGLASS model (example: ViT-Base with AIP-Wide input):

```bash
python src/xglass/train.py \
  --data_root data/processed/projection_dataset/ \
  --heatmap_dir data/processed/gaze_importance_maps/ \
  --splits_json splits/paper_splits.json \
  --projection_type aip_wide \
  --blend_alpha 0.75 \
  --target_mode area \
  --model_name vit_base_patch16_224.augreg_in21k \
  --max_epochs 20 \
  --batch_size 16 \
  --num_workers 4 \
  --accelerator gpu \
  --devices 1 \
  --checkpoint_dir checkpoints/vit_base_patch16_224.augreg_in21k_aip_wide_alpha0.75 \
  --exp_name xglass_vitb_aipwide_a075 \
  --wandb_project XGLASS
```

**Logs (W&B offline):**
This script runs Weights & Biases in **offline mode** (`WANDB_MODE=offline`). Logs are saved locally under `wandb_logs/` by default (or under `$WANDB_DIR` if defined). To upload runs later, login once and sync:

```bash
wandb login
wandb sync wandb_logs/
```

**Checkpoints:**
Model checkpoints are written to `--checkpoint_dir` (e.g., `checkpoints/`). The training keeps only the **selected** checkpoint and uses it for the final `trainer.test(..., ckpt_path="best")` evaluation.

---

## Checkpoints

Pretrained XGLASS weights will be linked here. A download link is not available yet.
Place the checkpoint in
`checkpoints/vit_base_patch16_224.augreg_in21k_aip_wide_alpha0.75/`
and set its path in the inference notebook or ranking-export command.

## Evaluation

Evaluate a trained checkpoint:

```bash
python src/xglass/eval.py \
  --checkpoints_root checkpoints \
  --data_root data/processed/projection_dataset/ \
  --heatmap_dir data/processed/gaze_importance_maps/ \
  --splits_json splits/paper_splits.json \
  --out_csv results/evaluation_results.csv \
  --batch_size 16 \
  --num_workers 8
```

Reported metrics include:

* AR@10 (macro recall over CT--finding cases with positive target slices)
* ROC-AUC
* Spearman’s rank correlation
* Hit@K (K ∈ {1, 3, 5, 10, 20})

---

## Downstream Slice-wise VLM Screening

Run cached slice-wise screening with MedGemma or Lingshu and evaluate
XGLASS Top-K subsets against all slices and other selectors. See
[the downstream instructions](docs/downstream.md) for CT preparation,
prediction generation, and matched-budget evaluation commands.

## Paper Split

Use `--splits_json splits/paper_splits.json` for the paper cohort in training,
evaluation and downstream commands. The small local split used by the demo
notebooks is for illustration and does not reproduce the full paper cohort.
The paper split must be used with the matching complete projection manifest.

## Notes on Supervision

The paper's main model regresses normalized slice-area targets derived from
ReXGroundingCT finding masks during preprocessing. Those masks are not model
inputs. The released preprocessing also uses lung/heart segmentations to
construct chest bounding boxes; these are distinct from finding masks.
Binary and Gaussian slice targets are supported as alternative supervision.

## Contact

For questions or issues, please open a GitHub issue.

---


## License

Project code is available under the [MIT License](LICENSE), permitting use,
modification and redistribution, including commercial use, with the license
and copyright notice retained. Third-party datasets, model weights and
medical-image examples remain subject to their own terms; see
[third-party notices](THIRD_PARTY_NOTICES.md).
