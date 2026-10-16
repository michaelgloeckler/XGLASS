# Downstream Slice-wise Finding Screening

Each CT--finding pair is screened one axial slice at a time. Predictions are
cached once, then reused for every selector and slice budget. Precision,
recall and F1 are computed over inspected slices only, as in the paper.
They are not whole-volume diagnostic sensitivity. All selectors use the same
test cases and cached downstream predictions.

Run commands from the repository root. Prepare the projection dataset, raw
gaze maps and split JSON according to [data preparation](../data/README.md).
Obtain access to the selected downstream model through its official source.

## 1. Install Inference Dependencies

```bash
pip install -r requirements-downstream.txt
```

For Lingshu in 4-bit mode, additionally install `bitsandbytes`. Choose a
PyTorch build compatible with your GPU. Installing these optional dependencies
is unnecessary when evaluating already cached predictions.

For BioMedCLIP inference, additionally install `open_clip_torch`.

## 2. Prepare Aligned Axial CT Crops

This command uses each projection's saved `crop_window_iso`, canonicalizes and
resamples the raw CT to 1.25 mm, then saves the matching 240 axial slices.
Use the same spacing as projection preprocessing. Existing aligned crops can
be reused directly. The output is named `<ct_stem>_cropped.nii.gz`.

```bash
python data/prepare_downstream_cts.py \
  --data_root data/processed/projection_dataset \
  --splits_json splits/paper_splits.json \
  --ct_dir data/ReXGroundingCT/cts \
  --out_dir data/processed/cropped_test_cts
```

## 3. Export XGLASS Rankings

Use your checkpoint path below. Each ranking is a permutation of the
zero-based axial indices 0--239. Rankings are keyed by both CT and finding.

```bash
python src/xglass/export_orders.py \
  --ckpt_path checkpoints/vit_base_patch16_224.augreg_in21k_aip_wide_alpha0.75/epoch=06.ckpt \
  --out_orders_dir results/orders/xglass
```

The default data paths match steps 2 and 4. Override `--data_root`,
`--heatmap_dir` and `--splits_json` when using a different local layout.

## 4. Generate Cached Slice Predictions

The prompt asks whether the queried finding is visible in one axial slice.
Decoding is deterministic. The main paper uses `multi_window_rgb`, with
wide, soft-tissue and mediastinal windows in separate channels. `wide_rgb`
replicates the wide window and is available for the rendering sensitivity
analysis. Keep rendering fixed within a comparison.

MedGemma:

```bash
python src/xglass/predict_slices.py \
  --model_id google/medgemma-4b-it \
  --slice_rendering multi_window_rgb \
  --dtype bfloat16 \
  --quantization none \
  --out_jsonl results/medgemma_slices.jsonl
```

Lingshu:

```bash
python src/xglass/predict_slices.py \
  --model_id lingshu-medical-mllm/Lingshu-7B \
  --slice_rendering multi_window_rgb \
  --dtype bfloat16 \
  --quantization 4bit \
  --out_jsonl results/lingshu_slices.jsonl
```

Re-running the same command resumes completed slice predictions from the
JSONL file. Use a separate output file for each model, rendering, or inference
configuration. A quick inference smoke test can use `--max_rows 1
--max_slices_per_row 2`; complete all predictions before full evaluation.

## 5. Evaluate Matched Budgets

```bash
python src/xglass/evaluate_downstream.py \
  --predictions results/medgemma_slices.jsonl \
  --data_root data/processed/projection_dataset \
  --splits_json splits/paper_splits.json \
  --orders XGLASS=results/orders/xglass \
  --include_random \
  --k 30 60 120 180 \
  --out_summary results/medgemma_downstream_summary.csv \
  --out_records results/medgemma_downstream_records.csv
```

Repeat with the Lingshu prediction file. The evaluator automatically includes
the all-240-slice endpoint. Summary CSVs contain confusion counts, inspected
slice counts, precision, recall, F1 and accuracy; record CSVs contain counts
for each CT--finding case. Unknown VLM answers count as negative and are also
reported separately.

To compare another selector, add `--orders NAME=PATH`. Each folder must contain
`<safe_ct_stem>__<safe_finding>.txt`, or `<safe_ct_stem>.txt` for a
disease-agnostic selector. Each file must contain one ranked index per line,
covering every axial index exactly once. The exporter replaces characters
outside letters, digits, `.`, `_` and `-` with `_` in filenames.

Incomplete predictions fail by default. `--allow_incomplete` explicitly
restricts every selector to the same complete cases and prints their count;
such runs do not reproduce the full test cohort. Missing or malformed
selector orders and mismatched ground-truth labels always fail.

## BioMedCLIP Screening

BioMedCLIP uses the paper's five-prompt ensemble and a wide-window slice
rendering. Cache its binary decisions with a fixed similarity threshold:

```bash
python src/xglass/predict_biomedclip.py \
  --threshold 0.3751 \
  --out_jsonl results/biomedclip_slices.jsonl
```

Then run the same evaluator with `--predictions results/biomedclip_slices.jsonl`.
The default threshold is the rounded threshold reported in the paper. To
reproduce exact cached results, use the full-precision threshold from the
original run. For a new experiment, choose a threshold on a separate
validation cohort and keep it fixed across all slice budgets and selectors;
the script does not optimize a threshold on test labels. Do not use
BioMedCLIP retrieval as a selector in its own downstream screening comparison.

Top-30 inspects 12.5% of the slices and therefore uses 87.5% fewer downstream
slice calls than all-slice screening. This call-count reduction does not
include XGLASS preprocessing or ranking time.
