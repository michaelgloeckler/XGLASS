#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checkpoint Evaluation Script for Slice Predictor

This script evaluates ALL trained model checkpoints found within a root
directory, parses their settings from the folder name, and runs them
on the test set.

It saves everything to a final CSV file and prints a summary table.
"""

import argparse
from pathlib import Path
import sys
import re
import csv
import pandas as pd
import pytorch_lightning as pl

# Import our custom classes
from model import SlicePredictor
from dataset import SliceDataModule


def log(m): 
    print(f"[Eval Log] {m}", flush=True)


def parse_exp_name(exp_name: str) -> dict:
    """
    Parses the experiment folder name to get the settings.

    Expected pattern includes one of:
      ..._aip_wide_...
      ..._aip_lung_...
      ..._aip_soft_...

    Optional:
      alpha0.75
      frozen
    """
    log(f"Parsing experiment name: {exp_name}")

    projection_type = None
    model_tag = None
    rest = ""

    if "_aip_wide_" in exp_name:
        projection_type = "aip_wide"
        parts = exp_name.split("_aip_wide_")
        model_tag, rest = parts[0], parts[1]
    elif "_aip_lung_" in exp_name:
        projection_type = "aip_lung"
        parts = exp_name.split("_aip_lung_")
        model_tag, rest = parts[0], parts[1]
    elif "_aip_soft_" in exp_name:
        projection_type = "aip_soft"
        parts = exp_name.split("_aip_soft_")
        model_tag, rest = parts[0], parts[1]

    elif "_bl_cxr_" in exp_name:
        projection_type = "bl_cxr"
        model_tag, rest = exp_name.split("_bl_cxr_", 1)

    if not projection_type:
        raise ValueError(f"Could not parse projection_type from {exp_name}")

    # Recreate the model name from the tag
    model_name = model_tag.replace('__', ':').replace('hf_hub:', 'hf_hub:')

    # Parse alpha if present
    alpha = 0.0
    if "alpha" in rest:
        m = re.search(r'alpha(\d+\.\d+)', rest)
        if m:
            alpha = float(m.group(1))
        else:
            log(f"WARN: Could not parse alpha from '{rest}', defaulting to 0.0")

    is_frozen = "frozen" in rest

    settings = {
        "model_name": model_name,
        "projection_type": projection_type,
        "blend_alpha": alpha,
        "backbone_frozen": is_frozen
    }
    log(f"Parsed settings: {settings}")
    return settings


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained Slice Predictor checkpoints.")

    parser.add_argument('--checkpoints_root', type=str, default="./checkpoints",
                        help='Path to the ROOT checkpoint directory (e.g., ./checkpoints)')
    parser.add_argument('--data_root', type=str, default='data/processed/projection_dataset',
                        help='Root directory of the final dataset')
    parser.add_argument('--heatmap_dir', type=str, default='data/processed/gaze_importance_maps',
                        help='Directory with raw REFLACX heatmaps')
    parser.add_argument('--splits_json', type=str, default='splits/paper_splits.json',
                        help='Path or name of the splits file (e.g., new_balanced_splits.json)')
    parser.add_argument('--out_csv', type=str, default='evaluation_results.csv',
                        help='Output CSV file to save results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Set to 0 for local testing")

    args = parser.parse_args()

    root_ckpt_dir = Path(args.checkpoints_root)
    if not root_ckpt_dir.is_dir():
        log(f"ERROR: Root checkpoint directory not found: {root_ckpt_dir}")
        sys.exit(1)

    all_results = []
    failed_experiments = 0
    log("--- Starting Evaluation of Trained Models ---")

    for exp_dir in sorted(root_ckpt_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        log(f"\n--- Evaluating Experiment: {exp_name} ---")

        try:
            ckpts = sorted(exp_dir.glob("*.ckpt"))
            if not ckpts:
                log(f"WARN: No .ckpt file found in {exp_dir}, skipping.")
                continue
            ckpt_path = ckpts[0]
            log(f"Found checkpoint file: {ckpt_path}")

            settings = parse_exp_name(exp_name)

            # Splits path: either absolute/relative file, or inside data_root
            splits_json_path = Path(args.splits_json)
            if not splits_json_path.is_file():
                splits_json_path = Path(args.data_root) / args.splits_json
            if not splits_json_path.is_file():
                log(f"ERROR: splits_json not found: {args.splits_json} (also tried {splits_json_path})")
                failed_experiments += 1
                continue

            datamodule = SliceDataModule(
                data_root=args.data_root,
                heatmap_dir=args.heatmap_dir,
                splits_json=str(splits_json_path),
                projection_type=settings["projection_type"],
                blend_alpha=settings["blend_alpha"],
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

            log("Loading model from checkpoint...")
            # cache_dir=None overrides any saved hparams cache path
            model = SlicePredictor.load_from_checkpoint(str(ckpt_path), cache_dir=None, pretrained=False)

            trainer = pl.Trainer(
                accelerator='auto',
                devices=1,
                logger=False
            )

            log("Running test set...")
            test_results = trainer.test(model, datamodule=datamodule, verbose=False)
            if not test_results or not test_results[0]:
                raise RuntimeError("Evaluation produced no test metrics")
            metrics = test_results[0]

            row = {"experiment": exp_name}
            row.update(metrics)
            all_results.append(row)

            log(f"Successfully evaluated {exp_name}.")

        except Exception as e:
            failed_experiments += 1
            log(f"ERROR: Failed to evaluate {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        log("No results to save. Exiting.")
        sys.exit(1)

    # Collect metric keys from all runs (union), stable order
    metric_keys = sorted({k for r in all_results for k in r.keys() if k.startswith("test_")})
    fieldnames = ["experiment"] + metric_keys

    out_csv_path = Path(args.out_csv)
    log(f"\nWriting final results to {out_csv_path}")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for r in all_results:
            csv_row = {}
            for k in fieldnames:
                if k not in r:
                    csv_row[k] = ""
                    continue
                v = r[k]
                if k == "experiment":
                    csv_row[k] = str(v)
                else:
                    if hasattr(v, "item"):
                        v = v.item()
                    try:
                        csv_row[k] = f"{float(v):.6f}"
                    except Exception:
                        csv_row[k] = str(v)
            writer.writerow(csv_row)

    # Console summary
    log("\n--- FINAL RESULTS ---")
    df = pd.DataFrame(all_results).set_index("experiment")

    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)

    cols_preferred = [
        "test_auc_epoch", "test_spearman_epoch",
        "test_hit_rate_1_epoch", "test_hit_rate_3_epoch",
        "test_hit_rate_5_epoch", "test_hit_rate_10_epoch",
        "test_hit_rate_20_epoch", "test_ar_10_epoch", "test_loss"
    ]
    cols_to_show = [c for c in cols_preferred if c in df.columns]
    if not cols_to_show:
        cols_to_show = list(df.columns)

    sort_key = "test_auc_epoch" if "test_auc_epoch" in df.columns else cols_to_show[0]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df[cols_to_show].sort_values(by=sort_key, ascending=False))

    log("--- Evaluation Complete ---")
    if failed_experiments:
        log(f"ERROR: {failed_experiments} experiment(s) failed; saved successful results only.")
        sys.exit(1)


if __name__ == "__main__":
    main()
