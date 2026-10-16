#!/usr/bin/env python3
"""Export category-aware XGLASS slice orders for the held-out test split."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import SliceDataset
from model import SlicePredictor


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("Export category-aware XGLASS orders.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_root", default="data/processed/projection_dataset")
    parser.add_argument("--heatmap_dir", default="data/processed/gaze_importance_maps")
    parser.add_argument("--splits_json", default="splits/paper_splits.json")
    parser.add_argument("--projection_type", default="aip_wide")
    parser.add_argument("--blend_alpha", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_orders_dir", default="results/orders/xglass")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    splits_path = Path(args.splits_json)
    if not splits_path.is_file():
        splits_path = data_root / splits_path
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(data_root / "manifest.csv")
    test_df = manifest[manifest["ct_name"].isin(set(splits["test"]))].reset_index(drop=True)

    dataset = SliceDataset(
        data_root=data_root,
        manifest=test_df,
        heatmap_dir=Path(args.heatmap_dir),
        projection_type=args.projection_type,
        blend_alpha=args.blend_alpha,
        augment=False,
        target_mode="area",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlicePredictor.load_from_checkpoint(args.ckpt_path, cache_dir=None)
    model.eval().to(device)

    out_dir = Path(args.out_orders_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_offset = 0
    for images, _, _ in loader:
        preds = model(images.to(device)).detach().cpu().numpy()
        for i, scores in enumerate(preds):
            row = test_df.iloc[row_offset + i]
            order = np.argsort(-scores, kind="stable")
            out_path = out_dir / (
                f"{safe_filename(str(row['ct_stem']))}__"
                f"{safe_filename(str(row['category']))}.txt"
            )
            out_path.write_text(
                "\n".join(str(int(index)) for index in order) + "\n",
                encoding="utf-8",
            )
        row_offset += len(preds)

    print(f"Exported {row_offset} orders to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
