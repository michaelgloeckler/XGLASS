#!/usr/bin/env python3
"""Recreate axial CT crops from the projection dataset's saved crop metadata."""
import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--splits_json", type=Path, required=True)
    parser.add_argument("--ct_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--spacing_mm", type=float, default=1.25)
    args = parser.parse_args()
    test = set(json.loads(args.splits_json.read_text(encoding="utf-8"))["test"])
    with (args.data_root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["ct_name"] in test]
    if not rows or args.spacing_mm <= 0:
        parser.error("A nonempty test split and positive spacing are required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for row in rows:
        meta = args.data_root / row.get("bbox_json_path", str(Path(row["data_path"]) / "bbox_256.json"))
        bounds = json.loads(meta.read_text(encoding="utf-8"))["crop_window_iso"]
        stem = row["ct_stem"]
        if stem in seen:
            if seen[stem] != bounds:
                raise ValueError(f"Inconsistent crop metadata across findings: {stem}")
            continue
        seen[stem] = bounds
        sources = list(args.ct_dir.rglob(row["ct_name"]))
        if len(sources) != 1:
            raise ValueError(f"Expected one source CT for {row['ct_name']}, found {len(sources)}")
        ct = nib.as_closest_canonical(nib.load(str(sources[0])))
        iso = resample_to_output(ct, voxel_sizes=(args.spacing_mm,) * 3, order=1)
        x0, x1, z0, z1 = map(int, bounds)
        if not (0 <= x0 < x1 <= iso.shape[0] and 0 <= z0 < z1 <= iso.shape[2] and z1 - z0 == 240):
            raise ValueError(f"Invalid 240-slice crop for {stem}: {bounds}")
        crop = iso.get_fdata(dtype=np.float32)[x0:x1, :, z0:z1]
        affine = iso.affine.copy()
        affine[:3, 3] = (iso.affine @ [x0, 0, z0, 1])[:3]
        dest = args.out_dir / f"{stem}_cropped.nii.gz"
        nib.save(nib.Nifti1Image(crop, affine), dest)
        print(dest)


if __name__ == "__main__":
    main()
