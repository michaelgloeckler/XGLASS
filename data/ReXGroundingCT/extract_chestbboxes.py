#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ChestBBoxes for ReXGroundingCT using TotalSegmentator (CLI).
- Inputs:   cts/*.nii.gz
- Outputs:  ChestBBoxes/<stem>_chest_bbox.json
- Method:   Run TS into a temp dir, union selected chest organs (lungs + heart),
            build bbox on a canonical RAS + isotropic (1.25mm) grid, save JSON.
- Resume:   Skips CTs whose JSON already exists.
"""

import os, sys, json, shutil, subprocess, argparse, tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to

# Which organ masks to union from TotalSegmentator outputs
CHEST_ORGANS = [
    # lungs (lobes)
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
    # heart
    "heart",
]

def run_totalseg_cli(ct_path: Path, out_dir: Path) -> None:
    """
    Run TotalSegmentator CLI to produce per-organ NIfTIs in out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use the standard-resolution model and retain the requested organ subset.
    cmd = [
        "TotalSegmentator",
        "-i", str(ct_path),
        "-o", str(out_dir),
        "-ta", "total",
    ]
    cmd += ["--roi_subset"] + CHEST_ORGANS
    print(f"[TS] {ct_path.name} -> {out_dir}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"TotalSegmentator failed for {ct_path.name}: {e}")

def load_or_fail(p: Path) -> nib.Nifti1Image:
    if not p.exists():
        raise FileNotFoundError(f"Missing organ mask: {p}")
    return nib.load(str(p))

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int,int,int,int,int,int]]:
    if not mask.any():
        return None
    x, y, z = np.where(mask)  # axis order matches array axis order
    return int(x.min()), int(x.max()), int(y.min()), int(y.max()), int(z.min()), int(z.max())

def process_one(ct_path: Path,
                out_root: Path,
                tmp_root: Path,
                target_spacing: float = 1.25,
                overwrite: bool = False) -> Optional[Path]:
    stem = ct_path.stem.replace(".nii", "")  # handle *.nii.gz
    json_path = out_root / f"{stem}_chest_bbox.json"
    if json_path.exists() and not overwrite:
        print(f"[SKIP] {json_path.name} exists")
        return json_path

    # 1) Canonical reference grid from CT
    ct_native = nib.load(str(ct_path))                         # native orientation
    ct_ras = nib.as_closest_canonical(ct_native)               # RAS canonical
    ref_iso = resample_to_output(ct_ras, voxel_sizes=(target_spacing,)*3, order=1)  # linear for image
    ref_shape = tuple(int(s) for s in ref_iso.shape)
    spacing_mm = tuple(float(s) for s in nib.affines.voxel_sizes(ref_iso.affine))

    # 2) Run TS (per-organ outputs) into temp dir
    tmp_out = tmp_root / stem
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True, exist_ok=True)
    try:
        run_totalseg_cli(ct_path, tmp_out)
    except Exception as e:
        print(f"[ERR] TS failed on {ct_path.name}: {e}")
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

    # 3) Union chest organs in native space, then wrap with CT affine
    try:
        union_native = None
        for organ in CHEST_ORGANS:
            fp = tmp_out / f"{organ}.nii.gz"
            organ_img = load_or_fail(fp)
            organ_mask = organ_img.get_fdata().astype(np.uint8) > 0
            if union_native is None:
                union_native = organ_mask
            else:
                union_native |= organ_mask
        if union_native is None:
            raise RuntimeError("No organ masks loaded.")
        # Ensure mask is 3D and matches CT shape (may differ by a voxel due to TS)
        if union_native.shape != ct_native.shape:
            # Safe intersect in case of tiny mismatches
            tgt = np.zeros(ct_native.shape, dtype=np.uint8)
            sz = tuple(min(a,b) for a,b in zip(union_native.shape, tgt.shape))
            tgt[:sz[0], :sz[1], :sz[2]] = union_native[:sz[0], :sz[1], :sz[2]]
            union_native = tgt

        union_img_native = nib.Nifti1Image(union_native.astype(np.uint8), ct_native.affine, ct_native.header)

        # 4) Resample union mask to ref_iso grid (nearest)
        union_iso = resample_from_to(union_img_native, ref_iso, order=0)
        union_mask = union_iso.get_fdata().astype(bool)

        # 5) Compute bbox on canonical grid
        bbox = bbox_from_mask(union_mask)
        if bbox is None:
            print(f"[WARN] Empty chest union after resample: {ct_path.name}")
            shutil.rmtree(tmp_out, ignore_errors=True)
            return None

        # 6) Save JSON
        json_path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "bbox_vox": list(map(int, bbox)),   # [x0,x1,y0,y1,z0,z1] in canonical grid
            "spacing_mm": list(map(float, spacing_mm)),
            "shape": list(map(int, ref_shape)),
            "source_ct": str(ct_path),
            "totalseg_version": "cli",
            "organs": CHEST_ORGANS,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"[OK] {json_path.name}")
        return json_path
    finally:
        # Clean temp organ files to save space
        shutil.rmtree(tmp_out, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser(description="Build ChestBBoxes (voxel bbox in canonical grid) for ReXGroundingCT.")
    ap.add_argument("--root", type=str, default="data/ReXGroundingCT",
                    help="ReXGroundingCT root (contains cts/)")
    ap.add_argument("--cts_dir", type=str, default=None,
                    help="Optional override for cts dir; default: <root>/cts")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Where to write ChestBBoxes; default: <root>/ChestBBoxes")
    ap.add_argument("--tmp_dir", type=str, default=None,
                    help="Scratch dir for TS outputs; default: <root>/_ts_tmp")
    ap.add_argument("--pattern", type=str, default="*.nii.gz",
                    help="Glob for CTs (default: *.nii.gz)")
    ap.add_argument("--target_spacing", type=float, default=1.25,
                    help="Isotropic spacing (mm) for canonical grid")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers (Python multiprocessing)")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if JSON exists")
    ap.add_argument("--reverse", action="store_true", help="Process CTs in reverse order")
    ap.add_argument("--shard_idx", type=int, default=0, help="Shard index (0-based)")
    ap.add_argument("--shard_n",   type=int, default=1, help="Total number of shards")
    args = ap.parse_args()

    root = Path(args.root)
    cts_dir = Path(args.cts_dir) if args.cts_dir else (root / "cts")
    out_dir = Path(args.out_dir) if args.out_dir else (root / "ChestBBoxes")
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else (root / "_ts_tmp")

    cts = sorted(cts_dir.glob(args.pattern))
    if not cts:
        print(f"No CTs found in: {cts_dir} (pattern {args.pattern})")
        sys.exit(1)

    # sharding and reverse ordering
    if args.reverse:
        cts = list(reversed(cts))
    if args.shard_n > 1:
        cts = cts[args.shard_idx::args.shard_n]
    print(f"[INFO] after sharding idx={args.shard_idx}/{args.shard_n}, reverse={args.reverse}: {len(cts)} CTs")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] tmp_dir: {tmp_dir} (will be cleaned per-CT)")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.workers <= 1:
        for i, ct in enumerate(cts, 1):
            print(f"\n[{i}/{len(cts)}] {ct.name}")
            process_one(ct, out_dir, tmp_dir, target_spacing=float(args.target_spacing),
                        overwrite=bool(args.overwrite))
    else:
        # MP pool
        import functools
        from multiprocessing import Pool
        fn = functools.partial(process_one,
                               out_root=out_dir, tmp_root=tmp_dir,
                               target_spacing=float(args.target_spacing),
                               overwrite=bool(args.overwrite))
        with Pool(processes=int(args.workers)) as pool:
            for i, _ in enumerate(pool.imap_unordered(fn, cts), 1):
                print(f"[{i}/{len(cts)}] done")

if __name__ == "__main__":
    main()