#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Projection Dataset (AIP + Beer–Lambert) – Fixed-Slice-Block Method

Drop-in replacement for your Beer–Lambert-only script, but additionally writes:
  - aip_wide.png, aip_lung.png, aip_soft.png
  - bl_cxr.png

Keeps the rest identical:
  - Resampling to isotropic spacing
  - Fixed Z-crop around bbox
  - GT vector from axial mask area + GT order (positives first, negatives shuffled)
  - bbox_256.json
  - manifest.csv
  - optional overlays with optional REFLACX heatmap overlay

NOTE: This script, like your original, crops X and Z, but keeps full Y (":").
"""

import json, re, argparse, random, sys, time, cv2, csv
from pathlib import Path
from collections import defaultdict
from typing import List, Any

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # cluster/headless safe
import matplotlib.pyplot as plt
from nibabel.processing import resample_to_output, resample_from_to
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import gaussian_filter

# --- Kategorien (wie in deinem Script) ---
CATS_TO_PROCESS = [
    "Atelectasis", "Consolidation", "Groundglass opacity",
    "High lung volume / emphysema", "Lung nodule or mass", "Pleural abnormality",
]
CAT2SLUG = {c: re.sub(r'[^a-z0-9]+','_',c.lower()).strip('_') for c in CATS_TO_PROCESS}

HEATMAP_SLUGS = {
    "Atelectasis": "atelectasis",
    "Consolidation": "consolidation",
    "Groundglass opacity": "groundglass-opacity",
    "High lung volume / emphysema": "high-lung-volume--emphysema",
    "Lung nodule or mass": "lung-nodule-or-mass",
    "Pleural abnormality": "pleural-abnormality",
}

# --- Regex wie vorher ---
CRE = {
    "Atelectasis": re.compile(r"\batelectas(?:is|tic)\b|\bcollaps(?:e|ed|ing)\b|\bsubsegmental[-\s]+atelectasis\b|\bplate[-\s]+atelectasis\b|\bvolume[-\s]+loss\b", re.I),
    "Consolidation": re.compile(r"\bconsolidat(?:e|ion|ed|ing)\b|\bair[-\s]*space(?:\s+opacity)?\b|\balveolar[-\s]+opacity\b|\blobar[-\s]+consolidation\b|\bsegmental[-\s]+consolidation\b", re.I),
    "Groundglass opacity": re.compile(r"\bground[-\s]?glass\b|\bGGO\b|\bground[-\s]+glass[-\s]+opacit(?:y|ies)\b", re.I),
    "High lung volume / emphysema": re.compile(r"\bhyperinflation\b|\bhyperexpand(?:ed|s|ing)?\b|\bhigh[-\s]+lung[-\s]+volume\b|\bemphysema(?:tous)?\b|\bhyperinflated[-\s]+lungs?\b", re.I),
    "Lung nodule or mass": re.compile(r"\bpulmonary[-\s]+nodule(?:s)?\b|\blung[-\s]+nodule(?:s)?\b|\bnodule(?:s)?\b|\bnodular[-\s]+opacity(?:ies)?\b|\bpulmonary[-\s]+mass\b|\bmass[-\s]+like[-\s]+opacity\b|\bmass\b|\blesion\b", re.I),
    "Pleural abnormality": re.compile(r"\bpleural[-\s]+effusion(?:s)?\b|\bpleural[-\s]+fluid\b|\bpleural[-\s]+thickening\b|(?:\bblunting.*costophrenic|costophrenic.*blunting)\b", re.I),
}

# --- Helpers ---
def log(m): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

def findings_to_text(f: Any) -> str:
    if f is None: return ""
    if isinstance(f, str): return f.lower()
    parts=[]
    if isinstance(f, dict):
        for v in f.values():
            if isinstance(v, str) and v.strip(): parts.append(v)
            elif isinstance(v, (list,tuple)):
                parts.extend([t for t in v if isinstance(t, str) and t.strip()])
    elif isinstance(f, (list,tuple)):
        for v in f:
            if isinstance(v, str) and v.strip(): parts.append(v)
            elif isinstance(v, dict):
                parts.extend([vv for vv in v.values() if isinstance(vv, str) and vv.strip()])
    return " ; ".join(parts).lower()

def match_cats(txt: str) -> List[str]:
    return [c for c,rx in CRE.items() if rx.search(txt or "")]

def load_chest_bbox(chest_bbox_dir: Path, ct_name: str) -> List[int]:
    stem = Path(ct_name).name
    base = stem[:-7] if stem.endswith(".nii.gz") else Path(stem).stem
    p = chest_bbox_dir / f"{base}_chest_bbox.json"
    if not p.is_file(): return []
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        bb = d.get("bbox_vox")
        if not (isinstance(bb,(list,tuple)) and len(bb)==6): return []
        return [int(v) for v in bb]
    except Exception:
        return []

def save_png(img2d: np.ndarray, path: Path):
    """Saves a float image as grayscale PNG. Keeps your original orientation convention."""
    img_to_save = np.flipud(img2d.T)  # (W,H)->(H,W) + flip
    mn, mx = float(img_to_save.min()), float(img_to_save.max())
    if mx > mn:
        img_norm = (img_to_save - mn) / (mx - mn)
    else:
        img_norm = np.zeros_like(img_to_save, dtype=np.float32)
    im8 = (np.clip(img_norm, 0, 1) * 255 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(im8, mode="L").save(str(path))

def apply_window(vol: np.ndarray, center: float, width: float) -> np.ndarray:
    lo, hi = center - width/2.0, center + width/2.0
    x = vol.astype(np.float32)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-6)

def beer_lambert_cxr(vol_hu: np.ndarray,
                     axis: int,
                     spacing_mm: float,
                     mu_water_mm: float = 0.02,
                     scale: float = 1.0,
                     hu_min_clip: int = -1000,
                     invert: bool = True) -> np.ndarray:
    hu = np.clip(vol_hu.astype(np.float32), hu_min_clip, None)
    mu_rel = (hu / 1000.0 + 1.0)         # ≈ rho/rho_water
    mu = mu_rel * float(mu_water_mm)     # [1/mm]
    line_int = np.sum(mu, axis=axis) * float(spacing_mm)
    I = np.exp(-float(scale) * line_int) # transmission
    img = 1.0 - I if invert else I
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img.astype(np.float32)

def union_mask_on_ct_affine(ct_img_native: nib.Nifti1Image,
                            gt_img_native: nib.Nifti1Image,
                            idxs: List[int]) -> nib.Nifti1Image:
    arr = gt_img_native.get_fdata()
    union = np.zeros(ct_img_native.shape, dtype=np.uint8)
    if arr.ndim == 4:
        if not idxs:
            return nib.Nifti1Image(union, ct_img_native.affine, ct_img_native.header)
        valid_idxs = [i for i in idxs if i < arr.shape[0]]
        if not valid_idxs:
            return nib.Nifti1Image(union, ct_img_native.affine, ct_img_native.header)
        union_4d = (arr[valid_idxs, ...] > 0).any(axis=0).astype(np.uint8)
    elif arr.ndim == 3:
        union_4d = (arr > 0).astype(np.uint8)
    else:
        raise RuntimeError(f"Unexpected GT ndim {arr.ndim}")

    if union_4d.shape != ct_img_native.shape:
        target = np.zeros(ct_img_native.shape, dtype=np.uint8)
        s = tuple(min(s,t) for s,t in zip(union_4d.shape, target.shape))
        target[:s[0], :s[1], :s[2]] = union_4d[:s[0], :s[1], :s[2]]
        union = target
    else:
        union = union_4d
    return nib.Nifti1Image(union, ct_img_native.affine, ct_img_native.header)

# ---------------------------- Main ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate AIP (3 windows) + Beer–Lambert projection dataset.")
    ap.add_argument("--data_root", type=str, default="./mini_dataset")
    ap.add_argument("--input_json", type=str, default="dataset.json")
    ap.add_argument("--splits_json", type=str, default="splits/paper_splits.json")
    ap.add_argument("--heatmap_dir", type=str, default="./avg_np")
    ap.add_argument("--out_dir", type=str, default="./projection_dataset_fixed_z_aip_bl")
    ap.add_argument("--target_spacing_mm", type=float, default=1.25)
    ap.add_argument("--target_z_slices", type=int, default=240)
    ap.add_argument("--target_xy_dim", type=int, default=256)
    ap.add_argument("--overlay_count", type=int, default=2)

    # Beer–Lambert params
    ap.add_argument("--bl_mu_water_mm", type=float, default=0.02)
    ap.add_argument("--bl_scale", type=float, default=1.0)
    ap.add_argument("--bl_hu_min_clip", type=int, default=-1000)

    # AIP window params (defaults match your earlier script)
    ap.add_argument("--wide_center", type=float, default=0.0)
    ap.add_argument("--wide_width",  type=float, default=2000.0)
    ap.add_argument("--lung_center", type=float, default=-600.0)
    ap.add_argument("--lung_width",  type=float, default=1500.0)
    ap.add_argument("--soft_center", type=float, default=40.0)
    ap.add_argument("--soft_width",  type=float, default=400.0)

    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_data_root = out_dir / "data"; out_data_root.mkdir(parents=True, exist_ok=True)
    out_overlay_root = out_dir / "overlays"; out_overlay_root.mkdir(parents=True, exist_ok=True)

    heatmap_dir = Path(args.heatmap_dir)
    if not heatmap_dir.is_dir():
        log(f"INFO: Heatmap directory not found: {heatmap_dir}. Overlays without heatmaps.")
        heatmap_dir = None

    ct_root = data_root / "cts"
    seg_root = data_root / "segmentations"
    bbox_root = data_root / "ChestBBoxes"

    dataset_json_path = data_root / args.input_json
    if not dataset_json_path.is_file():
        log(f"ERROR: {dataset_json_path} not found"); sys.exit(1)

    splits_json_path = Path(args.splits_json)
    if not splits_json_path.is_file():
        splits_json_path = data_root / args.splits_json
    if not splits_json_path.is_file():
        log(f"ERROR: {splits_json_path} not found"); sys.exit(1)

    log(f"Loading splits from {splits_json_path} ...")
    splits_data = json.loads(splits_json_path.read_text(encoding="utf-8"))
    ct_to_split = {}
    for split_name in ["train", "val", "test"]:
        for ct_name in splits_data.get(split_name, []):
            ct_to_split[ct_name] = split_name
    log(f"Loaded splits for {len(ct_to_split)} CTs.")

    log(f"Loading manifest {dataset_json_path} ...")
    main_data = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    all_items = []
    if isinstance(main_data, dict):
        for split_items in main_data.values():
            if isinstance(split_items, list):
                all_items.extend(split_items)
    elif isinstance(main_data, list):
        all_items = main_data
    log(f"Found manifest items: {len(all_items)}")

    overlays_written = defaultdict(int)
    processed_cts_per_cat = defaultdict(set)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "category","ct_name","ct_stem","split","data_path",
            "aip_wide_path","aip_lung_path","aip_soft_path",
            "bl_cxr_path",
            "gt_data_json_path","gt_data_csv_path","bbox_json_path"
        ])

        for item in all_items:
            ct_name = item.get("name")
            if not ct_name:
                continue

            # Keep your original behavior (but with safe default):
            if ct_name not in ct_to_split:
                continue
            split_name = ct_to_split[ct_name]

            findings_text = findings_to_text(item.get("findings"))
            hits = [cat for cat in match_cats(findings_text) if cat in CATS_TO_PROCESS]
            if not hits:
                continue

            ct_stem = Path(ct_name).stem

            try:
                ct_path = ct_root / ct_name
                seg_path = seg_root / ct_name
                bbox_vox = load_chest_bbox(bbox_root, ct_name)

                if not ct_path.is_file(): log(f"[SKIP] Missing CT: {ct_name}"); 
                if not seg_path.is_file(): log(f"[SKIP] Missing Seg: {ct_name}"); continue
                if not bbox_vox: log(f"[SKIP] Missing BBox: {ct_name}"); continue

                x0, x1, y0, y1, z0_bbox, z1_bbox = bbox_vox
                if not (x1 > x0 and y1 > y0 and z1_bbox > z0_bbox):
                    log(f"[SKIP] Invalid BBox: {ct_name}"); continue

                ct_native = nib.load(str(ct_path))
                seg_native = nib.load(str(seg_path))

                ct_ras = nib.as_closest_canonical(ct_native)
                ref_iso_nii = resample_to_output(ct_ras, voxel_sizes=(args.target_spacing_mm,)*3, order=1)
                ct_iso_vol = ref_iso_nii.get_fdata()

                total_iso_slices = ct_iso_vol.shape[2]
                if total_iso_slices < args.target_z_slices:
                    log(f"[SKIP] Slices {total_iso_slices} < target {args.target_z_slices} ({ct_name})"); continue
                if (z1_bbox - z0_bbox) > args.target_z_slices:
                    log(f"[SKIP] BBox Z-size {z1_bbox - z0_bbox} > target {args.target_z_slices} ({ct_name})"); continue

                bbox_center_z = (z0_bbox + z1_bbox) / 2.0
                crop_z_start = int(bbox_center_z - (args.target_z_slices / 2.0))
                crop_z_end = crop_z_start + args.target_z_slices
                if crop_z_start < 0:
                    crop_z_start = 0; crop_z_end = args.target_z_slices
                elif crop_z_end > total_iso_slices:
                    crop_z_end = total_iso_slices; crop_z_start = crop_z_end - args.target_z_slices
                if crop_z_start > z0_bbox or crop_z_end < z1_bbox:
                    log(f"[SKIP] BBox {z0_bbox}-{z1_bbox} not in crop {crop_z_start}-{crop_z_end} ({ct_name})"); continue

                # Crop (same as your scripts: X + Z, keep all Y)
                x0_crop, x1_crop = max(0, x0), min(ct_iso_vol.shape[0], x1)
                ct_cropped = ct_iso_vol[x0_crop:x1_crop, :, crop_z_start:crop_z_end]

                # --- Projections computed ONCE per CT ---
                # AIPs (mean over Y axis=1 after windowing)
                wide = apply_window(ct_cropped, args.wide_center, args.wide_width)
                lung = apply_window(ct_cropped, args.lung_center, args.lung_width)
                soft = apply_window(ct_cropped, args.soft_center, args.soft_width)

                aip_wide_raw = np.mean(wide, axis=1)
                aip_lung_raw = np.mean(lung, axis=1)
                aip_soft_raw = np.mean(soft, axis=1)

                # Beer–Lambert (integrate along same axis=1)
                bl_raw = beer_lambert_cxr(
                    ct_cropped,
                    axis=1,
                    spacing_mm=args.target_spacing_mm,
                    mu_water_mm=args.bl_mu_water_mm,
                    scale=args.bl_scale,
                    hu_min_clip=args.bl_hu_min_clip,
                    invert=True
                )

                # Resize all to 256×256
                target_dim_cv2 = (args.target_xy_dim, args.target_xy_dim)
                aip_wide_final = cv2.resize(aip_wide_raw, target_dim_cv2, interpolation=cv2.INTER_LINEAR)
                aip_lung_final = cv2.resize(aip_lung_raw, target_dim_cv2, interpolation=cv2.INTER_LINEAR)
                aip_soft_final = cv2.resize(aip_soft_raw, target_dim_cv2, interpolation=cv2.INTER_LINEAR)
                bl_final = cv2.resize(bl_raw, target_dim_cv2, interpolation=cv2.INTER_LINEAR)

                # bbox mapping into 256×256 (same as before)
                w_iso_crop = x1_crop - x0_crop
                z_iso_crop_block = args.target_z_slices
                x_scale = args.target_xy_dim / w_iso_crop
                z_scale = args.target_xy_dim / z_iso_crop_block
                x0_rel = x0 - x0_crop; x1_rel = x1 - x0_crop
                z0_rel = z0_bbox - crop_z_start; z1_rel = z1_bbox - crop_z_start
                x0_final = x0_rel * x_scale; x1_final = x1_rel * x_scale
                z0_final_img = z0_rel * z_scale; z1_final_img = z1_rel * z_scale

                bbox_256_data = {
                    "bbox_256_xyxy": [x0_final, z0_final_img, x1_final, z1_final_img],
                    "original_bbox_iso": [x0, y0, z0_bbox, x1, y1, z1_bbox],
                    "crop_window_iso": [x0_crop, x1_crop, crop_z_start, crop_z_end]
                }

                # --- per category ---
                for cat in [c for c in hits if ct_name not in processed_cts_per_cat[c]]:
                    processed_cts_per_cat[cat].add(ct_name)

                    # category-specific union mask indices
                    findings = item.get("findings") or {}
                    f_idxs = []
                    if seg_native.get_fdata().ndim == 4:
                        F = seg_native.shape[0]
                        for f in range(F):
                            txt = findings.get(str(f), "")
                            if cat in match_cats(txt):
                                f_idxs.append(f)

                    union_mask_native = union_mask_on_ct_affine(ct_native, seg_native, f_idxs)
                    mask_iso_nii = resample_from_to(union_mask_native, ref_iso_nii, order=0)
                    mask_iso_vol = mask_iso_nii.get_fdata().astype(np.uint8)
                    mask_cropped = mask_iso_vol[x0_crop:x1_crop, :, crop_z_start:crop_z_end]

                    # GT vector + order (unchanged)
                    gt_axial_areas = np.sum(mask_cropped, axis=(0, 1)).astype(np.float32)
                    gt_vec_norm = gt_axial_areas / np.max(gt_axial_areas) if np.max(gt_axial_areas) > 0 else gt_axial_areas

                    all_indices = np.arange(args.target_z_slices)
                    gt_indices = all_indices[gt_vec_norm > 0]
                    zero_indices = all_indices[gt_vec_norm == 0]
                    sorted_gt_indices = gt_indices[np.argsort(gt_vec_norm[gt_indices])[::-1]]
                    random.shuffle(zero_indices)
                    final_gt_order = np.concatenate([sorted_gt_indices, zero_indices]).astype(int)

                    cat_slug = CAT2SLUG[cat]
                    out_data_dir = (out_dir / "data" / split_name / cat_slug / ct_stem)
                    out_data_dir.mkdir(parents=True, exist_ok=True)

                    # write files
                    p_aip_wide = out_data_dir / "aip_wide.png"
                    p_aip_lung = out_data_dir / "aip_lung.png"
                    p_aip_soft = out_data_dir / "aip_soft.png"
                    p_bl       = out_data_dir / "bl_cxr.png"
                    p_gt_json  = out_data_dir / "gt_data.json"
                    p_gt_csv   = out_data_dir / "gt_data_human.csv"
                    p_bbox     = out_data_dir / "bbox_256.json"

                    save_png(aip_wide_final, p_aip_wide)
                    save_png(aip_lung_final, p_aip_lung)
                    save_png(aip_soft_final, p_aip_soft)
                    save_png(bl_final, p_bl)

                    with p_gt_json.open("w", encoding="utf-8") as f_json:
                        json.dump({"gt_vector": gt_vec_norm.tolist(),
                                   "gt_order": final_gt_order.tolist()}, f_json)

                    with p_gt_csv.open("w", newline="", encoding="utf-8") as f_csv:
                        writer_h = csv.writer(f_csv)
                        writer_h.writerow(["sorted_slice_index", "normalized_gt_area"])
                        for sidx in final_gt_order:
                            writer_h.writerow([int(sidx), float(gt_vec_norm[sidx])])

                    with p_bbox.open("w", encoding="utf-8") as f_json:
                        json.dump(bbox_256_data, f_json)

                    writer.writerow([
                        cat, ct_name, ct_stem, split_name, out_data_dir.relative_to(out_dir),
                        p_aip_wide.relative_to(out_dir),
                        p_aip_lung.relative_to(out_dir),
                        p_aip_soft.relative_to(out_dir),
                        p_bl.relative_to(out_dir),
                        p_gt_json.relative_to(out_dir),
                        p_gt_csv.relative_to(out_dir),
                        p_bbox.relative_to(out_dir),
                    ])

                    # --- overlays (limited) ---
                    if overlays_written[cat] < args.overlay_count:
                        coronal_mask_proj = np.max(mask_cropped, axis=1)
                        mask_to_plot = cv2.resize(coronal_mask_proj.T, target_dim_cv2, interpolation=cv2.INTER_NEAREST)

                        # 4-panel overlay: AIP wide + lung + soft + BL
                        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
                        fig.suptitle(f"CT: {ct_name} | Category: {cat} (Fixed Z-crop)", fontsize=12)

                        panels = [
                            ("AIP Wide", aip_wide_final.T),
                            ("AIP Lung", aip_lung_final.T),
                            ("AIP Soft", aip_soft_final.T),
                            ("Beer–Lambert", bl_final.T),
                        ]

                        # Optional heatmap overlay only on first panel (AIP Wide), like your original style
                        heatmap_rgba = None
                        if heatmap_dir:
                            heatmap_slug = HEATMAP_SLUGS.get(cat)
                            if heatmap_slug:
                                npz_path = heatmap_dir / f"03_avg_raw_{heatmap_slug}.npz"
                                if npz_path.is_file():
                                    try:
                                        hm = np.load(npz_path)
                                        gaze_counts = hm["gaze_counts"].astype(np.float32)
                                        gaze_blurred = gaussian_filter(gaze_counts, sigma=10)
                                        gmin, gmax = gaze_blurred.min(), gaze_blurred.max()
                                        gaze_norm = (gaze_blurred - gmin) / (gmax - gmin) if gmax > gmin else np.zeros_like(gaze_blurred)

                                        x0_f, z0_f, x1_f, z1_f = bbox_256_data["bbox_256_xyxy"]
                                        bbox_w = int(np.round(x1_f - x0_f))
                                        bbox_h = int(np.round(z1_f - z0_f))
                                        if bbox_w > 0 and bbox_h > 0:
                                            gaze_resized = cv2.resize(gaze_norm, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
                                            heatmap_colors = plt.cm.Reds(gaze_resized)
                                            heatmap_colors[..., 3] = gaze_resized * 0.7
                                            heatmap_rgba = np.zeros((args.target_xy_dim, args.target_xy_dim, 4), dtype=np.float32)

                                            x0_i, z0_i = int(np.round(x0_f)), int(np.round(z0_f))
                                            h_slice = slice(z0_i, min(z0_i + bbox_h, args.target_xy_dim))
                                            w_slice = slice(x0_i, min(x0_i + bbox_w, args.target_xy_dim))
                                            h_heat = min(bbox_h, args.target_xy_dim - z0_i)
                                            w_heat = min(bbox_w, args.target_xy_dim - x0_i)
                                            if h_heat > 0 and w_heat > 0:
                                                heatmap_rgba[h_slice, w_slice] = heatmap_colors[:h_heat, :w_heat]
                                    except Exception as e:
                                        log(f"[WARN] Heatmap overlay failed: {npz_path}: {e}")

                        for i, ax in enumerate(axes):
                            title, img_to_plot = panels[i]
                            ax.imshow(img_to_plot, cmap="gray", origin="lower", aspect="equal",
                                      extent=(0, args.target_xy_dim, 0, args.target_xy_dim))

                            if i == 0 and heatmap_rgba is not None:
                                ax.imshow(heatmap_rgba, origin="lower", aspect="equal",
                                          extent=(0, args.target_xy_dim, 0, args.target_xy_dim))

                            if np.any(mask_to_plot):
                                ax.contour(mask_to_plot, levels=[0.5], colors=["lime"], linewidths=1.0,
                                           origin="lower", extent=(0, args.target_xy_dim, 0, args.target_xy_dim))

                            x0_f, z0_f, x1_f, z1_f = bbox_256_data["bbox_256_xyxy"]
                            rect = Rectangle((x0_f, z0_f), (x1_f - x0_f), (z1_f - z0_f),
                                             fill=False, linewidth=1.5, edgecolor="red")
                            ax.add_patch(rect)

                            ax.set_title(title, fontsize=10)
                            ax.set_xticks([]); ax.set_yticks([])

                        out_path = out_overlay_root / split_name / f"{ct_stem}_{CAT2SLUG[cat]}_overlay_aip_bl.png"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        plt.tight_layout(rect=[0, 0.02, 1, 0.92])
                        plt.savefig(out_path, dpi=150)
                        plt.close(fig)
                        overlays_written[cat] += 1

            except Exception as e:
                log(f"[ERROR] Failed for {ct_name}: {e}")
                import traceback; traceback.print_exc()

    log("--- Dataset Creation Complete (AIP + BL) ---")
    log(f"Manifest: {manifest_path}")
    log(f"Data Root: {out_data_root.resolve()}")
    log(f"Overlays:  {out_overlay_root.resolve()}")

if __name__ == "__main__":
    main()
