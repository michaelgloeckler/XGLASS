#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create reusable disease-conditioned slice-level VLM predictions.

For each test CT-condition row and each axial slice, ask MedGemma a short
yes/no question for the queried condition:

  "Are there signs of <condition> in this axial CT slice?"

This is the expensive step. The output can be reused to evaluate any slice
ordering without rerunning MedGemma.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def norm_hu(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip(arr, lo, hi).astype(np.float32)
    x = (x - lo) / (hi - lo + 1e-6)
    return np.round(np.clip(x * 255.0, 0, 255)).astype(np.uint8)


def slice_to_image(
    slice_hu: np.ndarray,
    rotate_90: bool,
    rendering: str = "multi_window_rgb",
) -> Image.Image:
    if rotate_90:
        slice_hu = np.rot90(slice_hu)
    wide = norm_hu(slice_hu, -1024, 1024)
    if rendering == "wide_rgb":
        r = g = b = wide
    elif rendering == "multi_window_rgb":
        r = wide
        g = norm_hu(slice_hu, -135, 215)
        b = norm_hu(slice_hu, 0, 80)
    else:
        raise ValueError(f"Unknown slice rendering: {rendering}")
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def build_messages(image: Image.Image, category: str) -> List[Dict[str, Any]]:
    system = (
        "You are an expert thoracic radiologist. Evaluate only the queried "
        "finding on one axial chest CT slice."
    )
    user = (
        f"Are there clear visual signs of {category} in this axial CT slice?\n"
        "Ignore other abnormalities. If the queried finding is not visible in "
        "this slice, answer no.\n"
        "Return only one short line, with no explanation:\n"
        "Answer: yes\n"
        "or\n"
        "Answer: no"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": image}]},
    ]


YESNO_RE = re.compile(r"answer\s*:\s*(yes|no)\b", re.IGNORECASE)


def parse_yes_no(raw: str) -> str:
    text = raw or ""
    m = YESNO_RE.search(text)
    if m:
        return m.group(1).lower()
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    yes = re.search(r"\byes\b", t)
    no = re.search(r"\bno\b", t)
    if yes and no:
        return "yes" if yes.start() < no.start() else "no"
    if yes:
        return "yes"
    if no:
        return "no"
    return "unknown"


def extract_generated_text(output: Any) -> str:
    """Normalize conversational pipeline outputs across supported VLM families."""
    if isinstance(output, list) and len(output) == 1 and isinstance(output[0], dict):
        output = output[0]
    if not isinstance(output, dict):
        return str(output)
    generated = output.get("generated_text", "")
    if isinstance(generated, str):
        return generated
    if isinstance(generated, list) and generated:
        last = generated[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "\n".join(texts)
        return str(last)
    return str(generated)


def run_vlm_batch(
    pipe,
    images: List[Image.Image],
    categories: List[str],
    max_new_tokens: int,
    batch_size: int,
) -> List[Tuple[str, str]]:
    conversations = [
        build_messages(image, category)
        for image, category in zip(images, categories)
    ]
    outputs = pipe(
        conversations,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        batch_size=batch_size,
    )
    if len(conversations) == 1 and isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        outputs = [outputs]
    results: List[Tuple[str, str]] = []
    for output in outputs:
        raw = extract_generated_text(output)
        results.append((parse_yes_no(raw), raw))
    if len(results) != len(conversations):
        raise RuntimeError(
            f"Pipeline returned {len(results)} outputs for {len(conversations)} inputs."
        )
    return results


def map_z_to_gt_idx(z: int, zdim: int, gt_len: int) -> int:
    if gt_len <= 1:
        return 0
    if zdim <= 1:
        return max(0, min(gt_len - 1, z))
    return max(0, min(gt_len - 1, int(round(z / (zdim - 1) * (gt_len - 1)))))


def load_test_manifest(data_root: Path, splits_json: Path) -> pd.DataFrame:
    manifest = pd.read_csv(data_root / "manifest.csv")
    with splits_json.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    return manifest[manifest["ct_name"].isin(set(splits["test"]))].reset_index(drop=True)


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[str(row["cache_key"])] = row
    log(f"Loaded {len(cache)} cached conditioned slice predictions from {path}")
    return cache


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create disease-conditioned MedGemma slice predictions.")
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--data_root", type=str, default="data/processed/projection_dataset")
    parser.add_argument("--splits_json", type=str, default="splits/paper_splits.json")
    parser.add_argument("--cropped_ct_dir", type=str, default="data/processed/cropped_test_cts")
    parser.add_argument("--out_jsonl", type=str, default="results/slice_predictions.jsonl")
    parser.add_argument("--gt_threshold", type=float, default=0.0)
    parser.add_argument("--max_rows", type=int, default=0, help="Limit CT-condition rows for smoke tests.")
    parser.add_argument("--max_slices_per_row", type=int, default=0, help="Limit slices per CT-condition row.")
    parser.add_argument("--rotate_90", action="store_true", default=True)
    parser.add_argument("--no_rotate_90", action="store_false", dest="rotate_90")
    parser.add_argument(
        "--slice_rendering",
        choices=("multi_window_rgb", "wide_rgb"),
        default="multi_window_rgb",
        help=(
            "multi_window_rgb maps wide, soft-tissue, and mediastinal windows "
            "to RGB; wide_rgb replicates the wide window across all channels."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of independent CT slices evaluated per pipeline batch.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model compute dtype. Use bfloat16 for A100/H100 evaluation.",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "8bit", "4bit"),
        default="none",
        help="Optional bitsandbytes quantization for models that do not fit fully on the GPU.",
    )
    args = parser.parse_args()
    from transformers import BitsAndBytesConfig, pipeline
    if args.batch_size < 1 or args.gt_threshold != 0:
        parser.error("Use a positive batch size and gt_threshold=0 for paper slice targets")

    data_root = Path(args.data_root)
    splits_json = Path(args.splits_json)
    if not splits_json.is_file():
        splits_json = data_root / args.splits_json
    cropped_dir = Path(args.cropped_ct_dir)
    out_path = Path(args.out_jsonl)

    manifest = load_test_manifest(data_root, splits_json)
    if args.max_rows:
        manifest = manifest.iloc[: args.max_rows].copy()
    cache = load_cache(out_path)

    log(f"CT-condition rows to process: {len(manifest)}")
    log(f"Loading MedGemma: {args.model_id}")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if args.quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    log(
        f"Quantization: {args.quantization}; dtype: {args.dtype}; "
        f"batch size: {args.batch_size}; rendering: {args.slice_rendering}"
    )
    pipe = pipeline("image-text-to-text", model=args.model_id, **model_kwargs)
    pipe.model.config.use_cache = True

    for _, row in tqdm(list(manifest.iterrows()), desc="Conditioned VLM slice predictions"):
        ct_stem = str(row["ct_stem"])
        category = str(row["category"])
        ct_path = cropped_dir / f"{ct_stem}_cropped.nii.gz"
        if not ct_path.is_file():
            raise FileNotFoundError(ct_path)

        gt_path = data_root / str(row["gt_data_json_path"])
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        with gt_path.open("r", encoding="utf-8") as f:
            gt_vec = json.load(f)["gt_vector"]

        ct_hu = nib.load(str(ct_path)).get_fdata().astype(np.float32)
        zdim = int(ct_hu.shape[2])
        if zdim != 240 or len(gt_vec) != 240:
            raise ValueError(f"Expected aligned 240-slice crop and target: {ct_path}")
        zs = list(range(zdim))
        if args.max_slices_per_row:
            zs = zs[: args.max_slices_per_row]

        pending: List[Tuple[int, str, int, bool, Image.Image]] = []
        for z in zs:
            if args.slice_rendering == "multi_window_rgb":
                # Preserve compatibility with previously generated predictions.
                key = (
                    f"{args.model_id}|{ct_stem}|{category}|z{z}|"
                    f"rot{int(args.rotate_90)}|conditioned_yesno_v1"
                )
            else:
                key = (
                    f"{args.model_id}|{ct_stem}|{category}|z{z}|"
                    f"rot{int(args.rotate_90)}|render{args.slice_rendering}|"
                    "conditioned_yesno_v1"
                )
            key += f"|dtype{args.dtype}|quant{args.quantization}|tokens{args.max_new_tokens}|gt{args.gt_threshold}"
            if key in cache:
                continue
            gt_idx = map_z_to_gt_idx(z, zdim, len(gt_vec))
            gt_yes = float(gt_vec[gt_idx]) > args.gt_threshold
            image = slice_to_image(
                ct_hu[:, :, z],
                rotate_90=args.rotate_90,
                rendering=args.slice_rendering,
            )
            pending.append((z, key, gt_idx, gt_yes, image))

        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            results = run_vlm_batch(
                pipe,
                [item[4] for item in batch],
                [category] * len(batch),
                args.max_new_tokens,
                args.batch_size,
            )
            for (z, key, _gt_idx, gt_yes, _image), (answer, raw) in zip(batch, results):
                out = {
                    "cache_key": key,
                    "model_id": args.model_id,
                    "ct_stem": ct_stem,
                    "category": category,
                    "z": int(z),
                    "zdim": zdim,
                    "slice_rendering": args.slice_rendering,
                    "dtype": args.dtype,
                    "quantization": args.quantization,
                    "max_new_tokens": args.max_new_tokens,
                    "gt_threshold": args.gt_threshold,
                    "rotate_90": args.rotate_90,
                    "gt_yes": bool(gt_yes),
                    "answer": answer,
                    "pred_yes": answer == "yes",
                    "raw": raw,
                }
                cache[key] = out
                append_jsonl(out_path, out)

    log(f"Saved/reused predictions in: {out_path.resolve()}")


if __name__ == "__main__":
    main()
