#!/usr/bin/env python3
"""Cache BioMedCLIP slice-screening predictions using a fixed similarity threshold."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import torch

from evaluate_downstream import load_cases, load_predictions
from biomedclip_utils import (build_prompts, cosine_scores, encode_images_in_chunks,
                             encode_text, get_slice_as_image, load_biomedclip)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data/processed/projection_dataset"))
    parser.add_argument("--splits_json", type=Path, default=Path("splits/paper_splits.json"))
    parser.add_argument("--cropped_ct_dir", type=Path, default=Path("data/processed/cropped_test_cts"))
    parser.add_argument("--out_jsonl", type=Path, default=Path("results/biomedclip_slices.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.3751,
                        help="Fixed paper threshold; for a new setup calibrate on validation data separately")
    parser.add_argument("--chunk_size", type=int, default=32)
    args = parser.parse_args()
    if args.chunk_size <= 0 or not -1 <= args.threshold <= 1:
        parser.error("chunk_size must be positive and threshold must lie in [-1, 1]")
    cached = {}
    if args.out_jsonl.is_file() and args.out_jsonl.stat().st_size:
        cached, model_id = load_predictions(args.out_jsonl, 240)
        if model_id != "BioMedCLIP":
            raise ValueError("Output file contains predictions from a different model")
        if any(row.get("threshold") != args.threshold for case in cached.values() for row in case.values()):
            raise ValueError("Use a different output file when changing the threshold")
    cases = load_cases(args.data_root, args.splits_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = load_biomedclip(device)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    text_features = {}
    last_ct, image_features = None, None
    for case in cases:
        key = (case["ct_stem"], case["category"])
        previous = cached.get(key, {})
        if set(previous) == set(range(240)):
            continue
        target_path = args.data_root / case["gt_data_json_path"]
        targets = json.loads(target_path.read_text(encoding="utf-8"))["gt_vector"]
        ct_path = args.cropped_ct_dir / f"{key[0]}_cropped.nii.gz"
        if key[0] != last_ct:
            ct = nib.load(str(ct_path))
            if ct.ndim != 3 or ct.shape[2] != 240:
                raise ValueError(f"Expected aligned 240-slice CT crop: {ct_path}")
            images = [get_slice_as_image(ct, z) for z in range(240)]
            image_features = encode_images_in_chunks(model, preprocess, images, device, args.chunk_size)
            last_ct = key[0]
        if len(targets) != 240:
            raise ValueError(f"Expected 240 target values: {target_path}")
        if key[1] not in text_features:
            text_features[key[1]] = encode_text(model, tokenizer, build_prompts(key[1]), device).cpu()
        scores = cosine_scores(image_features, text_features[key[1]])
        with args.out_jsonl.open("a", encoding="utf-8") as handle:
            for z, score in enumerate(scores):
                if z in previous:
                    continue
                positive = bool(score >= args.threshold)
                row = dict(model_id="BioMedCLIP", ct_stem=key[0], category=key[1], z=z, zdim=240,
                           slice_rendering="wide_rgb_center0_width2000", threshold=args.threshold,
                           gt_yes=bool(float(targets[z]) > 0), pred_yes=positive,
                           answer="yes" if positive else "no", similarity=float(score))
                handle.write(json.dumps(row) + "\n")
        print(f"Scored {key}")


if __name__ == "__main__":
    main()
