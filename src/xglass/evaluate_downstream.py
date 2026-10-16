#!/usr/bin/env python3
"""Evaluate cached slice-wise finding predictions at matched inspection budgets.

Precision, recall and F1 are micro-averaged over inspected slices, as in the
paper. Uninspected slices do not enter the confusion matrix. This is not
whole-volume sensitivity. Unknown VLM answers are treated as negative.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import re


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def load_predictions(path, expected_slices):
    predictions = {}
    model_ids = set()
    configurations = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("ct_stem", "category", "z", "gt_yes", "pred_yes"):
                if field not in row:
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            if not isinstance(row["gt_yes"], bool) or not isinstance(row["pred_yes"], bool):
                raise ValueError(f"{path}:{line_number}: labels must be JSON booleans")
            z = row["z"]
            if isinstance(z, bool) or not isinstance(z, int) or not 0 <= z < expected_slices:
                raise ValueError(f"{path}:{line_number}: invalid axial index {z}")
            if row.get("zdim", expected_slices) != expected_slices:
                raise ValueError(f"{path}:{line_number}: unexpected volume depth")
            model_ids.add(row.get("model_id", "unspecified"))
            configurations.add(tuple(str(row.get(k, "unspecified")) for k in
                                     ("slice_rendering", "dtype", "quantization", "gt_threshold", "max_new_tokens", "rotate_90", "threshold")))
            key = (row["ct_stem"], row["category"])
            predictions.setdefault(key, {})[z] = row
    if not predictions:
        raise ValueError("Prediction file is empty")
    if len(model_ids) != 1 or len(configurations) != 1:
        raise ValueError("Use one model/rendering/inference configuration per prediction file")
    return predictions, next(iter(model_ids))


def load_cases(data_root, splits_json):
    splits = json.loads(splits_json.read_text(encoding="utf-8"))
    test_names = set(splits["test"])
    with (data_root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["ct_name"] in test_names]
    keys = [(r["ct_stem"], r["category"]) for r in rows]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Test manifest must contain nonempty, unique CT--finding cases")
    return rows


def read_order(directory, ct_stem, category, expected_slices):
    path = directory / f"{safe_filename(ct_stem)}__{safe_filename(category)}.txt"
    if not path.is_file():
        # Disease-agnostic selectors may share one order per CT.
        path = directory / f"{safe_filename(ct_stem)}.txt"
    values = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != expected_slices or set(values) != set(range(expected_slices)):
        raise ValueError(f"Order must be a permutation of 0..{expected_slices - 1}: {path}")
    return values


def confusion(predictions, selected):
    counts = dict(tp=0, fp=0, fn=0, tn=0, unknown=0)
    for z in selected:
        row = predictions[z]
        unknown = str(row.get("answer", "")).lower() == "unknown"
        pred = row["pred_yes"] and not unknown
        gt = row["gt_yes"]
        label = "tp" if pred and gt else "fp" if pred else "fn" if gt else "tn"
        counts[label] += 1
        counts["unknown"] += int(unknown)
    return counts


def metrics(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    divide = lambda a, b: a / b if b else 0.0
    return dict(precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                f1=divide(2 * tp, 2 * tp + fp + fn), accuracy=divide(tp + tn, tp + fp + fn + tn))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--splits_json", type=Path, required=True)
    parser.add_argument("--orders", action="append", required=True, metavar="NAME=DIRECTORY",
                        help="Repeat for each selector; orders are zero-based and finding-specific")
    parser.add_argument("--k", nargs="+", type=int, default=[30, 60, 120, 180])
    parser.add_argument("--expected_slices", type=int, default=240)
    parser.add_argument("--include_random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="Explicitly restrict ALL selectors to cases with complete prediction coverage")
    parser.add_argument("--out_summary", type=Path, default=Path("results/downstream_summary.csv"))
    parser.add_argument("--out_records", type=Path, default=Path("results/downstream_records.csv"))
    args = parser.parse_args()
    if args.expected_slices <= 0 or any(k <= 0 or k > args.expected_slices for k in args.k):
        parser.error("Budgets must lie between 1 and expected_slices")
    specs = {}
    for item in args.orders:
        name, separator, directory = item.partition("=")
        if not separator or not name or not directory or name in specs or name in ("all_slices", "random"):
            parser.error("Use distinct NAME=DIRECTORY selectors; all_slices and random are reserved")
        specs[name] = Path(directory)
    predictions, model_id = load_predictions(args.predictions, args.expected_slices)
    manifest = load_cases(args.data_root, args.splits_json)
    complete = []
    expected = set(range(args.expected_slices))
    for row in manifest:
        key = (row["ct_stem"], row["category"])
        if key not in predictions or set(predictions[key]) != expected:
            if args.allow_incomplete:
                continue
            raise ValueError(f"Missing/incomplete predictions for {key}; finish prediction generation first")
        target_path = args.data_root / row.get("gt_data_json_path", str(Path(row["data_path"]) / "gt_data.json"))
        target = json.loads(target_path.read_text(encoding="utf-8"))["gt_vector"]
        if len(target) != args.expected_slices:
            raise ValueError(f"Wrong target length: {target_path}")
        if any(predictions[key][z]["gt_yes"] != (float(target[z]) > 0) for z in expected):
            raise ValueError(f"Prediction labels do not match the positive-area target for {key}")
        complete.append(row)
    if not complete:
        raise ValueError("No complete test cases")
    print(f"Evaluating {len(complete)}/{len(manifest)} test cases on the same cohort for all selectors")
    if len(complete) < len(manifest):
        print("Partial-cohort run: these results do not represent the complete paper test set")
    rng = random.Random(args.seed)
    totals, records = {}, []
    for row in complete:
        key = (row["ct_stem"], row["category"])
        orders = {name: read_order(path, *key, args.expected_slices) for name, path in specs.items()}
        if args.include_random:
            order = list(range(args.expected_slices))
            rng.shuffle(order)
            orders["random"] = order
        subsets = [("all_slices", args.expected_slices, list(range(args.expected_slices)))]
        subsets += [(name, k, order[:k]) for name, order in orders.items()
                    for k in sorted(set(args.k + [args.expected_slices]))]
        for name, k, selected in subsets:
            counts = confusion(predictions[key], selected)
            records.append(dict(ct_stem=key[0], category=key[1], selector=name, K=k, **counts))
            total = totals.setdefault((name, k), dict(cases=0, inspected_slices=0, tp=0, fp=0, fn=0, tn=0, unknown=0))
            total["cases"] += 1
            total["inspected_slices"] += len(selected)
            for field, value in counts.items():
                total[field] += value
    summary = [dict(downstream_model=model_id, selector=name, K=k, **counts, **metrics(counts))
               for (name, k), counts in totals.items()]
    write_csv(args.out_summary, summary)
    write_csv(args.out_records, records)
    for row in summary:
        print(f"{row['selector']:16} K={row['K']:3} P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}")


if __name__ == "__main__":
    main()
