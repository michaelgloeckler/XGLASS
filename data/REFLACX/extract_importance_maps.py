#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFLACX -> disease-conditioned RAW gaze heatmaps (bbox-normalized)

- Aggregates gaze samples during positive transcript mentions of a disease label.
- Coordinates are normalized to the chest bounding box: u,v in [0,1].
- Saves RAW (unsmoothed) counts to NPZ. Smoothing can be applied downstream.

Expected folder structure (REFLACX root):
ROOT_DIR/
  main_data/
    metadata_phase_1.csv (optional; not required)
    <SID>/
      chest_bounding_box.csv
      timestamps_transcription.csv
      (other files ignored)
  gaze_data/
    <SID>/
      gaze.csv
"""

import os, re, json
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
ROOT_DIR   = r"mimic-eye\mimic-eye"      # set REFLACX root: contains main_data/ and gaze_data/
OUT_DIR    = r"avg_np"                  # output folder (npz files)
REGEX_FILE = "regex_med_anchors.json"   # optional external regex mapping

LABELS_TO_RUN = [
    "Abnormal mediastinal contour",
    "Acute fracture",
    "Atelectasis",
    "Consolidation",
    "Enlarged cardiac silhouette",
    "Enlarged hilum",
    "Groundglass opacity",
    "Hiatal hernia",
    "High lung volume / emphysema",
    "Interstitial lung disease",
    "Lung nodule or mass",
    "Pleural abnormality",
    "Pneumothorax",
    "Pulmonary edema",
]

HEAT_W, HEAT_H = 256, 256
PRE_PAD, POST_PAD = 2.0, 0.0
# ---------------------------------------

NEG_TOKENS   = r"(?:no|without|absent|free of|negative for|not|neither|den(?:y|ies|ied))"
CLAUSE_SPLIT = re.compile(r"\s*(?:,|;| but | however )\s*", re.I)
NEG_NEAR     = re.compile(NEG_TOKENS, re.I)

# Conservative defaults (fallback if REGEX_FILE not present).
# You can paste your full REGEX_FOR_DEFAULT here (recommended).
REGEX_FOR_DEFAULT = {
    "Atelectasis": r"\batelectas(?:is|tic)\b|\bcollaps(?:e|ed|ing)\b|\bvolume[-\s]+loss\b",
    "Consolidation": r"\bconsolidat(?:e|ion|ed|ing)\b|\bair[-\s]*space\b|\balveolar[-\s]+opacity\b",
    "Groundglass opacity": r"\bground[-\s]?glass\b|\bGGO\b",
    "High lung volume / emphysema": r"\bhyperinflation\b|\bemphysema(?:tous)?\b|\bhigh[-\s]+lung[-\s]+volume\b",
    "Lung nodule or mass": r"\bnodule(?:s)?\b|\bmass\b|\blesion\b",
    "Pleural abnormality": r"\bpleural[-\s]+effusion(?:s)?\b|\bpleural[-\s]+thickening\b|\bcostophrenic.*blunting\b",
    "Pneumothorax": r"\bpneumothorax\b|\bPTX\b|\bpleural[-\s]+air\b",
    "Pulmonary edema": r"\bpulmonary[-\s]+edema\b|\binterstitial[-\s]+edema\b|\bperihilar[-\s]+edema\b|\bbat[-\s]+wing\b",
    # NOTE: Add remaining labels here or provide them in REGEX_FILE.
}

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def secify(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmax(x) if x.size else 0.0
    if m > 1e6: return x/1e6
    if m > 1e4: return x/1e3
    return x

def pick(cols, names):
    for n in names:
        if n in cols:
            return n
    return None

def list_sessions_reflacx_root(root):
    """List session IDs present in both main_data/ and gaze_data/ under REFLACX root."""
    md_root = os.path.join(root, "main_data")
    gz_root = os.path.join(root, "gaze_data")
    if not (os.path.isdir(md_root) and os.path.isdir(gz_root)):
        raise FileNotFoundError(f"Expected folders: {md_root} and {gz_root}")

    md = {d for d in os.listdir(md_root) if os.path.isdir(os.path.join(md_root, d))}
    gz = {d for d in os.listdir(gz_root) if os.path.isdir(os.path.join(gz_root, d))}
    # Filter out obvious non-session folders/files (e.g., metadata files)
    md = {d for d in md if not d.lower().endswith(".csv")}
    gz = {d for d in gz if not d.lower().endswith(".csv")}
    return sorted(md & gz)

def sentence_table(trans: pd.DataFrame):
    """Split transcript into sentences (delimited by '.') using required columns."""
    w = trans[["word","timestamp_start_word","timestamp_end_word"]].copy()
    w.columns = ["word","t_start","t_end"]
    w["word"] = w["word"].astype(str)
    w["norm"] = (
        w["word"].str.lower()
                 .str.replace(r"[_/]", " ", regex=True)
                 .str.replace(r"[^a-z0-9\.\-\s]+"," ", regex=True)
                 .str.replace(r"\s+"," ", regex=True)
                 .str.strip()
    )

    sents, start = [], 0
    for i in range(len(w)):
        if w.iloc[i]["word"] == ".":
            if i > start:
                sents.append(dict(
                    idx=list(range(start, i)),
                    text=" ".join(w.iloc[start:i]["norm"].tolist())
                ))
            start = i + 1
    if start < len(w):
        sents.append(dict(
            idx=list(range(start, len(w))),
            text=" ".join(w.iloc[start:len(w)]["norm"].tolist())
        ))
    return w, sents

def clausewise_positive_sentence(sentence_text: str, lab_regex: str) -> bool:
    """Returns True if sentence contains non-negated mention of label."""
    start = 0
    for part in CLAUSE_SPLIT.split(sentence_text):
        idx = sentence_text.find(part, start)
        if idx < 0: idx = start
        for m in re.finditer(lab_regex, part, flags=re.I):
            before = part[max(0, m.start()-120):m.start()]
            if NEG_NEAR.search(before):
                continue
            return True
        start = idx + len(part)
    return False

def mention_windows(trans: pd.DataFrame, label: str, regex_for: dict):
    """
    Collect (t0,t1) windows for all words in sentences containing a positive (non-negated) mention.
    Uses PRE_PAD/POST_PAD per-word.
    """
    words, sents = sentence_table(trans)
    patt = regex_for[label]
    windows = []
    for S in sents:
        if clausewise_positive_sentence(S["text"], patt):
            for wi in S["idx"]:
                wrow = words.iloc[wi]
                t0 = max(0.0, float(wrow["t_start"]) - PRE_PAD)
                t1 = float(wrow["t_end"]) + POST_PAD
                windows.append((t0, t1))
    return windows

def add_points_to_heatmap(u, v, heat):
    if len(u) == 0:
        return
    H, _, _ = np.histogram2d(v, u, bins=[HEAT_H, HEAT_W], range=[[0,1],[0,1]])
    heat += H

def load_regex_file_or_default(path):
    regex_for = dict(REGEX_FOR_DEFAULT)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                ext = json.load(f)
            if isinstance(ext, dict):
                for k, v in ext.items():
                    if isinstance(v, str) and v.strip():
                        regex_for[k] = v
            print(f"[OK] Loaded regexes from {path}")
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}. Using defaults.")
    else:
        print(f"[INFO] No {path} found. Using built-in defaults.")
    return regex_for

def build_raw_heatmap_for_label(label: str, regex_for: dict):
    heat = np.zeros((HEAT_H, HEAT_W), dtype=np.float32)

    sessions_seen = 0
    sessions_used = 0
    gaze_samples_used = 0

    for sid in list_sessions_reflacx_root(ROOT_DIR):
        sessions_seen += 1

        mdir = os.path.join(ROOT_DIR, "main_data", sid)
        gdir = os.path.join(ROOT_DIR, "gaze_data", sid)

        trans_path = os.path.join(mdir, "timestamps_transcription.csv")
        cbox_path  = os.path.join(mdir, "chest_bounding_box.csv")
        gaze_path  = os.path.join(gdir, "gaze.csv")
        if not (os.path.exists(trans_path) and os.path.exists(cbox_path) and os.path.exists(gaze_path)):
            continue

        trans = pd.read_csv(trans_path)
        req_cols = {"word","timestamp_start_word","timestamp_end_word"}
        if not req_cols.issubset(set(trans.columns)):
            continue

        windows = mention_windows(trans, label, regex_for)
        if not windows:
            continue

        cbox = pd.read_csv(cbox_path).iloc[0]
        x0, y0, x1, y1 = [float(cbox[c]) for c in ["xmin","ymin","xmax","ymax"]]
        if (x1 - x0) <= 1 or (y1 - y0) <= 1:
            continue

        gaze = pd.read_csv(gaze_path)
        tcol = pick(gaze.columns, ["timestamp_sample","timestamp","time","time_seconds"])
        xcol = pick(gaze.columns, ["x_position","x"])
        ycol = pick(gaze.columns, ["y_position","y"])
        if tcol is None or xcol is None or ycol is None:
            continue

        t = secify(pd.to_numeric(gaze[tcol], errors="coerce").to_numpy())
        x = pd.to_numeric(gaze[xcol], errors="coerce").to_numpy()
        y = pd.to_numeric(gaze[ycol], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
        if not np.any(ok):
            continue
        t, x, y = t[ok], x[ok], y[ok]

        sel = np.zeros_like(t, dtype=bool)
        for t0, t1 in windows:
            sel |= (t >= t0) & (t <= t1)
        if not np.any(sel):
            continue

        # bbox-normalized coords in [0,1]
        u = (x[sel] - x0) / (x1 - x0)
        v = (y[sel] - y0) / (y1 - y0)
        m = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        if not np.any(m):
            continue

        add_points_to_heatmap(u[m], v[m], heat)
        sessions_used += 1
        gaze_samples_used += int(np.sum(m))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_npz = os.path.join(OUT_DIR, f"gaze_raw_{slugify(label)}.npz")
    meta = {
        "label": label,
        "heat_hw": [HEAT_H, HEAT_W],
        "pre_pad_s": float(PRE_PAD),
        "post_pad_s": float(POST_PAD),
        "sessions_seen": int(sessions_seen),
        "sessions_used": int(sessions_used),
        "gaze_samples_used": int(gaze_samples_used),
        "coords": "bbox-normalized (u,v) in [0,1]",
        "stored": "RAW counts only (no smoothing)",
    }
    np.savez_compressed(
        out_npz,
        gaze_counts=heat.astype(np.float32),
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    )
    print(f"[OK] Saved {out_npz} | sessions_used={sessions_used}, gaze_samples={gaze_samples_used}")

def main():
    import argparse
    global ROOT_DIR, OUT_DIR, REGEX_FILE, LABELS_TO_RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--regex_file', default=REGEX_FILE)
    parser.add_argument('--labels', nargs='+', default=list(REGEX_FOR_DEFAULT)[:6])
    args = parser.parse_args()
    ROOT_DIR, OUT_DIR, REGEX_FILE = args.root, args.out, args.regex_file
    LABELS_TO_RUN = args.labels
    regex_for = load_regex_file_or_default(REGEX_FILE)
    missing = [l for l in LABELS_TO_RUN if l not in regex_for]
    if missing:
        raise KeyError(
            "Missing regex for labels: " + ", ".join(missing) +
            f". Add them to REGEX_FOR_DEFAULT or provide them in {REGEX_FILE}."
        )

    for label in LABELS_TO_RUN:
        build_raw_heatmap_for_label(label, regex_for)

if __name__ == "__main__":
    main()
