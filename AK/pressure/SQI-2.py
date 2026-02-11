#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQI-2.py — Build a pressure gate for ESP32 from your CSV.

Hard-coded dataset path:
  /home/apc-3/PycharmProjects/PythonProjectAK/SQI/dataset/thumb_spectral_pressure.csv

What it does:
  1) Loads CSV with 555/585 nm columns and label (Yes/No)
  2) Computes Δ = |A585 - A555| and ratio = A585 / A555
  3) Auto-fits thresholds to maximize accuracy (Δ-only or Δ+ratio)
  4) Emits pressure_thresholds_2.h/.cpp for your ESP32 sketch
"""

import csv
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from textwrap import dedent

# ---------- CONFIG ----------
ROOT = "/home/apc-3/PycharmProjects/PythonProjectAK/SQI"
DATASET_PATH = f"{ROOT}/dataset/thumb_spectral_pressure.csv"
MODE = "full"   # "full" = Δ OR ratio (recommended), or "delta" = Δ only
IDX_555 = 4     # raw index of 555 nm (for your info/snippet only)
IDX_585 = 6     # raw index of 585 nm (for your info/snippet only)
OUTDIR = f"{ROOT}/output"
HEADER_FILENAME = "pressure_thresholds_2.h"
SOURCE_FILENAME = "pressure_thresholds_2.cpp"
# ----------------------------

# ----------------- Data loading -----------------
@dataclass
class Dataset:
    a555: List[float]
    a585: List[float]
    labels: List[Optional[bool]]  # True=yes/with-pressure, False=no/without, None=unknown

def parse_bool_label(s: str) -> Optional[bool]:
    if s is None:
        return None
    s = str(s).strip().lower()
    # Treat these as "with pressure"
    if s in {"yes", "y", "1", "true", "with", "pressure", "with-pressure"} or "yes" in s:
        return True
    # Treat these as "no pressure"
    if s in {"no", "n", "0", "false", "without", "no-pressure"} or "no" in s:
        return False
    return None

def find_col_by_candidates_or_contains(cols, candidates, contains_token):
    # exact candidate match first
    for c in candidates:
        if c in cols:
            return c
    # then: case-insensitive "contains"
    tok = contains_token.lower()
    for c in cols:
        if tok in c.lower():
            return c
    return None

def load_csv(path: str) -> Tuple[Dataset, str, str, Optional[str]]:
    """
    Robustly locate columns:
      - exact candidates (A555/a555/Abs555/555 nm/etc)
      - OR any header containing '555' (case-insensitive)
    Same for 585. Label: try 'pressure','label','class', etc.
    Returns dataset + the actual column names used (for logging).
    """
    a555_candidates = {"A555", "a555", "Abs555", "abs555", "555", "555 nm"}
    a585_candidates = {"A585", "a585", "Abs585", "abs585", "585", "585 nm"}
    label_candidates = {
        "pressure", "Pressure", "label", "Label",
        "is_pressure", "IsPressure", "Class", "class",
        "Status", "status", "Target", "target", "Type", "type"
    }

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        if not cols:
            raise ValueError(f"CSV has no header row: {path}")

        a555_col = find_col_by_candidates_or_contains(cols, a555_candidates, "555")
        a585_col = find_col_by_candidates_or_contains(cols, a585_candidates, "585")
        label_col = None
        for c in label_candidates:
            if c in cols:
                label_col = c
                break

        if not a555_col or not a585_col:
            raise ValueError(
                f"Could not find 555/585 columns. Headers found: {cols}\n"
                f"Tip: make sure your headers include '555' and '585' (e.g. '555 nm', '585 nm')."
            )

        a555, a585, labels = [], [], []
        for row in reader:
            try:
                v555 = float(str(row[a555_col]).strip())
                v585 = float(str(row[a585_col]).strip())
            except Exception:
                continue
            a555.append(v555)
            a585.append(v585)
            labels.append(parse_bool_label(row[label_col]) if label_col and (row.get(label_col) is not None) else None)

    return Dataset(a555, a585, labels), a555_col, a585_col, label_col

# ----------------- Metrics & search -----------------
@dataclass
class Scores:
    accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int
    n: int

def eval_rule(a555, a585, labels, delta_thr, ratio_thr, mode) -> Scores:
    tp = tn = fp = fn = 0
    n = 0
    for v555, v585, lab in zip(a555, a585, labels):
        if lab is None:
            continue
        delta = abs(v585 - v555)
        if mode == "delta":
            pred = delta >= delta_thr
        else:
            ratio = v585 / (v555 + 1e-6)
            pred = (delta >= delta_thr) or (ratio >= ratio_thr)
        if lab and pred: tp += 1
        elif lab and not pred: fn += 1
        elif not lab and pred: fp += 1
        else: tn += 1
        n += 1
    acc = (tp + tn) / n if n else 0.0
    return Scores(acc, tp, tn, fp, fn, n)

def percentile(values, q):
    vals = sorted(values)
    if not vals: return 0.0
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(vals) - 1)
    lo = int(pos); hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo]*(1-frac) + vals[hi]*frac

def auto_fit_thresholds(ds: Dataset, mode: str):
    deltas = [abs(b - a) for a, b in zip(ds.a555, ds.a585)]
    ratios = [b / (a + 1e-6) for a, b in zip(ds.a555, ds.a585)]

    # grids (dense near mid-range)
    delta_qs = [0.05,0.10,0.20,0.30,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
    ratio_qs = [0.30,0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
    delta_candidates = sorted(set(percentile(deltas, q) for q in delta_qs))

    if mode == "delta":
        best_delta = delta_candidates[0]
        best_scores = eval_rule(ds.a555, ds.a585, ds.labels, best_delta, None, "delta")
        for dthr in delta_candidates:
            sc = eval_rule(ds.a555, ds.a585, ds.labels, dthr, None, "delta")
            if (sc.accuracy > best_scores.accuracy) or (abs(sc.accuracy - best_scores.accuracy) < 1e-9 and sc.tp > best_scores.tp):
                best_delta, best_scores = dthr, sc
        return best_delta, None, best_scores

    ratio_candidates = sorted(set(percentile(ratios, q) for q in ratio_qs))
    best_delta, best_ratio = delta_candidates[0], ratio_candidates[0]
    best_scores = eval_rule(ds.a555, ds.a585, ds.labels, best_delta, best_ratio, "full")
    for dthr in delta_candidates:
        for rthr in ratio_candidates:
            sc = eval_rule(ds.a555, ds.a585, ds.labels, dthr, rthr, "full")
            if (sc.accuracy > best_scores.accuracy) or (abs(sc.accuracy - best_scores.accuracy) < 1e-9 and sc.tp > best_scores.tp):
                best_delta, best_ratio, best_scores = dthr, rthr, sc
    return best_delta, best_ratio, best_scores

# --------------- Codegen (H / CPP) ---------------
def render_header(delta_thr, ratio_thr):
    ratio_define = f"#define RATIO_THRESH      {ratio_thr:.3f}f\n" if ratio_thr is not None else ""
    return dedent(f"""\
    #ifndef PRESSURE_THRESHOLDS_2_H
    #define PRESSURE_THRESHOLDS_2_H

    // Auto-generated by SQI-2.py — DO NOT EDIT.
    // Gate runs on RAW spectrum (before any preprocessing).

    #define DELTA_PEAK_THRESH {delta_thr:.3f}f
    {ratio_define}// Provide RAW indices for 555/585 nm in your sketch (IDX_555, IDX_585).

    #ifdef __cplusplus
    extern "C" {{
    #endif

    // Δ-only decision
    bool is_pressure_detected(float delta_peak);

    // |A585 - A555|
    float compute_delta_peak(const float* spectrum, int idx555, int idx585);

    // Full decision (Δ OR ratio)
    bool is_pressure_detected_full(float a555, float a585);

    #ifdef __cplusplus
    }} // extern "C"
    #endif

    #endif // PRESSURE_THRESHOLDS_2_H
    """)

def render_source(delta_thr, ratio_thr):
    ratio_block = "    // Ratio criterion disabled in this build.\n"
    if ratio_thr is not None:
        ratio_block = dedent("""\
            // Ratio criterion
            const float ratio = a585 / (a555 + 1e-6f);
            if (ratio >= RATIO_THRESH) {
                return true;
            }
        """)
    return dedent(f"""\
    #include "pressure_thresholds_2.h"
    #include <math.h>

    // Auto-generated by SQI-2.py — DO NOT EDIT.

    bool is_pressure_detected(float delta_peak) {{
        // Use >= to avoid float fencepost misses on the boundary.
        return (delta_peak >= DELTA_PEAK_THRESH);
    }}

    float compute_delta_peak(const float* spectrum, int idx555, int idx585) {{
        const float a555 = spectrum[idx555];
        const float a585 = spectrum[idx585];
        return fabsf(a585 - a555);
    }}

    bool is_pressure_detected_full(float a555, float a585) {{
        const float delta = fabsf(a585 - a555);
        if (delta >= DELTA_PEAK_THRESH) {{
            return true;
        }}
    {ratio_block}    return false;
    }}
    """)

def write_files(outdir: str, header_text: str, source_text: str) -> Tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    hpath = os.path.join(outdir, HEADER_FILENAME)
    spath = os.path.join(outdir, SOURCE_FILENAME)
    with open(hpath, "w", encoding="utf-8") as f: f.write(header_text)
    with open(spath, "w", encoding="utf-8") as f: f.write(source_text)
    return hpath, spath

# --------------------- Main -----------------------
def main():
    print(f"[load] {DATASET_PATH}")
    ds, a555_col, a585_col, label_col = load_csv(DATASET_PATH)
    print(f"[cols] Using columns -> 555: '{a555_col}', 585: '{a585_col}', label: '{label_col}'")
    print(f"[data] rows parsed: {len(ds.a555)}")

    print(f"[fit] Auto-fitting thresholds (mode={MODE}) ...")
    delta_thr, ratio_thr, scores = auto_fit_thresholds(ds, MODE)
    print(f"[fit] thresholds: Δ={delta_thr:.3f}" + ("" if MODE == "delta" else f", ratio={ratio_thr:.3f}"))
    print(f"[fit] accuracy={scores.accuracy*100:.2f}%  TP={scores.tp}  TN={scores.tn}  FP={scores.fp}  FN={scores.fn}")

    htxt = render_header(delta_thr, ratio_thr if MODE == "full" else None)
    ctxt = render_source(delta_thr, ratio_thr if MODE == "full" else None)
    hpath, spath = write_files(OUTDIR, htxt, ctxt)
    print(f"[generate] Wrote {hpath}")
    print(f"[generate] Wrote {spath}")

    # Quick usage hint
    print(dedent(f"""
    -------- Example in your .ino --------
    #include "pressure_thresholds_2.h"
    // const int IDX_555 = {IDX_555};
    // const int IDX_585 = {IDX_585};
    void gate_and_predict(float* raw, int len) {{
        const float a555 = raw[IDX_555];
        const float a585 = raw[IDX_585];
        const float delta = compute_delta_peak(raw, IDX_555, IDX_585);
        // Δ-only:
        if (is_pressure_detected(delta)) {{
            Serial.println("[SQI] Pressure detected — please relax your thumb and rescan.");
            return;
        }}
        // Full (recommended):
        if (is_pressure_detected_full(a555, a585)) {{
            Serial.println("[SQI] Pressure detected — please relax your thumb and rescan.");
            return;
        }}
        // proceed to preprocessing + glucose prediction...
    }}
    --------------------------------------
    """))

if __name__ == "__main__":
    main()
