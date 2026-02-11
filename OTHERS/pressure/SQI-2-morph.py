#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQI-2.py — Build (1) pressure detector AND (2) local Hb morph fixer for ESP32 from your CSV.

Hard-coded dataset path:
  /home/apc-3/PycharmProjects/PythonProjectAK/SQI/dataset/thumb_spectral_pressure.csv

Outputs (in OUTDIR):
  - pressure_thresholds_2_morph.h
  - pressure_thresholds_2_morph.cpp
  - hb_local_morph.h            <-- NEW
  - hb_local_morph.cpp          <-- NEW
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

# Hb morph tuner defaults (you can tweak BLEND_ALPHA here if you lik e)
DEFAULT_BLEND_ALPHA = 0.65
CLAMP_MIN_L, CLAMP_MAX_L = 0.80, 0.95   # allowed box for lower clamp
CLAMP_MIN_U, CLAMP_MAX_U = 1.20, 1.50   # allowed box for upper clamp

# Filenames
HEADER_THRESH = "pressure_thresholds_2_morph.h"
SOURCE_THRESH = "pressure_thresholds_2_morph.cpp"
HEADER_MORPH  = "hb_local_morph.h"
SOURCE_MORPH  = "hb_local_morph.cpp"
# ----------------------------

# ----------------- Data loading -----------------
@dataclass
class Dataset:
    a555: List[float]
    a585: List[float]
    labels: List[Optional[bool]]  # True=yes/with-pressure, False=no/without, None=unknown
    a555_col: str
    a585_col: str
    label_col: Optional[str]

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
    for c in candidates:
        if c in cols:
            return c
    tok = contains_token.lower()
    for c in cols:
        if tok in c.lower():
            return c
    return None

def load_csv(path: str) -> Dataset:
    """
    Robustly locate columns:
      - exact candidates (A555/a555/Abs555/555 nm/etc)
      - OR any header containing '555'/'585' (case-insensitive)
    Label: try 'pressure','label','class', etc.
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

    return Dataset(a555, a585, labels, a555_col, a585_col, label_col)

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

def auto_fit_thresholds(a555, a585, labels, mode: str):
    deltas = [abs(b - a) for a, b in zip(a555, a585)]
    ratios = [b / (a + 1e-6) for a, b in zip(a555, a585)]

    delta_qs = [0.05,0.10,0.20,0.30,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
    ratio_qs = [0.30,0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
    delta_candidates = sorted(set(percentile(deltas, q) for q in delta_qs))

    if mode == "delta":
        best_delta = delta_candidates[0]
        best_scores = eval_rule(a555, a585, labels, best_delta, None, "delta")
        for dthr in delta_candidates:
            sc = eval_rule(a555, a585, labels, dthr, None, "delta")
            if (sc.accuracy > best_scores.accuracy) or (abs(sc.accuracy - best_scores.accuracy) < 1e-9 and sc.tp > best_scores.tp):
                best_delta, best_scores = dthr, sc
        return best_delta, None, best_scores

    ratio_candidates = sorted(set(percentile(ratios, q) for q in ratio_qs))
    best_delta, best_ratio = delta_candidates[0], ratio_candidates[0]
    best_scores = eval_rule(a555, a585, labels, best_delta, best_ratio, "full")
    for dthr in delta_candidates:
        for rthr in ratio_candidates:
            sc = eval_rule(a555, a585, labels, dthr, rthr, "full")
            if (sc.accuracy > best_scores.accuracy) or (abs(sc.accuracy - best_scores.accuracy) < 1e-9 and sc.tp > best_scores.tp):
                best_delta, best_ratio, best_scores = dthr, rthr, sc
    return best_delta, best_ratio, best_scores

# --------------- Codegen: Pressure thresholds (existing) ---------------
def render_header_thresholds(delta_thr, ratio_thr):
    ratio_define = f"#define RATIO_THRESH      {ratio_thr:.3f}f\n" if ratio_thr is not None else ""
    return dedent(f"""\
    #ifndef pressure_thresholds_2_morph_H
    #define pressure_thresholds_2_morph_H

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

    #endif // pressure_thresholds_2_morph_H
    """)

def render_source_thresholds(delta_thr, ratio_thr):
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
    #include "pressure_thresholds_2_morph.h"
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

# --------------- Codegen: Hb local morph (NEW) ---------------
def compute_hb_targets_and_clamps(a555, a585, labels, blend_alpha=DEFAULT_BLEND_ALPHA):
    import math
    import numpy as np

    a555 = np.asarray(a555, dtype=float)
    a585 = np.asarray(a585, dtype=float)
    is_no  = np.array([False if v is None else (not v) for v in labels])
    is_yes = np.array([False if v is None else v       for v in labels])

    delta_no = np.abs(a585[is_no] - a555[is_no])
    ratio_no = a585[is_no] / (a555[is_no] + 1e-6)

    # robust centers as targets
    delta_target = float(np.median(delta_no)) if delta_no.size else 0.062
    ratio_target = float(np.median(ratio_no)) if ratio_no.size else 1.28

    # propose clamps from YES distribution
    delta_yes = np.abs(a585[is_yes] - a555[is_yes])
    ratio_yes = a585[is_yes] / (a555[is_yes] + 1e-6)

    s_delta = np.where(delta_yes < 1e-9, 1.0, delta_target / np.maximum(delta_yes, 1e-9))
    s_ratio = np.where(ratio_yes < 1e-9, 1.0, ratio_target / np.maximum(ratio_yes, 1e-9))
    s_mix   = blend_alpha * s_delta + (1.0 - blend_alpha) * s_ratio

    import numpy as np
    if s_mix.size:
        lo = float(np.percentile(s_mix, 5))
        hi = float(np.percentile(s_mix, 95))
    else:
        lo, hi = 0.9, 1.3

    # safe clipping to avoid wild corrections
    clamp_l = max(CLAMP_MIN_L, min(CLAMP_MAX_L, lo))
    clamp_u = min(CLAMP_MAX_U, max(CLAMP_MIN_U, hi))

    return delta_target, ratio_target, blend_alpha, clamp_l, clamp_u

def render_header_morph(delta_target, ratio_target, blend_alpha, clamp_l, clamp_u):
    return dedent(f"""\
    #ifndef HB_LOCAL_MORPH_H
    #define HB_LOCAL_MORPH_H

    // Auto-generated by SQI-2.py — DO NOT EDIT.
    // Local Hb window fixer (555–585 nm). Call ONLY if pressure was detected.

    #define DELTA_TARGET  {delta_target:.3f}f   // median_no(|A585 - A555|)
    #define RATIO_TARGET  {ratio_target:.2f}f   // median_no(A585 / A555)
    #define BLEND_ALPHA   {blend_alpha:.2f}f    // weight toward Δ target
    #define SCALE_CLAMP_L {clamp_l:.2f}f        // ~5th pct (clipped)
    #define SCALE_CLAMP_U {clamp_u:.2f}f        // ~95th pct (clipped)

    #ifdef __cplusplus
    extern "C" {{
    #endif

    // Modify only 555–585nm window in-place (and gently blend neighbor ~560 nm)
    void local_hb_morph(float* x, int n, int idx555, int idx585);

    #ifdef __cplusplus
    }} // extern "C"
    #endif

    #endif // HB_LOCAL_MORPH_H
    """)

def render_source_morph():
    return dedent("""\
    #include "hb_local_morph.h"
    #include <math.h>

    // Keep changes local to 555–585 nm, preserving the midpoint (local DC).
    void local_hb_morph(float* x, int n, int idx555, int idx585) {
      if (!x || n <= 0) return;
      if (idx555 < 0 || idx585 >= n || idx555 >= idx585) return;

      const int idx560 = (idx555 + 1 < n) ? idx555 + 1 : idx555;  // neighbor (~560 nm)

      float a555 = x[idx555];
      float a585 = x[idx585];
      float delta = fabsf(a585 - a555);
      float ratio = a585 / (a555 + 1e-6f);

      const float m = 0.5f * (a585 + a555);             // midpoint anchor (preserve local baseline)
      const float s_delta = (delta < 1e-6f) ? 1.f : (DELTA_TARGET / fmaxf(delta, 1e-6f));
      const float s_ratio = (ratio < 1e-6f) ? 1.f : (RATIO_TARGET / fmaxf(ratio, 1e-6f));
      float s = BLEND_ALPHA * s_delta + (1.f - BLEND_ALPHA) * s_ratio;

      // Clamp the correction factor to avoid over-shoot
      if (s < SCALE_CLAMP_L) s = SCALE_CLAMP_L;
      if (s > SCALE_CLAMP_U) s = SCALE_CLAMP_U;

      const float new_delta = s * delta;

      float a555_new, a585_new;
      if (a585 >= a555) {
        a555_new = m - 0.5f * new_delta;
        a585_new = m + 0.5f * new_delta;
      } else {
        a555_new = m + 0.5f * new_delta;
        a585_new = m - 0.5f * new_delta;
      }

      // Gently morph neighbor (~560 nm) toward the line between the endpoints
      const float t = (idx560 - idx555) / (float)(idx585 - idx555); // typically ~0.33
      const float a560_line = a555_new + t * (a585_new - a555_new);
      if (idx560 != idx555 && idx560 != idx585) {
        x[idx560] = 0.5f * x[idx560] + 0.5f * a560_line;  // 50% blend for continuity
      }

      x[idx555] = a555_new;
      x[idx585] = a585_new;
    }
    """)

# --------------------- Main -----------------------
def main():
    print(f"[load] {DATASET_PATH}")
    ds = load_csv(DATASET_PATH)
    print(f"[cols] Using columns -> 555: '{ds.a555_col}', 585: '{ds.a585_col}', label: '{ds.label_col}'")
    print(f"[data] rows parsed: {len(ds.a555)}")

    # --- Fit pressure thresholds (existing) ---
    print(f"[fit] Auto-fitting pressure thresholds (mode={MODE}) ...")
    delta_thr, ratio_thr, scores = auto_fit_thresholds(ds.a555, ds.a585, ds.labels, MODE)
    print(f"[fit] thresholds: Δ={delta_thr:.3f}" + ("" if MODE == "delta" else f", ratio={ratio_thr:.3f}"))
    print(f"[fit] accuracy={scores.accuracy*100:.2f}%  TP={scores.tp}  TN={scores.tn}  FP={scores.fp}  FN={scores.fn}")

    # --- Compute Hb morph constants (NEW) ---
    print("[morph] Computing local Hb morph targets/clamps from dataset ...")
    delta_t, ratio_t, alpha, clamp_l, clamp_u = compute_hb_targets_and_clamps(
        ds.a555, ds.a585, ds.labels, blend_alpha=DEFAULT_BLEND_ALPHA
    )
    print(f"[morph] DELTA_TARGET={delta_t:.3f}  RATIO_TARGET={ratio_t:.2f}  "
          f"BLEND_ALPHA={alpha:.2f}  CLAMP_L={clamp_l:.2f}  CLAMP_U={clamp_u:.2f}")

    # --- Write files ---
    os.makedirs(OUTDIR, exist_ok=True)

    # Pressure detector
    htxt = render_header_thresholds(delta_thr, ratio_thr if MODE == "full" else None)
    ctxt = render_source_thresholds(delta_thr, ratio_thr if MODE == "full" else None)
    with open(os.path.join(OUTDIR, HEADER_THRESH), "w", encoding="utf-8") as f: f.write(htxt)
    with open(os.path.join(OUTDIR, SOURCE_THRESH), "w", encoding="utf-8") as f: f.write(ctxt)
    print(f"[generate] Wrote {os.path.join(OUTDIR, HEADER_THRESH)}")
    print(f"[generate] Wrote {os.path.join(OUTDIR, SOURCE_THRESH)}")

    # Hb morph fixer
    mh = render_header_morph(delta_t, ratio_t, alpha, clamp_l, clamp_u)
    ms = render_source_morph()
    with open(os.path.join(OUTDIR, HEADER_MORPH), "w", encoding="utf-8") as f: f.write(mh)
    with open(os.path.join(OUTDIR, SOURCE_MORPH), "w", encoding="utf-8") as f: f.write(ms)
    print(f"[generate] Wrote {os.path.join(OUTDIR, HEADER_MORPH)}")
    print(f"[generate] Wrote {os.path.join(OUTDIR, SOURCE_MORPH)}")

    # Usage hint
    print(dedent(f"""
    -------- Example in your .ino --------
    #include "pressure_thresholds_2_morph.h"
    #include "hb_local_morph.h"

    const int IDX_555 = {IDX_555};
    const int IDX_585 = {IDX_585};

    // inside setup() after you have input_data[19]:
    const float a555 = input_data[IDX_555];
    const float a585 = input_data[IDX_585];

    if (is_pressure_detected_full(a555, a585)) {{
      Serial.println("[SQI] Pressure detected — applying local Hb morph (555–585 nm)...");
      local_hb_morph(input_data, 19, IDX_555, IDX_585);

      // Optional: re-check
      const float delta2 = compute_delta_peak(input_data, IDX_555, IDX_585);
      Serial.print("[SQI] delta after morph = "); Serial.println(delta2, 6);
      if (is_pressure_detected(delta2)) {{
        Serial.println("[SQI] Residual pressure — please relax your thumb and rescan.");
        return;
      }}
    }}
    // then run your existing preprocessing + prediction ...
    --------------------------------------
    """))

if __name__ == "__main__":
    main()
