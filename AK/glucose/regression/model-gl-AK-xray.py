#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot Model XRAY + Verification
----------------------------------
Produces ALL of the following in OUTDIR:

| File                         | Meaning                               |
| ---------------------------- | ------------------------------------- |
| verification_report.txt      | Concrete verdict on preprocessing     |
| XRAY_internal_vs_raw.csv     | Internal early tensor vs raw samples  |
| savedmodel_report.txt        | Model signatures info                 |
| tflite_report.txt            | IO shapes & earliest (N,10) tensor    |
| tflite_ops_full.txt          | All ops (best-effort)                 |
| tflite_tensors_full.csv      | Tensor shapes/dtypes/quantization     |
"""

import os, re, csv
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import savgol_filter

np.set_printoptions(precision=4, suppress=True)

# ========= EDIT THESE 3 PATHS =========
DIR = "/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose/model-NEW-FLOW/10th_r2_0.61_77_"
RAW_CSV  = "/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose/dataset/dataset-1k-glucose/pls/dataset-1k-glucose-pls-top10-csv/Lablink_1k_glucose_RAW_top_10.csv"
SAVEDMODEL_DIR = f"{DIR}/Lablink_1k_glucose_RAW_top_10_Norm_Euc_SNV_1st_Deriv_Baseline_batch64_r2_0.61_77_"
TFLITE_PATH    = f"{DIR}/10th_Lablink_1k_glucose_RAW_top_10_Norm_Euc_SNV_1st_Deriv_Baseline_batch64_r2_0.61_77_.tflite"
OUTDIR         = f"{DIR}/xray_out"
# RAW columns in exact order:
RAW_COLS = ["555 nm","560 nm","585 nm","630 nm","645 nm","680 nm","810 nm","860 nm","900 nm","940 nm"]
# ======================================

def ensure_dir(path): os.makedirs(path, exist_ok=True)
def write_text(path, txt): 
    with open(path, "w", encoding="utf-8") as f: f.write(txt)

# ---------- External pipeline used for verification ----------
def norm_euclidean(X):
    X = np.asarray(X, dtype=np.float32)
    l2 = np.sqrt((X*X).sum(axis=1, keepdims=True)) + 1e-8
    return X / l2

def snv(X):
    X = np.asarray(X, dtype=np.float32)
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    s = np.where(s == 0, 1e-8, s)
    return (X - m) / s

def savgol_deriv(X, window=9, poly=2, deriv=1):
    return savgol_filter(np.asarray(X, np.float32), window_length=window, polyorder=poly, deriv=deriv, axis=1)

def baseline_remove(X):
    X = np.asarray(X, dtype=np.float32)
    return X - X.mean(axis=1, keepdims=True)

def external_pipeline_from_raw(raw10):
    # EXACT sequence used in your training: Norm_Euc -> SNV -> SavGol(d=1,w=9,p=2) -> Baseline
    return baseline_remove(savgol_deriv(snv(norm_euclidean(raw10)), window=9, poly=2, deriv=1)).astype(np.float32)

def corr_cols(A, B):
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)
    B = (B - B.mean(axis=0)) / (B.std(axis=0) + 1e-12)
    return float(np.mean(np.sum(A * B, axis=0) / A.shape[0]))

def mae(A, B): return float(np.mean(np.abs(np.asarray(A) - np.asarray(B))))

# ---------- SavedModel XRAY ----------
def xray_savedmodel(saved_dir: str, outdir: str, sample_shape=(1,10)):
    lines = [f"=== SavedModel X-RAY ===", f"Path: {saved_dir}"]
    sm = tf.saved_model.load(saved_dir)
    sigs = list(sm.signatures.keys())
    lines.append(f"Signatures: {sigs}")
    fn = sm.signatures.get("serving_default", next(iter(sm.signatures.values())))

    try:
        inputs  = {k: v for k, v in fn.structured_input_signature[1].items()}
        outputs = fn.structured_outputs
        lines.append("\n[SavedModel IO]")
        lines.append(f"Inputs : { {k:(v.dtype, v.shape) for k,v in inputs.items()} }")
        lines.append(f"Outputs: { {k:(v.dtype, v.shape) for k,v in outputs.items()} }")
    except Exception as e:
        lines.append(f"Structured IO unavailable: {repr(e)}")

    # Graph scan (signature is a ConcreteFunction)
    try:
        gdef = fn.graph.as_graph_def(add_shapes=True)
        names = [n.name for n in gdef.node][:80]
        lines.append("\n[First ~80 node names]")
        for nm in names: lines.append("• " + nm)

        patt = re.compile(r"(L2|Normalize|Norm|Mean|Sub|Div|Sqrt|Conv1D|Pad|MirrorPad|FusedBatchNorm|BatchNorm|Reshape)", re.I)
        suspects = [(n.name, n.op) for n in gdef.node if patt.search(n.op) or patt.search(n.name)]
        lines.append(f"\n[Suspect ops near input / preprocess-like] (showing up to 40 of {len(suspects)})")
        for nm, op in suspects[:40]:
            lines.append(f"• {op:16s}  {nm}")
    except Exception as e:
        lines.append(f"[Graph scan] skipped: {repr(e)}")

    write_text(os.path.join(outdir, "savedmodel_report.txt"), "\n".join(lines))
    print("Saved →", os.path.join(outdir, "savedmodel_report.txt"))

# ---------- TFLite XRAY ----------
def dump_all_ops_and_tensors(interpreter: tf.lite.Interpreter, outdir: str):
    # Tensors table
    tpath = os.path.join(outdir, "tflite_tensors_full.csv")
    tens = interpreter.get_tensor_details()
    with open(tpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index","name","shape","shape_signature","dtype","quant_scale","quant_zero_point"])
        for t in tens:
            q = t.get("quantization", (None, None))
            w.writerow([t["index"], t.get("name",""), list(t.get("shape",[])), list(t.get("shape_signature",[])), str(t.get("dtype")), q[0], q[1]])
    print("Saved →", tpath)

    # Operators list (best-effort)
    opath = os.path.join(outdir, "tflite_ops_full.txt")
    lines = ["[Operators in order]"]
    try:
        ops = interpreter._get_ops_details()  # private API; very useful
        for i, op in enumerate(ops):
            def fmt(v):
                if isinstance(v, list):
                    return "[" + ", ".join(f"(t{d.get('tensor_index','?')}, shape={d.get('shape',None)})" for d in v if isinstance(d, dict)) + "]"
                return "[]"
            lines.append(f"{i:03d}: {op.get('op_name','?'):25s} inputs:{fmt(op.get('inputs'))} -> outputs:{fmt(op.get('outputs'))}")
    except Exception as e:
        lines.append(f"Op listing not available: {repr(e)}")
    write_text(opath, "\n".join(lines))
    print("Saved →", opath)

def find_earliest_runtime_10d(interpreter: tf.lite.Interpreter,
                              input_idx: int,
                              sample_inputs: np.ndarray | None):
    """
    Pick the first [*,10] activation that actually CHANGES with the input.
    Returns (tensor_index, shape, name) or None.
    """
    # If we can't probe variability, just fall back to the first [*,10]
    fallback = None

    for t in interpreter.get_tensor_details():
        shp = list(t.get("shape_signature", [])) or list(t.get("shape", []))
        if len(shp) == 2 and shp[1] == 10 and (shp[0] in (1, -1)):
            if sample_inputs is None or len(sample_inputs) < 2:
                return t["index"], shp, t.get("name", f"tensor_{t['index']}")  # fallback
            # Probe variability on a few rows
            vals = []
            for i in range(min(3, len(sample_inputs))):
                interpreter.set_tensor(input_idx, sample_inputs[i:i+1].astype(np.float32))
                interpreter.invoke()
                vals.append(np.squeeze(interpreter.get_tensor(t["index"])))
            v = np.vstack(vals)
            if np.max(np.std(v, axis=0)) > 1e-6:
                return t["index"], shp, t.get("name", f"tensor_{t['index']}")
            # remember the first candidate if all are constant
            if fallback is None:
                fallback = (t["index"], shp, t.get("name", f"tensor_{t['index']}"))
    return fallback

def xray_tflite(tfl_path: str, outdir: str, raw_df: pd.DataFrame | None):
    lines = ["=== TFLite X-RAY ===", f"Path: {tfl_path}"]
    if not os.path.isfile(tfl_path):
        lines.append("⚠️ TFLite file not found; skipping.")
        write_text(os.path.join(outdir, "tflite_report.txt"), "\n".join(lines))
        print("Saved →", os.path.join(outdir, "tflite_report.txt"))
        return None

    intr = tf.lite.Interpreter(model_path=tfl_path)
    intr.allocate_tensors()
    inp = intr.get_input_details()[0]
    out = intr.get_output_details()[0]
    lines.append("\n[TFLite IO]")
    lines.append(f"INPUT : {{'index': {inp['index']}, 'dtype': {inp['dtype']}, 'shape': {inp['shape']}, 'quant': {inp.get('quantization')}}}")
    lines.append(f"OUTPUT: {{'index': {out['index']}, 'dtype': {out['dtype']}, 'shape': {out['shape']}, 'quant': {out.get('quantization')}}}")

    # Save IO report
    write_text(os.path.join(outdir, "tflite_report.txt"), "\n".join(lines))
    print("Saved →", os.path.join(outdir, "tflite_report.txt"))

    # Full dumps
    dump_all_ops_and_tensors(intr, outdir)

    # Prepare a small batch of sample inputs from RAW CSV
    Xp = None
    if raw_df is not None and len(raw_df) > 0:
        nprobe = min(50, len(raw_df))
        Xp = raw_df.iloc[:nprobe].astype(np.float32).values

    # Pick a dynamic [*,10] activation (not a constant parameter tensor)
    tap = find_earliest_runtime_10d(intr, inp["index"], Xp)
    side_csv = None
    if tap is not None:
        tap_idx, tap_shape, tap_name = tap
        with open(os.path.join(outdir, "tflite_report.txt"), "a", encoding="utf-8") as f:
            f.write(f"\nEarliest runtime 10D tensor: idx={tap_idx}, shape={tap_shape}, name={tap_name}\n")
        print(f"Earliest runtime 10D tensor: idx={tap_idx}, shape={tap_shape}, name={tap_name}")

        if Xp is not None:
            Z = []
            for i in range(Xp.shape[0]):
                intr.set_tensor(inp["index"], Xp[i:i+1])
                intr.invoke()
                Z.append(np.squeeze(intr.get_tensor(tap_idx)))
            Z = np.vstack(Z)

            df_side = pd.concat([
                pd.DataFrame(Z, columns=[f"early_{j+1}" for j in range(Z.shape[1])]).reset_index(drop=True),
                pd.DataFrame(Xp, columns=[f"raw_{j+1}"   for j in range(Xp.shape[1])]).reset_index(drop=True)
            ], axis=1)

            side_csv = os.path.join(outdir, "XRAY_internal_vs_raw.csv")
            df_side.to_csv(side_csv, index=False)
            print("Saved →", side_csv)

    return side_csv


# ---------- Verification ----------
def verify(outdir: str, side_csv: str | None):
    vpath = os.path.join(outdir, "verification_report.txt")
    if not side_csv or not os.path.isfile(side_csv):
        write_text(vpath, "Verification skipped: no side-by-side CSV found (ensure RAW_CSV exists).")
        print("Saved →", vpath)
        return

    df = pd.read_csv(side_csv)
    early_cols = [c for c in df.columns if c.startswith("early_")]
    raw_cols   = [c for c in df.columns if c.startswith("raw_")]
    if len(early_cols) != 10 or len(raw_cols) != 10:
        write_text(vpath, f"Unexpected column count: early={len(early_cols)} raw={len(raw_cols)}")
        print("Saved →", vpath); return

    EARLY = df[early_cols].to_numpy(np.float32)
    RAW   = df[raw_cols].to_numpy(np.float32)
    PIPE  = external_pipeline_from_raw(RAW.copy())

    c_EP   = corr_cols(EARLY, PIPE)
    mae_EP = mae(EARLY, PIPE)
    c_ER   = corr_cols(EARLY, RAW)
    c_PR   = corr_cols(PIPE,  RAW)

    lines = [
        "=== Verification Report ===",
        f"Rows compared: {EARLY.shape[0]}",
        f"Corr(EARLY , ExternalPipeline) = {c_EP:.6f}",
        f"MAE (EARLY , ExternalPipeline) = {mae_EP:.6e}",
        f"Corr(EARLY , RAW)              = {c_ER:.6f}",
        f"Corr(ExternalPipeline , RAW)   = {c_PR:.6f}",
    ]
    if (c_EP >= 0.95) and (mae_EP < 1e-3):
        lines.append("\nVERDICT: MATCH ✅  (Model’s internal transform ≈ your external pipeline)")
        lines.append("→ Feed RAW (no external preprocess) to avoid double-processing.")
    else:
        lines.append("\nVERDICT: NO MATCH ❌  (Internal transform ≠ SNV+Deriv+Baseline)")
        lines.append("→ Do NOT apply external pipeline for this export; feed RAW 10-band float32 in exact order.")

    write_text(vpath, "\n".join(lines))
    print("Saved →", vpath)

# ---------- Main ----------
def main():
    ensure_dir(OUTDIR)

    # Load RAW CSV (optional but required for verification)
    raw_df = None
    if os.path.isfile(RAW_CSV):
        df = pd.read_csv(RAW_CSV)
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            print("⚠️ Missing RAW columns in CSV:", missing)
        use_cols = [c for c in RAW_COLS if c in df.columns]
        raw_df = df[use_cols].copy()
        print(f"Loaded RAW CSV: {RAW_CSV}  shape={raw_df.shape}")
    else:
        print(f"⚠️ RAW CSV not found: {RAW_CSV} (verification will be skipped)")

    # SavedModel report
    xray_savedmodel(SAVEDMODEL_DIR, OUTDIR)

    # TFLite report + dumps + side-by-side CSV
    side_csv = xray_tflite(TFLITE_PATH, OUTDIR, raw_df)

    # Verification (creates verification_report.txt)
    verify(OUTDIR, side_csv)

if __name__ == "__main__":
    main()
