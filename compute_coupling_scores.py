#!/usr/bin/env python3
"""
compute_coupling_scores.py

Compute FCC, SCC, and TCC from saved FC and structural ROI files.

- Expects subject FC matrices saved as numpy .npy files (e.g. sub-101006_mean_fc_z.npy).
- Structural files can be:
    - true SC square matrices (n x n), OR
    - structural ROI feature arrays (n_rois x n_features) produced by struct_roi_features.py
    - 1D vectors

- Produces a CSV with columns:
    subject_id, FCC_global, SCC_global, TCC_global, FCC_loocv, SCC_loocv, TCC_loocv

Author: ChatGPT (adapted for your pipeline)
"""
import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ========== USER SETTINGS ==========
DATA_DIR = r"./data"      # directory containing FC and structural files
# use wildcards so glob finds files like:
#   sub-101006_mean_fc_z.npy  and  sub-101006_fc.npy
# and structural ROI files like:
#   T1w_101006struct_rois.npy  or  T1w 101410struct_rois.npy
FC_GLOB = "*_fc*.npy"
SC_GLOB = "*struct_rois*.npy"
OUT_CSV = "coupling_scores.csv"
COMPUTE_LOOCV = True             # whether to compute leave-one-out templates
USE_UPPER_TRIANGLE = True        # use upper-triangle (no diagonal) for FC/SC matrices
VERBOSE = True
# ===================================


def subject_id_from_filename(fname: str) -> str:
    """
    Normalize a filename to a subject id string such as '101006'.
    Handles:
      - sub-101006_mean_fc_z.npy
      - sub_101006_fc.npy
      - T1w_101006struct_rois.npy
      - T1w 101410struct_rois.npy
      - 101006_sc.npy
    """
    base = os.path.basename(fname)
    name = base.replace(".npy", "")

    # unify spaces
    name = name.replace(" ", "_")

    # remove long known suffixes first (longest first)
    for suf in ("_mean_fc_z", "_mean_fc", "_fc_z", "_fc", "struct_rois", "_rois", "_sc"):
        if name.endswith(suf):
            name = name[:-len(suf)]

    # remove common prefixes like sub-, sub_, T1w_, T1w-
    name = re.sub(r'^(sub-|sub_|T1w_|T1w-)', '', name, flags=re.IGNORECASE)

    # remove any non-alphanumeric/underscore/dash characters left
    name = re.sub(r'[^\w\-]', '', name)

    # final cleanup: strip underscores/dashes
    name = name.strip("_-")
    return name


def dedup_sorted_glob(pattern: str, data_dir: str):
    """Return deduplicated, sorted glob results (preserve order)."""
    raw = glob.glob(os.path.join(data_dir, pattern))
    seen = set()
    items = []
    for f in sorted(raw):
        if f not in seen:
            seen.add(f)
            items.append(f)
    return items


def load_matrices_map(pattern: str, data_dir: str):
    """
    Return a dict mapping subject_id -> filepath for files matching pattern.
    If multiple files map to same subject id, the last one found will be used (warned).
    """
    files = dedup_sorted_glob(pattern, data_dir)
    subj_map = {}
    for f in files:
        sid = subject_id_from_filename(f)
        if sid in subj_map:
            # warn when duplicate mapping occurs
            print(f"Warning: multiple files mapped to same subject id '{sid}'. Overwriting: {subj_map[sid]} <- {f}")
        subj_map[sid] = f
    return subj_map


def vectorize_upper_triangle(mat: np.ndarray):
    """Return upper-triangular vector excluding diagonal."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square.")
    i, j = np.triu_indices(mat.shape[0], k=1)
    return mat[i, j]


def robust_spearman(a: np.ndarray, b: np.ndarray):
    """Compute Spearman correlation safely (nan-aware). Returns float (nan if invalid)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Vectors must have same shape for Spearman.")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return np.nan
    r, _ = spearmanr(a[mask], b[mask])
    return float(r)


def main():
    # discover files
    fc_files = load_matrices_map(FC_GLOB, DATA_DIR)
    sc_files = load_matrices_map(SC_GLOB, DATA_DIR)

    if VERBOSE:
        print("Discovered FC files (sid -> file):")
        for sid, f in fc_files.items():
            print(f"  {sid} -> {f}")
        print("Discovered structural files (sid -> file):")
        for sid, f in sc_files.items():
            print(f"  {sid} -> {f}")

    # match subjects present in both modalities
    subjects = sorted(set(fc_files.keys()) & set(sc_files.keys()))
    if len(subjects) == 0:
        raise RuntimeError("No subjects found with both FC and structural files using the given patterns.")

    if VERBOSE:
        print(f"Found {len(subjects)} subjects with both FC and structural files.")

    # -------------------------
    # load FC and SC (or structural features) into vectors
    # -------------------------
    fc_vecs = {}
    sc_vecs = {}

    for sid in subjects:
        fc_path = fc_files[sid]
        sc_path = sc_files[sid]

        # load FC
        fc = np.load(fc_path)
        # produce fc_v
        if fc.ndim == 2 and fc.shape[0] == fc.shape[1] and USE_UPPER_TRIANGLE:
            fc_v = vectorize_upper_triangle(fc)
        else:
            fc_v = fc.flatten()

        # load structural file (may be a square SC matrix OR structural ROI features)
        sc = np.load(sc_path)

        # Case A: true square SC matrix -> vectorize upper triangle
        if sc.ndim == 2 and sc.shape[0] == sc.shape[1] and USE_UPPER_TRIANGLE:
            sc_v = vectorize_upper_triangle(sc)

        # Case B: 2D structural features (n_rois x n_features) -> flatten as fingerprint
        elif sc.ndim == 2 and sc.shape[0] != sc.shape[1]:
            sc_v = sc.flatten()

        # Case C: already 1D (vector) -> use directly
        elif sc.ndim == 1:
            sc_v = sc

        else:
            # fallback: flatten
            sc_v = sc.flatten()

        fc_vecs[sid] = np.asarray(fc_v, dtype=float)
        sc_vecs[sid] = np.asarray(sc_v, dtype=float)

        if VERBOSE:
            print(f"Loaded {sid}: FC shape {fc.shape} -> vec len {fc_vecs[sid].size}; "
                  f"STRUCT shape {sc.shape} -> vec len {sc_vecs[sid].size}")

    # Ensure all FC vectors have equal length, and all SC vectors have equal length
    fc_lengths = {len(v) for v in fc_vecs.values()}
    sc_lengths = {len(v) for v in sc_vecs.values()}
    if len(fc_lengths) != 1:
        raise RuntimeError(f"FC vectors have inconsistent lengths across subjects: {sorted(list(fc_lengths))}")
    if len(sc_lengths) != 1:
        raise RuntimeError(f"Structural vectors have inconsistent lengths across subjects: {sorted(list(sc_lengths))}")

    # Stack to compute group templates
    all_fc_matrix = np.vstack([fc_vecs[s] for s in subjects])
    all_sc_matrix = np.vstack([sc_vecs[s] for s in subjects])

    # Global templates: mean across subjects (axis=0)
    fc_template = np.nanmean(all_fc_matrix, axis=0)
    sc_template = np.nanmean(all_sc_matrix, axis=0)

    results = []
    n_subj = len(subjects)
    sum_fc = np.nansum(all_fc_matrix, axis=0)
    sum_sc = np.nansum(all_sc_matrix, axis=0)

    for idx, sid in enumerate(subjects):
        fc_v = fc_vecs[sid]
        sc_v = sc_vecs[sid]

        # Global correlations
        try:
            FCC_global = robust_spearman(fc_v, fc_template)
        except Exception as e:
            FCC_global = np.nan
            if VERBOSE:
                print(f"Warning: could not compute FCC_global for {sid}: {e}")

        try:
            SCC_global = robust_spearman(sc_v, sc_template)
        except Exception as e:
            SCC_global = np.nan
            if VERBOSE:
                print(f"Warning: could not compute SCC_global for {sid}: {e}")

        TCC_global = np.nanmean([x for x in (FCC_global, SCC_global) if np.isfinite(x)]) if (np.isfinite(FCC_global) or np.isfinite(SCC_global)) else np.nan

        # LOOCV templates (optional)
        FCC_loocv = SCC_loocv = TCC_loocv = np.nan
        if COMPUTE_LOOCV and n_subj > 1:
            loocv_fc_template = (sum_fc - fc_v) / (n_subj - 1)
            loocv_sc_template = (sum_sc - sc_v) / (n_subj - 1)
            try:
                FCC_loocv = robust_spearman(fc_v, loocv_fc_template)
            except Exception:
                FCC_loocv = np.nan
            try:
                SCC_loocv = robust_spearman(sc_v, loocv_sc_template)
            except Exception:
                SCC_loocv = np.nan
            TCC_loocv = np.nanmean([x for x in (FCC_loocv, SCC_loocv) if np.isfinite(x)]) if (np.isfinite(FCC_loocv) or np.isfinite(SCC_loocv)) else np.nan

        results.append({
            "subject_id": sid,
            "FCC_global": FCC_global,
            "SCC_global": SCC_global,
            "TCC_global": TCC_global,
            "FCC_loocv": FCC_loocv,
            "SCC_loocv": SCC_loocv,
            "TCC_loocv": TCC_loocv
        })

        if VERBOSE and (idx % 10 == 0 or idx == n_subj - 1):
            print(f"[{idx+1}/{n_subj}] {sid}: FCC={FCC_global:.4f}, SCC={SCC_global:.4f}, TCC={TCC_global:.4f}")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print("Saved coupling scores to:", OUT_CSV)


if __name__ == "__main__":
    main()
