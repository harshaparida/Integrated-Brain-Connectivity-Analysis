#!/usr/bin/env python3
"""
map_rois_to_schaefer.py

Usage (PowerShell or terminal):
  python map_rois_to_schaefer.py --atlas "D:\Thesis\Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"

What it does:
 - Loads the Schaefer atlas NIfTI file.
 - Finds non-zero parcel labels (expected 1..200).
 - Computes the center-of-mass (MNI) for each parcel.
 - Looks for a sidecar label file (CSV/TSV/TXT) in atlas folder with parcel names & network names; if found, uses it.
 - Writes:
      roi_index_to_label.csv
      mask_top_edges_labeled.csv   (if mask_viz/mask_top_edges.csv exists)
      stable_edges_labeled.csv     (if stable_edges.csv exists)

Dependencies:
  pip install nibabel numpy pandas scipy
"""

import os, sys, argparse, glob
import numpy as np
import pandas as pd

try:
    import nibabel as nib
except Exception as e:
    print("ERROR: nibabel required. Install with: pip install nibabel")
    raise

try:
    from scipy import ndimage
except Exception as e:
    print("ERROR: scipy required. Install with: pip install scipy")
    raise

def find_sidecar(atlas_path):
    d = os.path.dirname(atlas_path)
    candidates = []
    for ext in (".csv", ".tsv", ".txt"):
        candidates += glob.glob(os.path.join(d, "*" + ext))
    # Prefer files that contain 'Schaefer' or 'parcel' or 'label' in name
    def score(fn):
        s = fn.lower()
        score = 0
        if "schaefer" in s: score += 2
        if "parcel" in s or "parcellation" in s or "label" in s: score += 1
        return -score, fn  # negative for sort (higher score first)
    candidates = sorted(candidates, key=score)
    return candidates[0] if candidates else None

def try_parse_sidecar(path):
    # attempt to parse common formats
    if path is None:
        return None
    try:
        if path.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", comment="#", engine="python")
        else:
            df = pd.read_csv(path, comment="#", engine="python")
    except Exception:
        try:
            # fallback: whitespace-separated
            df = pd.read_csv(path, sep="\s+", comment="#", engine="python")
        except Exception:
            return None
    # Normalize columns: try to find index/parcel/network/name columns
    cols = {c.lower(): c for c in df.columns}
    # possible column names
    idx_col = None
    for candidate in ("index", "idx", "parcel_id", "parcel", "label_id", "label"):
        if candidate in cols:
            idx_col = cols[candidate]; break
    name_col = None
    for candidate in ("name", "parcel_name", "label_name", "parcellabel", "parcel_label"):
        if candidate in cols:
            name_col = cols[candidate]; break
    network_col = None
    for candidate in ("network", "net", "system", "y", "large_network"):
        if candidate in cols:
            network_col = cols[candidate]; break
    # If we at least have idx -> name or network, create mapping
    if idx_col is None:
        return None
    mapping = {}
    for _, row in df.iterrows():
        try:
            idx = int(row[idx_col])
        except Exception:
            continue
        name = row[name_col] if name_col else f"Parcel_{idx}"
        net = row[network_col] if network_col else "unknown"
        mapping[idx] = {"name": str(name), "network": str(net)}
    return mapping if mapping else None

def compute_centers(atlas_img):
    data = atlas_img.get_fdata().astype(int)
    affine = atlas_img.affine
    labels = np.unique(data)
    labels = labels[labels != 0]  # drop background
    centers = {}
    for lab in labels:
        mask = (data == lab)
        if mask.sum() == 0:
            continue
        # compute center of mass in voxel space then convert to MNI
        com_vox = ndimage.center_of_mass(mask)
        com_vox = np.array(com_vox)
        com_vox = np.append(com_vox, 1.0)  # homogeneous
        mni = affine.dot(com_vox)[:3]
        centers[int(lab)] = tuple([float(x) for x in mni])
    return centers

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", required=True, help="path to Schaefer NIfTI atlas")
    p.add_argument("--mask-edges-csv", default="mask_viz/mask_top_edges.csv", help="edge CSV to label (optional)")
    p.add_argument("--stable-edges-csv", default="stable_edges.csv", help="stable edges CSV to label (optional)")
    args = p.parse_args()

    atlas_path = args.atlas
    if not os.path.exists(atlas_path):
        print("Atlas path not found:", atlas_path)
        sys.exit(1)

    print("Loading atlas:", atlas_path)
    atlas_img = nib.load(atlas_path)
    centers = compute_centers(atlas_img)
    print(f"Found {len(centers)} parcels (non-zero labels).")

    sidecar = find_sidecar(atlas_path)
    mapping = None
    if sidecar:
        print("Attempting to parse sidecar label file:", sidecar)
        mapping = try_parse_sidecar(sidecar)
        if mapping:
            print("Parsed sidecar mapping for", len(mapping), "parcels.")
        else:
            print("Could not parse sidecar:", sidecar)
    else:
        print("No sidecar label file found in atlas folder.")

    # build rows
    rows = []
    for idx in sorted(centers.keys()):
        name = f"Parcel_{idx}"
        net = "unknown"
        if mapping and idx in mapping:
            name = mapping[idx].get("name", name)
            net = mapping[idx].get("network", net)
        rows.append({"index": idx, "parcel_name": name, "network": net,
                     "x_mni": centers[idx][0], "y_mni": centers[idx][1], "z_mni": centers[idx][2]})
    df = pd.DataFrame(rows)
    out_csv = "roi_index_to_label.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote ROI label CSV:", out_csv)

    # Try to label edges CSVs
    def label_edges_csv(in_csv, out_csv):
        if not os.path.exists(in_csv):
            print("Skipped (not found):", in_csv)
            return
        print("Labeling edges in", in_csv)
        edf = pd.read_csv(in_csv)
        # Expect columns node_i,node_j,... or roi_i,roi_j
        possible_i = None
        for c in ("node_i","roi_i","i","from"):
            if c in edf.columns:
                possible_i = c; break
        possible_j = None
        for c in ("node_j","roi_j","j","to"):
            if c in edf.columns:
                possible_j = c; break
        if possible_i is None or possible_j is None:
            print("Could not find node index columns in", in_csv)
            return
        def map_idx(i):
            i = int(i)
            rec = df[df["index"]==i]
            if rec.shape[0]==0:
                return {"name": f"Parcel_{i}", "network": "unknown"}
            return {"name": rec["parcel_name"].values[0], "network": rec["network"].values[0]}
        out_rows = []
        for _, r in edf.iterrows():
            i = int(r[possible_i]); j = int(r[possible_j])
            mi = map_idx(i); mj = map_idx(j)
            new = r.to_dict()
            new.update({
                "roi_i_name": mi["name"], "roi_i_network": mi["network"],
                "roi_j_name": mj["name"], "roi_j_network": mj["network"]
            })
            out_rows.append(new)
        odf = pd.DataFrame(out_rows)
        odf.to_csv(out_csv, index=False)
        print("Wrote labeled edges to", out_csv)

    # label mask_top_edges if exists
    label_edges_csv(args.mask_edges_csv, "mask_top_edges_labeled.csv")
    label_edges_csv(args.stable_edges_csv, "stable_edges_labeled.csv")

    print("Done. If networks are 'unknown', provide the atlas sidecar (CSV/TSV) that maps parcel id -> name -> network.")

if __name__ == "__main__":
    main()
