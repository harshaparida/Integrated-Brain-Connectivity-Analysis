#!/usr/bin/env python3
"""
Robust converter: finds *_mean_fc_*.npy and *struct_rois*.npy recursively
and saves paired graphs as .pt in graphs_prepared/
Usage:
    python make_graphs_from_npy.py --data-dir . 
"""
import os, glob, argparse, re, numpy as np, torch
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='.', help='root folder to search for .npy files (recursive)')
    return p.parse_args()

def extract_subid(fn):
    bn = os.path.basename(fn)
    # try 'sub-123456'
    m = re.search(r'(sub-\d{6,})', bn)
    if m:
        return m.group(1)
    # try 6-digit id
    m2 = re.search(r'(\d{6})', bn)
    if m2:
        return 'sub-' + m2.group(1)
    # fallback to filename (unsafe)
    return None

def find_files(root, patterns):
    results = []
    for pat in patterns:
        # recursive glob
        results += glob.glob(os.path.join(root, '**', pat), recursive=True)
    return sorted(list(set(results)))

def main():
    args = parse_args()
    DATA_DIR = os.path.abspath(args.data_dir)
    OUT_DIR = Path('graphs_prepared')
    OUT_DIR.mkdir(exist_ok=True)
    print("Searching for .npy files under:", DATA_DIR)

    # patterns (covers common naming variations)
    fc_patterns = ['*_mean_fc_*.npy', '*_mean_fc*.npy', '*mean_fc*.npy', '*_FC_*.npy', '*_fc_*.npy', '*fc_*.npy']
    struct_patterns = ['*struct_rois*.npy', '*_struct_rois*.npy', '*struct*.npy', '*T1w*struct*.npy']

    fc_files = find_files(DATA_DIR, fc_patterns)
    struct_files = find_files(DATA_DIR, struct_patterns)

    print(f"FC files found: {len(fc_files)}")
    for i,f in enumerate(fc_files[:20]): print(f"  {i+1:02d}. {f}")
    print(f"Struct files found: {len(struct_files)}")
    for i,f in enumerate(struct_files[:20]): print(f"  {i+1:02d}. {f}")

    fc_map = {}
    st_map = {}
    for f in fc_files:
        sid = extract_subid(f)
        if sid:
            fc_map[sid] = f
    for f in struct_files:
        sid = extract_subid(f)
        if sid:
            st_map[sid] = f

    common = sorted(set(fc_map.keys()) & set(st_map.keys()))
    print(f"Subjects with both FC & struct: {len(common)}")
    if len(common) == 0:
        print("No matching subjects found. Possible causes:")
        print(" - files use unexpected naming pattern (inspect the file list above)")
        print(" - missing files for some subjects")
        return

    print("Subjects:", common)
    saved = 0
    for sid in common:
        fc_path = fc_map[sid]
        st_path = st_map[sid]
        try:
            FC = np.load(fc_path)
            X = np.load(st_path)
        except Exception as e:
            print(f"[{sid}] error loading files: {e}. Skipping.")
            continue
        # sanity checks
        if FC.ndim != 2 or FC.shape[0] != FC.shape[1]:
            print(f"[{sid}] FC invalid shape {FC.shape}. Skipping.")
            continue
        if X.ndim != 2 or X.shape[0] != FC.shape[0]:
            print(f"[{sid}] struct/features nodes mismatch {X.shape} vs FC {FC.shape}. Skipping.")
            continue

        graph = {'x': torch.from_numpy(X.astype(np.float32)), 'adj': torch.from_numpy(FC.astype(np.float32)), 'y': None}
        outp = OUT_DIR / f"{sid}.pt"
        torch.save(graph, outp)
        print(f"[{sid}] Saved -> {outp}   x:{X.shape} adj:{FC.shape}")
        saved += 1

    print(f"\nDone. Prepared graphs saved: {saved}  (folder: {OUT_DIR.resolve()})")

if __name__ == '__main__':
    main()
