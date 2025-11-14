# attach_labels_robust.py
import pandas as pd
import glob, torch, os, sys
from torch_geometric.data import Data
import warnings

def load_labels(path):
    # tolerant CSV read
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path)
    if df.shape[1] == 1:
        # split "sub-101006,102.5"
        df = df[0].str.split(',', expand=True)
    df = df.iloc[:, :2].copy()
    df.columns = ['subject','label']
    df['subject'] = df['subject'].astype(str).str.strip().apply(lambda s: s if s.startswith('sub-') else 'sub-'+s)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    labels = dict(zip(df['subject'], df['label']))
    return labels

def extract_data_obj(loaded):
    """
    Return a torch_geometric.data.Data object if possible, else None.
    Handles:
      - Data object
      - dict containing Data under key 'data' or 'graph' or similar
      - dict with keys 'x' and 'edge_index' -> construct Data(**dict)
    """
    # case 1: already Data
    if isinstance(loaded, Data):
        return loaded
    # case 2: dict containing Data value
    if isinstance(loaded, dict):
        # try common keys
        for k in ['data','graph','g','pyg_data','sample']:
            if k in loaded and isinstance(loaded[k], Data):
                return loaded[k]
        # try to find any Data instance in values
        for v in loaded.values():
            if isinstance(v, Data):
                return v
        # try to construct Data from dict if it contains core keys
        if 'x' in loaded and 'edge_index' in loaded:
            try:
                # ensure tensors
                x = torch.as_tensor(loaded['x']) if not isinstance(loaded['x'], torch.Tensor) else loaded['x']
                edge_index = torch.as_tensor(loaded['edge_index']) if not isinstance(loaded['edge_index'], torch.Tensor) else loaded['edge_index']
                kwargs = {'x': x, 'edge_index': edge_index}
                # optional keys
                for kk in ['edge_attr','y','pos','num_nodes']:
                    if kk in loaded:
                        kwargs[kk] = torch.as_tensor(loaded[kk]) if not isinstance(loaded[kk], torch.Tensor) else loaded[kk]
                return Data(**kwargs)
            except Exception:
                return None
    # case 3: tuple/list with Data inside
    if isinstance(loaded, (list, tuple)):
        for item in loaded:
            if isinstance(item, Data):
                return item
    return None

def attach_one_file(fpath, label_value):
    loaded = torch.load(fpath)   # keep default weights_only behavior for compatibility
    data_obj = extract_data_obj(loaded)
    if data_obj is None:
        # print diagnostics and return False
        print(f"[SKIP] Could not extract Data object from {fpath}. Loaded type: {type(loaded)}")
        # show keys if dict
        if isinstance(loaded, dict):
            print("  dict keys:", list(loaded.keys())[:20])
        return False
    # attach label (ensure shape is scalar tensor)
    try:
        data_obj.y = torch.tensor([float(label_value)], dtype=torch.float32)
    except Exception as e:
        print(f"[ERROR] could not set y for {fpath}: {e}")
        return False
    return data_obj

def attach(graphs_dir, labels_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = load_labels(labels_path)
    missing = []
    attached = 0
    for f in sorted(glob.glob(os.path.join(graphs_dir, "*.pt"))):
        fname = os.path.basename(f)
        # try to infer subject id from filename
        sid = None
        tokens = fname.replace('.pt','').split('_')
        for t in tokens:
            if t.startswith('sub-'):
                sid = t; break
        if sid is None:
            sid = os.path.splitext(fname)[0]
            if not sid.startswith('sub-'):
                sid = 'sub-'+sid
        if sid not in labels or pd.isna(labels[sid]):
            missing.append(sid)
            continue
        data_obj = attach_one_file(f, labels[sid])
        if data_obj is False:
            # could not extract, skip
            continue
        # save; preserve filename
        outpath = os.path.join(out_dir, fname)
        torch.save(data_obj, outpath)
        attached += 1
    print(f"Attached labels from {labels_path} to {attached} graphs. Missing labels for {len(missing)} subjects.")
    if missing:
        print("Missing examples (first 10):", missing[:10])

if __name__ == "__main__":
    # adjust these as needed
    attach("graphs_prepared", "label_fcc.csv", "graphs_labeled_fcc")
    attach("graphs_prepared", "label_ccc.csv", "graphs_labeled_ccc")
    attach("graphs_prepared", "labels.csv",    "graphs_labeled_tcc")
