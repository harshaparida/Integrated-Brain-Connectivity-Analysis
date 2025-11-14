# merge_graph_duplicates_clean.py
import os, shutil, torch, numpy as np, re
from pathlib import Path

GRAPHS = Path("graphs_prepared")
BACKUP = Path("graphs_prepared_backup")
BACKUP.mkdir(exist_ok=True)

def load_graph(path):
    g = torch.load(path)
    if isinstance(g, dict):
        return g
    return {
        'x': getattr(g, 'x', None),
        'adj': getattr(g, 'adj', None),
        'y': getattr(g, 'y', None)
    }

def save_graph_dict(d, path):
    torch.save(d, path)

# STEP 1 — Collect unique pairs
pairs = {}
for fn in GRAPHS.glob("*.pt"):
    name = fn.name
    m = re.match(r"(sub-\d{6})(?:_graph)?\.pt$", name)
    if not m:
        continue
    base = m.group(1) + ".pt"
    dup = m.group(1) + "_graph.pt"
    pairs[m.group(1)] = (base, dup)

print("Pairs to process:", pairs)

# STEP 2 — Process each pair only once
for sid, (base_fn, dup_fn) in pairs.items():
    base_path = GRAPHS / base_fn
    dup_path = GRAPHS / dup_fn

    if not base_path.exists():
        print(f"\nSkipping {sid}: base file missing:", base_path)
        continue
    if not dup_path.exists():
        print(f"\nSkipping {sid}: duplicate file missing:", dup_path)
        continue

    print(f"\nProcessing {sid}: merging {dup_fn} → {base_fn}")

    # Backup both
    shutil.copy(base_path, BACKUP / base_fn)
    shutil.copy(dup_path, BACKUP / dup_fn)

    base = load_graph(base_path)
    dup = load_graph(dup_path)

    # Convert x to numpy
    bx = base['x']
    dx = dup['x']
    bx_np = bx.cpu().numpy() if torch.is_tensor(bx) else np.asarray(bx)
    dx_np = dx.cpu().numpy() if (dx is not None and torch.is_tensor(dx)) else (np.asarray(dx) if dx is not None else None)

    # Merge x if needed
    if dx_np is not None and dx_np.shape[1] > bx_np.shape[1]:
        extra = dx_np[:, bx_np.shape[1]:]
        merged = np.concatenate([bx_np, extra], axis=1)
        base['x'] = torch.from_numpy(merged.astype(np.float32))
        print(f"  Added {extra.shape[1]} extra feature columns to base.")
    else:
        print("  No extra x features to merge.")

    # Merge adj
    if base['adj'] is None and dup['adj'] is not None:
        base['adj'] = dup['adj']
        print("  Copied adj from duplicate.")
    else:
        print("  Keeping base adj.")

    # Merge y
    if base['y'] is None and dup['y'] is not None:
        base['y'] = dup['y']
        print("  Copied label y from duplicate.")
    else:
        print("  Keeping base y.")

    save_graph_dict(base, base_path)
    print("  Saved merged graph.")

    # Remove duplicate file
    os.remove(dup_path)
    print("  Removed", dup_fn)

print("\nCleanup complete.")
print("Backups saved in:", BACKUP)
