# inspect_graphs.py
import os
import torch

GRAPHS_DIR = "graphs_prepared"

def safe_to_float(y):
    """Convert tensor/value to Python float if possible."""
    if y is None:
        return None
    try:
        if torch.is_tensor(y):
            return float(y.cpu().numpy().reshape(-1)[0])
        return float(y)
    except:
        return y

print(f"Inspecting graphs in: {GRAPHS_DIR}\n")

files = sorted(os.listdir(GRAPHS_DIR))
if not files:
    print("No files found in graphs_prepared/")
    exit()

for fn in files:
    path = os.path.join(GRAPHS_DIR, fn)
    try:
        g = torch.load(path)
    except Exception as e:
        print(f"{fn}: ERROR loading file → {e}")
        continue

    sid = fn.split(".")[0]

    # Extract x, adj, y safely whether dict or Data object
    if isinstance(g, dict):
        x = g.get("x")
        adj = g.get("adj")
        y = g.get("y")
    else:
        x = getattr(g, "x", None)
        adj = getattr(g, "adj", None)
        y = getattr(g, "y", None)

    # Convert shapes and label
    x_shape = tuple(x.shape) if x is not None else None
    adj_shape = tuple(adj.shape) if adj is not None else None
    label = safe_to_float(y)

    print(f"{sid}:  x={x_shape},  adj={adj_shape},  label={label}")
