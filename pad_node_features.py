# pad_node_features.py
import os, torch, numpy as np

PTH = "graphs_prepared"
files = sorted(os.listdir(PTH))
# find max columns
max_cols = 0
shapes = {}
for fn in files:
    g = torch.load(os.path.join(PTH, fn))
    x = g['x'] if isinstance(g, dict) else getattr(g, 'x', None)
    if x is None:
        print("Warning: no x in", fn)
        continue
    x_np = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    shapes[fn] = x_np.shape
    if x_np.shape[1] > max_cols:
        max_cols = x_np.shape[1]

print("Detected x shapes per file:")
for fn in files:
    print(" ", fn, shapes.get(fn))
print("Max columns across graphs:", max_cols)

# If all equal, nothing to do
all_equal = all(s[1] == max_cols for s in shapes.values())
if all_equal:
    print("All x already have same number of columns. No padding needed.")
else:
    print("Padding graphs to", max_cols, "columns.")
    for fn in files:
        path = os.path.join(PTH, fn)
        g = torch.load(path)
        x = g['x'] if isinstance(g, dict) else getattr(g, 'x', None)
        x_np = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        if x_np.shape[1] < max_cols:
            pad_width = max_cols - x_np.shape[1]
            pad = np.zeros((x_np.shape[0], pad_width), dtype=np.float32)
            x_new = np.hstack([x_np, pad])
            # write back
            if isinstance(g, dict):
                g['x'] = torch.from_numpy(x_new)
            else:
                setattr(g, 'x', torch.from_numpy(x_new))
            torch.save(g, path)
            print(" Padded", fn, "-> new x shape:", x_new.shape)
        else:
            print(" No padding for", fn)

print("Done.")
