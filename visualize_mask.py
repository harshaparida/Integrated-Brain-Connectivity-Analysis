# visualize_mask.py
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

# Optional: try to use nilearn if available for brain connectome plotting
try:
    from nilearn import plotting
    HAVE_NILEARN = True
except Exception:
    HAVE_NILEARN = False

MASK_PATH = "best_mask.npy"
GRAPHS_DIR = "graphs_prepared"  # to find one example FC to multiply with mask
THRESH = 0.52        # paper used 0.52 for visualization; change if needed
TOP_EDGES = 150      # number of edges to display in network chord layout

out_dir = Path("mask_viz")
out_dir.mkdir(exist_ok=True)

# load mask
M = np.load(MASK_PATH)
Q = M.shape[0]
print("Loaded mask shape:", M.shape)

# threshold
M_thresh = (M > THRESH).astype(int)
np.save(out_dir / "best_mask_thresh_052.npy", M_thresh)
print("Saved thresholded mask:", out_dir / "best_mask_thresh_052.npy")

# produce weighted mask for plotting (mask * example adjacency)
# find any graph to use its adjacency for edge weights
import torch, glob
example_adj = None
for p in sorted(glob.glob(os.path.join(GRAPHS_DIR, "*.pt"))):
    g = torch.load(p)
    if isinstance(g, dict):
        adj = g.get('adj', None)
    else:
        adj = getattr(g, "adj", None)
    if adj is None:
        continue
    if hasattr(adj, "numpy"):
        adj = adj.numpy()
    example_adj = adj
    break

if example_adj is None:
    print("Warning: no adjacency found in graphs_prepared/*.pt — using mask binary for plotting.")
    weighted = M * 1.0
else:
    # minmax scale example adj (upper triangular) similar to train preprocessing
    tri = np.triu_indices(Q, k=1)
    vals = example_adj[tri]
    if np.allclose(vals, vals[0]):
        scaled = np.zeros_like(vals)
    else:
        mn, mx = vals.min(), vals.max()
        scaled = (vals - mn) / (mx - mn + 1e-12)
    A_scaled = np.zeros_like(example_adj)
    A_scaled[tri] = scaled
    A_scaled = A_scaled + A_scaled.T
    weighted = M * A_scaled

# Build networkx graph using top absolute weights
flat_idx = np.triu_indices(Q, k=1)
weights = weighted[flat_idx]
order = np.argsort(weights)[::-1]
top_idx = order[:TOP_EDGES]
edges = []
for idx in top_idx:
    i = flat_idx[0][idx]; j = flat_idx[1][idx]
    w = weights[idx]
    if w <= 0: continue
    edges.append((i, j, w))

G = nx.Graph()
G.add_nodes_from(range(Q))
for i,j,w in edges:
    G.add_edge(i, j, weight=float(w))

# Node sizes by degree
deg = dict(G.degree(weight='weight'))
node_sizes = [50 + 400 * (deg.get(i, 0) / (max(deg.values()) + 1e-12)) for i in range(Q)]
node_colors = [deg.get(i, 0) for i in range(Q)]

# If nilearn available and you have ROI coords, draw on brain
coords_path = "roi_coords.npy"  # optional coords (Q x 3)
if HAVE_NILEARN and os.path.exists(coords_path):
    coords = np.load(coords_path)
    print("Using ROI coords from", coords_path)
    plotting.plot_connectome(weighted, coords, threshold=0.0, edge_threshold="90%", title="Weighted Masked Connectome",
                             output_file=str(out_dir / "connectome_nilearn.png"))
    print("Saved:", out_dir / "connectome_nilearn.png")
else:
    # fallback: circular layout
    pos = nx.circular_layout(G)
    plt.figure(figsize=(10,10))
    weights_plot = [d['weight'] for _,_,d in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, width=[2*(w/max(weights_plot)+1e-6) for w in weights_plot], alpha=0.8)
    # label top nodes only
    top_nodes = sorted(deg.keys(), key=lambda k: deg[k], reverse=True)[:20]
    labels = {n: str(n) for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title("Masked connectome - circular layout")
    plt.axis('off')
    plt.savefig(out_dir / "connectome_circular.png", bbox_inches='tight', dpi=200)
    print("Saved:", out_dir / "connectome_circular.png")

# also save edge list as CSV
import csv
edges_csv = out_dir / "mask_top_edges.csv"
with open(edges_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_i", "node_j", "weight"])
    for i,j,w in edges:
        writer.writerow([i, j, w])
print("Saved edge list:", edges_csv)

print("Done. Visualizations & CSV in", out_dir)
