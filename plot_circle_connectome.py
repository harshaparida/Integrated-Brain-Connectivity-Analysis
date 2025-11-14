import numpy as np
import pandas as pd
from nilearn import plotting

edges = pd.read_csv("mask_top_edges_labeled.csv")
coords = pd.read_csv("roi_index_to_label.csv")

# center coordinates
coords_xyz = coords[['x_mni','y_mni','z_mni']].values

# Build connectivity matrix
N = coords.shape[0]
conn = np.zeros((N,N))
for _, row in edges.iterrows():
    i = int(row["node_i"])
    j = int(row["node_j"])
    conn[i, j] = row["weight"]
    conn[j, i] = row["weight"]

# Plot
plotting.plot_connectome(
    conn,
    coords_xyz,
    node_size=20,
    edge_cmap='coolwarm',
    edge_threshold='95%',
    output_file='circle_connectome.png',
)
