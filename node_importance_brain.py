import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.image import new_img_like
import nibabel as nib

# Load your node statistics
nodes = pd.read_csv("mask_graph_node_stats.csv")
coords = pd.read_csv("roi_index_to_label.csv")

coords_xyz = coords[['x_mni','y_mni','z_mni']].values

# Node size scaling (degree)
sizes = nodes['degree'] / nodes['degree'].max() * 80 + 10
colors = nodes['eig']

# --- Create a dummy empty brain image (Nilearn requires one) ---
# Create a zero volume matching MNI152 2mm template
shape = (91, 109, 91)  # standard MNI152 2mm
affine = np.eye(4)
dummy_data = np.zeros(shape)
dummy_img = nib.Nifti1Image(dummy_data, affine)

# --- Plot Glass Brain ---
display = plotting.plot_glass_brain(
    dummy_img,
    display_mode='lyrz',
    title='Node Importance (Degree/Eigenvector)'
)

# --- Add markers (nodes) ---
display.add_markers(
    marker_coords=[tuple(coord) for coord in coords_xyz],
    marker_size=sizes.tolist(),
    marker_color=colors.tolist()
)

display.savefig("node_importance_glassbrain.png", dpi=300)
print("Saved node_importance_glassbrain.png")
