import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

M = np.load("best_mask.npy")
roi = pd.read_csv("roi_index_to_label.csv")

# Get ordering by network
roi_sorted = roi.sort_values(by="network")
idx = roi_sorted["index"].values - 1

M_sorted = M[idx][:, idx]

plt.figure(figsize=(10,10))
plt.imshow(M_sorted, cmap='coolwarm')
plt.title("Mask Matrix Sorted by Network")
plt.colorbar()
plt.savefig("mask_heatmap_sorted.png", dpi=300)
plt.close()
