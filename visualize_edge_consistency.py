import numpy as np
import matplotlib.pyplot as plt

cons = np.load("edge_consistency.npy")

# Threshold
thr = 0.8
mask = (cons >= thr).astype(int)

plt.figure(figsize=(6,6))
plt.imshow(mask, cmap="viridis")
plt.title(f"Edges with consistency ≥ {thr}")
plt.colorbar()
plt.savefig("edge_consistency_heatmap.png", dpi=300)
print("Saved: edge_consistency_heatmap.png")
