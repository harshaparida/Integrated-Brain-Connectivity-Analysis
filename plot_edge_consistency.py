import numpy as np
import matplotlib.pyplot as plt

EC = np.load("edge_consistency.npy")  # matrix NxN

plt.figure(figsize=(8,8))
plt.imshow(EC, cmap='viridis')
plt.colorbar(label='Consistency (0–1)')
plt.title("LOOCV Edge Consistency Matrix")
plt.savefig("edge_consistency_heatmap.png", dpi=300)
plt.close()
