# quick_view_mask.py
import numpy as np
M = np.load("best_mask.npy")
print("mask shape:", M.shape)
print("mask min/max:", M.min(), M.max())
print("top-10 mask values:", np.sort(M.flatten())[::-1][:10])
