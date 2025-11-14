import numpy as np

# Load learned mask
M = np.load("best_mask.npy")

# Threshold (0.52 is from the paper)
thr = 0.52
M_bin = (M > thr).astype(float)

# Save
np.save("best_mask_thresh_052.npy", M_bin)

print("Saved best_mask_thresh_052.npy with threshold =", thr)
print("Non-zero edges:", M_bin.sum())
