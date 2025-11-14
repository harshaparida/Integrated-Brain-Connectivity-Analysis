# edge_consistency.py
import glob, numpy as np
files = sorted(glob.glob("best_mask_fold_*.npy"))
Ms = [(np.load(f) > 0.52).astype(int) for f in files]        # use same threshold you visualized
cons = np.mean(np.stack(Ms), axis=0)                        # fraction of folds containing edge
np.save("edge_consistency.npy", cons)
print("Saved edge_consistency.npy; shape:", cons.shape)
