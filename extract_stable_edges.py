import numpy as np
import pandas as pd

cons = np.load("edge_consistency.npy")
Q = cons.shape[0]

edges = []
for i in range(Q):
    for j in range(i+1, Q):
        edges.append((i, j, cons[i,j]))

df = pd.DataFrame(edges, columns=["roi_i","roi_j","consistency"])
df = df.sort_values("consistency", ascending=False)

df.to_csv("stable_edges.csv", index=False)
print("Saved stable_edges.csv")
