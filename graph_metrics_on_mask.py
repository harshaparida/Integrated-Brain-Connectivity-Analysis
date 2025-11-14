# graph_metrics_on_mask.py
import numpy as np, networkx as nx, pandas as pd
M = np.load("best_mask_thresh_052.npy")  # produced earlier by visualize_mask.py
G = nx.from_numpy_array(M)
deg = dict(G.degree(weight='weight'))
eig = nx.eigenvector_centrality_numpy(G, weight='weight')
rows = []
for n in sorted(G.nodes()):
    rows.append({'node': n, 'degree': deg.get(n,0), 'eig': eig.get(n,0)})
df = pd.DataFrame(rows)
# attach parcel names if available
import os
if os.path.exists("roi_index_to_label.csv"):
    roi = pd.read_csv("roi_index_to_label.csv").set_index('index')
    df['parcel_name'] = df['node'].map(lambda x: roi.at[x,'parcel_name'] if x in roi.index else f'Parcel_{x}')
    df['network'] = df['node'].map(lambda x: roi.at[x,'network'] if x in roi.index else 'unknown')
df.to_csv("mask_graph_node_stats.csv", index=False)
print("Wrote mask_graph_node_stats.csv")
