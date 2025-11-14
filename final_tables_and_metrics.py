# final_tables_and_metrics.py
import pandas as pd
import numpy as np
# load files
edges = pd.read_csv("mask_top_edges_labeled.csv")
rois = pd.read_csv("roi_index_to_label.csv")
nodes_df = pd.read_csv("top_nodes_table.csv")

# convert node to int in nodes_df (first column might be 'node')
nodes_df['node'] = nodes_df['node'].astype(int)

# add parcel name & network to nodes_df
roi_map = rois.set_index('index')[['parcel_name','network']].to_dict(orient='index')
def lookup(n):
    n = int(n)
    if n in roi_map:
        return roi_map[n]['parcel_name'], roi_map[n]['network']
    else:
        return f"Parcel_{n}", "unknown"
nodes_df[['parcel_name','network']] = nodes_df['node'].apply(lambda x: pd.Series(lookup(x)))
nodes_df.to_csv("top_nodes_table_labeled.csv", index=False)
print("Wrote top_nodes_table_labeled.csv")

# produce top edges labeled (if not already)
# ensure node_i/node_j numeric
for c in ['node_i','node_j']:
    if c in edges.columns:
        edges[c] = edges[c].astype(int)
# reorder columns to a nice table
cols = ['node_i','roi_i_name','roi_i_network','node_j','roi_j_name','roi_j_network']
if 'weight' in edges.columns:
    cols += ['weight']
edges[cols].to_csv("top_edges_table_labeled.csv", index=False)
print("Wrote top_edges_table_labeled.csv")
