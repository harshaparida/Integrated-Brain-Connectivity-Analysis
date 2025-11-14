# compute_network_pair_counts.py
import pandas as pd
df = pd.read_csv("mask_top_edges_labeled.csv")
# ensure columns exist
if not set(['roi_i_network','roi_j_network']).issubset(df.columns):
    raise SystemExit("mask_top_edges_labeled.csv missing roi_i_network / roi_j_network columns")
# canonical pair (sorted) so A-B and B-A count same
pairs = df.apply(lambda r: tuple(sorted([str(r['roi_i_network']), str(r['roi_j_network'])])), axis=1)
counts = pairs.value_counts().reset_index()
counts.columns = ['net_pair','count']
counts[['net_a','net_b']] = pd.DataFrame(counts['net_pair'].tolist(), index=counts.index)
counts = counts[['net_a','net_b','count']].sort_values('count', ascending=False)
counts.to_csv("network_pair_counts.csv", index=False)
print("Wrote network_pair_counts.csv")
print(counts.head(20))
