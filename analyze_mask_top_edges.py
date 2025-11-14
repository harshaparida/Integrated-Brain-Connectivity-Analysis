#!/usr/bin/env python3
"""
analyze_mask_top_edges.py

Reads:   mask_top_edges_labeled.csv
Optional: roi_index_to_label.csv (for node coords & nicer labels)
Outputs:
  - top_edges_table.csv    (top N edges with labels and scores)
  - top_nodes_table.csv    (node importance by edge-weight sum / degree)
  - network_pair_counts.csv
  - connectome_plot.png    (if roi_index_to_label.csv available & nilearn installed)

Usage:
  python analyze_mask_top_edges.py --in mask_top_edges_labeled.csv --topn 50
"""

import argparse, os
import pandas as pd
import numpy as np

def load_edges(path):
    df = pd.read_csv(path)
    # try to normalize column names
    cols = {c.lower(): c for c in df.columns}
    # guess score column name
    score_col = None
    for cand in ("weight","score","consistency","value","mask_value","mask_score"):
        if cand in cols:
            score_col = cols[cand]; break
    if score_col is None:
        # fallback: pick numeric column that's not node indices
        numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if numeric:
            score_col = numeric[-1]
        else:
            raise ValueError("Could not find a score/weight column in edges CSV.")
    # find node index columns
    possible_i = None
    for c in ("node_i","roi_i","i","from"):
        if c in cols:
            possible_i = cols[c]; break
    possible_j = None
    for c in ("node_j","roi_j","j","to"):
        if c in cols:
            possible_j = cols[c]; break
    if possible_i is None or possible_j is None:
        # try first two numeric columns
        numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(numeric) >= 2:
            possible_i, possible_j = numeric[0], numeric[1]
        else:
            raise ValueError("Could not determine node index columns in edges CSV.")
    return df, possible_i, possible_j, score_col

def top_edges(df, i_col, j_col, score_col, topn=50):
    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    return df_sorted.head(topn)

def top_nodes_from_edges(df, i_col, j_col, score_col, topn=50):
    # node importance = sum of scores of incident edges and degree
    nodes = {}
    for _, r in df.iterrows():
        i, j, s = int(r[i_col]), int(r[j_col]), float(r[score_col])
        nodes.setdefault(i, []).append(s)
        nodes.setdefault(j, []).append(s)
    rows = []
    for n, vals in nodes.items():
        rows.append({"node": int(n), "edge_weight_sum": float(np.sum(vals)),
                     "edge_count": int(len(vals)),
                     "edge_weight_mean": float(np.mean(vals))})
    nd = pd.DataFrame(rows).sort_values("edge_weight_sum", ascending=False).reset_index(drop=True)
    return nd.head(topn), nd

def network_pair_counts(df, i_col_name="roi_i_network", j_col_name="roi_j_network"):
    if i_col_name in df.columns and j_col_name in df.columns:
        pairs = df.groupby([i_col_name, j_col_name]).size().reset_index(name="count")
        # also collapse symmetrical pairs
        def canonical(a,b):
            return tuple(sorted([a,b]))
        pairs["canon"] = pairs.apply(lambda r: canonical(r[i_col_name], r[j_col_name]), axis=1)
        agg = pairs.groupby("canon")["count"].sum().reset_index()
        agg[["net_a","net_b"]] = pd.DataFrame(agg["canon"].tolist(), index=agg.index)
        return agg.sort_values("count", ascending=False)[["net_a","net_b","count"]]
    else:
        return None

def try_plot_connectome(top_df, roi_csv, out_png="connectome_plot.png", threshold=0.0):
    try:
        from nilearn import plotting
    except Exception:
        print("nilearn not installed -> skipping connectome plot. Install with: pip install nilearn")
        return False

    roi_df = pd.read_csv(roi_csv)
    # expect index column 'index' and x,y,z columns
    if not set(["index","x_mni","y_mni","z_mni"]).issubset(set(roi_df.columns)):
        print("roi_index_to_label.csv missing required columns (index,x_mni,y_mni,z_mni). Skipping plot.")
        return False

    # get coords and a node list
    coords = {int(r["index"]): (r["x_mni"], r["y_mni"], r["z_mni"]) for _, r in roi_df.iterrows()}
    nodes = sorted({int(x) for x in pd.concat([top_df.iloc[:,0], top_df.iloc[:,1]]).astype(int)})
    coords_list = [coords[n] for n in nodes if n in coords]

    # build adj matrix for those nodes
    N = len(nodes)
    mat = np.zeros((N,N))
    node_to_idx = {n:i for i,n in enumerate(nodes)}
    for _, r in top_df.iterrows():
        i, j, s = int(r.iloc[0]), int(r.iloc[1]), float(r.iloc[-1])
        if s <= threshold: 
            continue
        if i in node_to_idx and j in node_to_idx:
            a, b = node_to_idx[i], node_to_idx[j]
            mat[a,b] = s
            mat[b,a] = s
    # plot
    plotting.plot_connectome(adjacency_matrix=mat, node_coords=coords_list,
                             title="Top mask edges (labeled)", output_file=out_png)
    print("Saved connectome image:", out_png)
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_csv", required=True)
    p.add_argument("--topn", type=int, default=50)
    p.add_argument("--roi", dest="roi_csv", default="roi_index_to_label.csv")
    args = p.parse_args()

    df, i_col, j_col, score_col = load_edges(args.in_csv)
    print("Detected node cols:", i_col, j_col, "score:", score_col)

    topn = top_edges(df, i_col, j_col, score_col, topn=args.topn)
    topn.to_csv("top_edges_table.csv", index=False)
    print("Wrote top_edges_table.csv (top {})".format(args.topn))

    top_nodes, all_nodes = top_nodes_from_edges(df, i_col, j_col, score_col, topn=args.topn)
    top_nodes.to_csv("top_nodes_table.csv", index=False)
    all_nodes.to_csv("all_nodes_table.csv", index=False)
    print("Wrote top_nodes_table.csv and all_nodes_table.csv")

    net_counts = network_pair_counts(df)
    if net_counts is not None:
        net_counts.to_csv("network_pair_counts.csv", index=False)
        print("Wrote network_pair_counts.csv")

    # try plotting if ROI CSV exists
    if os.path.exists(args.roi_csv):
        try_plot_connectome(topn[[i_col,j_col,score_col]], args.roi_csv, out_png="connectome_plot.png")
    else:
        print("No roi_index_to_label.csv found; connectome plot skipped.")

if __name__ == "__main__":
    main()
