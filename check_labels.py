# check_labels.py
import pandas as pd, glob, os
fns = ["label_fcc.csv","label_ccc.csv","labels.csv"]
for fn in fns:
    if not os.path.exists(fn):
        print("MISSING:", fn)
        continue
    try:
        df = pd.read_csv(fn, header=None)
        # try to detect two-column or single combined column
        if df.shape[1] == 1:
            print(fn, "has 1 column. First 5 lines:")
            print(df.head(5))
        else:
            print(fn, "shape:", df.shape)
            print(df.head(5))
    except Exception as e:
        print("ERROR reading", fn, e)

print("Graph files count:", len(glob.glob("graphs_prepared/*.pt")))
