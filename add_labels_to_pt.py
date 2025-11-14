# add_labels_to_pt.py
import csv, os, torch

LABELS_CSV = "labels.csv"
PTH_DIR = "graphs_prepared"

# read labels
labels = {}
with open(LABELS_CSV, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        # ignore comment or header lines that start with '#'
        if row[0].strip().startswith('#'): continue
        sid = row[0].strip()
        try:
            val = float(row[1])
        except:
            print("Skipping invalid line:", row)
            continue
        labels[sid] = val

if len(labels) == 0:
    print("No labels found in", LABELS_CSV)
else:
    print("Loaded labels for", len(labels), "subjects")

# attach labels
updated = 0
for fn in sorted(os.listdir(PTH_DIR)):
    path = os.path.join(PTH_DIR, fn)
    try:
        g = torch.load(path)
    except Exception as e:
        print("Could not load", fn, ":", e)
        continue
    sid = fn.split('.')[0]
    if sid in labels:
        if isinstance(g, dict):
            g['y'] = torch.tensor([labels[sid]], dtype=torch.float32)
        else:
            try:
                g.y = torch.tensor([labels[sid]], dtype=torch.float32)
            except Exception as e:
                print("Could not set y for", fn, ":", e)
                continue
        torch.save(g, path)
        updated += 1
        print("Wrote label", labels[sid], "->", fn)
    else:
        print("No label in CSV for", sid)

print(f"Finished. Labels attached to {updated} / {len(os.listdir(PTH_DIR))} graphs.")
