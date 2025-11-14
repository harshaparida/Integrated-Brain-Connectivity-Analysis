# debug_labels.py
import csv, os, re
LABELS_CSV = "labels.csv"
PTH_DIR = "graphs_prepared"

labels = {}
with open(LABELS_CSV, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        if row[0].strip().startswith('#'): continue
        sid = row[0].strip()
        val = row[1].strip()
        labels[sid] = val

print("Loaded label keys:", list(labels.keys()))
print()

def extract_sid(fn):
    m = re.search(r'(sub-\d{6})', fn)
    if m: return m.group(1)
    m2 = re.search(r'(\d{6})', fn)
    if m2: return 'sub-' + m2.group(1)
    return None

print("Files and extracted sid:")
for fn in sorted(os.listdir(PTH_DIR)):
    sid = extract_sid(fn)
    print(fn, "->", sid, "label_found:", sid in labels if sid else "No sid")
