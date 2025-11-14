# attach_labels_clean.py
import os, csv, re, torch

LABELS_CSV = "labels.csv"
PTH_DIR = "graphs_prepared"

if not os.path.exists(LABELS_CSV):
    print("labels.csv not found. Create it first.")
    raise SystemExit(1)

# Read labels using utf-8-sig to remove BOM if present
labels = {}
with open(LABELS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: 
            continue
        first = row[0].strip()
        if first.startswith('#') or first == '':
            continue
        if len(row) < 2:
            print("Skipping malformed line:", row)
            continue
        key = first
        # normalize: ensure key is like 'sub-123456'
        m = re.search(r'(\d{6})', key)
        if m:
            key_norm = 'sub-' + m.group(1)
        else:
            key_norm = key
        try:
            val = float(row[1])
        except:
            print("Skipping invalid value for", key, ":", row[1])
            continue
        labels[key_norm] = val

print("Loaded labels (normalized keys):", sorted(labels.keys()))
print()

def extract_subid(fn):
    m = re.search(r'(sub-\d{6})', fn)
    if m: 
        return m.group(1)
    m2 = re.search(r'(\d{6})', fn)
    if m2:
        return 'sub-' + m2.group(1)
    return None

def find_label_for_filename(fn):
    sid = extract_subid(fn)
    if sid and sid in labels:
        return sid, labels[sid]
    # fallback: try matching numeric substring to any key
    m = re.search(r'(\d{6})', fn)
    if m:
        digits = m.group(1)
        for k in labels:
            if digits in k:
                return k, labels[k]
    return None, None

updated = 0
total = 0
for fn in sorted(os.listdir(PTH_DIR)):
    total += 1
    path = os.path.join(PTH_DIR, fn)
    try:
        g = torch.load(path)
    except Exception as e:
        print("Could not load", fn, ":", e)
        continue
    key, val = find_label_for_filename(fn)
    if key is not None:
        if isinstance(g, dict):
            g['y'] = torch.tensor([val], dtype=torch.float32)
        else:
            try:
                g.y = torch.tensor([val], dtype=torch.float32)
            except Exception as e:
                print("Could not set y for", fn, ":", e)
                continue
        torch.save(g, path)
        updated += 1
        print(f"Wrote label {val} -> {fn}  (matched key: {key})")
    else:
        print(f"No label found for {fn} (extracted sid: {extract_subid(fn)})")

print(f"\nDone. Labels attached to {updated} / {total} files.")
