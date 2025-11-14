# pair_and_prepare_all.py
import os, glob, re, shutil, subprocess
from pprint import pprint

DATA_DIR = "data"
BACKUP_DIR = os.path.join(DATA_DIR, "backup_structs")
PREPARE_SCRIPT = "prepare_graph.py"   # will call this after renaming
DRY_RUN = False  # set True to only show planned changes without performing them

def list_npys():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
    return [os.path.basename(p) for p in files]

def find_fc_files():
    patterns = ["sub-101006_mean_fc_z.npy", "sub-101410_mean_fc_z (2).npy", "sub-101410_mean_fc_z.npy", "sub-101410_mean_fc_z.npy","sub-102008_mean_fc_z.npy","sub-102109_mean_fc_z.npy","sub-102311_mean_fc_z (2).npy","sub-102311_mean_fc_z.npy"]
    found = []
    for p in patterns:
        found += glob.glob(os.path.join(DATA_DIR, p))
    return sorted(list(set(found)))

def find_struct_candidates():
    patterns = ["T1w_struct1_rois.npy", "T1w_struct_rois.npy", "T1w_struct_rois (6).npy", "T1w_struct_rois (5).npy","T1w_struct_rois (4).npy", "T1w_struct_rois (3).npy", "T1w_struct_rois (2).npy","sub-101006_struct_rois.npy"]
    found = []
    for p in patterns:
        found += glob.glob(os.path.join(DATA_DIR, p))
    # exclude any files that already look like sub-XXX_struct_rois.npy
    filtered = []
    for f in sorted(set(found)):
        b = os.path.basename(f)
        if re.search(r"sub[-_]\d{3,7}.*struct", b, flags=re.IGNORECASE):
            # likely already named correctly, skip from candidates
            continue
        filtered.append(f)
    return sorted(filtered)

def extract_subid_from_fc(fname):
    b = os.path.basename(fname)
    m = re.search(r"(sub[-_]\d{3,7})", b, flags=re.IGNORECASE)
    if m:
        sid = m.group(1).replace("_","-")
        return sid
    m2 = re.search(r"(\d{3,7})", b)
    if m2:
        return "sub-" + m2.group(1)
    # else fallback to basename
    return os.path.splitext(b)[0]

def plan_and_execute():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    fc_paths = find_fc_files()
    struct_paths = find_struct_candidates()
    print("Files in data/:")
    pprint(list_npys())
    print("\nDetected FC files (count={}):".format(len(fc_paths)))
    for p in fc_paths: print("  ", os.path.basename(p))
    print("\nDetected struct candidate files (count={}):".format(len(struct_paths)))
    for p in struct_paths: print("  ", os.path.basename(p))

    if len(fc_paths) == 0:
        print("\nERROR: No FC files detected by patterns. Ensure fc files exist in data/ and contain 'fc' in filename.")
        return

    if len(struct_paths) == 0:
        print("\nERROR: No struct candidate files detected. If your struct files are already named with sub-ids, they may not need renaming.")
        return

    n_pairs = min(len(fc_paths), len(struct_paths))
    print("\nWill pair the first {} FC files with the first {} struct candidates (sorted order).".format(n_pairs, n_pairs))

    planned = []
    for i in range(n_pairs):
        fc = fc_paths[i]
        struct = struct_paths[i]
        sid = extract_subid_from_fc(fc)
        newname = f"{sid}_struct_rois.npy"
        planned.append((os.path.basename(fc), os.path.basename(struct), newname))

    print("\nPlanned renames (FC filename -> struct original -> struct renamed to):")
    for fc, struct, newname in planned:
        print(f"  {fc}  <--  {struct}  =>  {newname}")

    if DRY_RUN:
        print("\nDRY RUN enabled - no files will be changed. Set DRY_RUN=False to execute.")
        return

    # backup and rename
    for i in range(n_pairs):
        _, struct, newname = planned[i]
        struct_path = os.path.join(DATA_DIR, struct)
        new_path = os.path.join(DATA_DIR, newname)
        backup_path = os.path.join(BACKUP_DIR, struct)
        print(f"\nBacking up {struct} -> {os.path.basename(backup_path)}")
        shutil.copy2(struct_path, backup_path)
        # if new_path exists, create a unique name to avoid overwriting
        if os.path.exists(new_path):
            base, ext = os.path.splitext(new_path)
            j = 1
            unique = f"{base}_{j}{ext}"
            while os.path.exists(unique):
                j += 1
                unique = f"{base}_{j}{ext}"
            print(f"Target {newname} already exists, will rename to {os.path.basename(unique)} instead.")
            new_path = unique
        print(f"Renaming {struct} -> {os.path.basename(new_path)}")
        shutil.move(struct_path, new_path)

    print("\nAll renames done. Backup copies saved in:", BACKUP_DIR)
    print("Now running prepare_graph.py to build graphs...")

    # run prepare_graph.py
    if os.path.exists(PREPARE_SCRIPT):
        try:
            res = subprocess.run(["python", PREPARE_SCRIPT], check=False, capture_output=True, text=True)
            print("\n--- prepare_graph.py output ---\n")
            print(res.stdout)
            if res.stderr:
                print("\n--- prepare_graph.py errors ---\n")
                print(res.stderr)
        except Exception as e:
            print("Failed to run prepare_graph.py:", e)
    else:
        print("prepare_graph.py not found in repo root. Please run it manually after verifying files.")

if __name__ == "__main__":
    plan_and_execute()