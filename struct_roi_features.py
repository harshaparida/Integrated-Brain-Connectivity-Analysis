# struct_roi_features.py
import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker

# ---------------- USER SETTINGS ----------------
# Paths (change for each subject)
t1_file = r"D:/Thesis/102513/T1w/T1w_acpc_dc_restore_brain.nii.gz"  # your T1w path
# Optional: gray-matter probability map (from SPM/FSL/fMRIPrep). If not available, set to None.
gm_prob_file = r"D:/Thesis/102311/T1w/aparc+aseg.nii.gz"
# Schaefer atlas in MNI (you already downloaded)
atlas_img = r"D:/Thesis/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"

out_dir = r"D:/Thesis/102513/MNINonLinear/Results/roi_outputs"
os.makedirs(out_dir, exist_ok=True)
# ------------------------------------------------

def load_img_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return nib.load(path)

# 1) Load T1 and atlas (resample atlas -> T1 space to ensure exact ROI voxels)
t1 = load_img_safe(t1_file)
atlas = load_img_safe(atlas_img)

# Resample atlas to T1 voxel grid (nearest interpolation to preserve labels)
print("Resampling atlas to T1 space (nearest)...")
atlas_resampled = resample_to_img(atlas, t1, interpolation='nearest')

# 2) Prepare masker with resampled atlas
masker = NiftiLabelsMasker(labels_img=atlas_resampled, standardize=False, detrend=False)
# masker.fit_transform returns (timepoints,x) for 4D; for 3D it will compute mean per ROI
print("Extracting mean intensity per ROI from T1...")
t1_data = masker.fit_transform(t1)  # shape: (n_rois,)  OR (1, n_rois) depending on nilearn version
# make sure we have 1D array of length n_rois
t1_means = np.asarray(t1_data).squeeze()
if t1_means.ndim == 2 and t1_means.shape[0] == 1:
    t1_means = t1_means[0]
n_rois = t1_means.shape[0]
print("n_rois:", n_rois)

# 3) If GM probability map provided, compute mean GM and regional GM volume
gm_means = None
gm_volumes = None
voxel_volume_ml = None

if gm_prob_file and os.path.exists(gm_prob_file):
    print("Loading GM probability map and resampling to T1 space...")
    gm = load_img_safe(gm_prob_file)
    gm_res = resample_to_img(gm, t1, interpolation='continuous')  # keep probabilities smooth

    print("Extracting mean GM probability per ROI...")
    gm_data = masker.fit_transform(gm_res)
    gm_means = np.asarray(gm_data).squeeze()
    if gm_means.ndim == 2 and gm_means.shape[0] == 1:
        gm_means = gm_means[0]

    # compute voxel volume in ml (mm^3 -> ml)
    hdr = t1.header
    zooms = hdr.get_zooms()[:3]
    voxel_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])
    voxel_volume_ml = voxel_vol_mm3 / 1000.0

    # compute number of voxels per ROI by thresholding atlas_resampled
    atlas_arr = atlas_resampled.get_fdata()
    gm_volumes = np.zeros(n_rois, dtype=float)
    for lab in range(1, n_rois + 1):
        mask = (atlas_arr == lab)
        nvox = int(mask.sum())
        # regional GM volume = mean_prob * nvox * voxel_volume_ml
        gm_volumes[lab - 1] = gm_means[lab - 1] * nvox * voxel_volume_ml

    print("Computed regional GM volumes (ml).")
else:
    print("No GM probability map provided or file not found. Skipping GM-derived features.")

# 4) Also compute ROI voxel counts (useful)
atlas_arr = atlas_resampled.get_fdata()
voxel_counts = np.zeros(n_rois, dtype=int)
for lab in range(1, n_rois + 1):
    voxel_counts[lab - 1] = int((atlas_arr == lab).sum())

# 5) Compose feature matrix: columns = [mean_T1, mean_GMprob (if avail), GM_volume_ml (if avail), voxel_count]
features = [t1_means]
colnames = ["mean_T1"]
if gm_means is not None:
    features.append(gm_means)
    colnames.append("mean_GMprob")
if gm_volumes is not None:
    features.append(gm_volumes)
    colnames.append("GM_volume_ml")
features.append(voxel_counts)
colnames.append("voxel_count")

feat_mat = np.vstack(features).T   # shape: (n_rois, n_features)
print("Feature matrix shape:", feat_mat.shape)
print("Columns:", colnames)

# 6) Save
subname = os.path.basename(os.path.dirname(t1_file)) if os.path.dirname(t1_file) else "subject"
out_file = os.path.join(out_dir, f"{subname}_struct_rois.npy")
np.save(out_file, feat_mat)
print("Saved structural ROI features to:", out_file)

# Optional: also save column names for later reference
np.save(os.path.join(out_dir, f"{subname}_struct_rois_colnames.npy"), np.array(colnames))
print("Saved column names.")
