# roi_extract_and_fc.py
import os
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.signal import clean


# === USER SETTINGS ===
base = r"D:/Thesis/102513/MNINonLinear/Results"   # folder with volumes
runs = {
    "rfMRI_REST1_LR": "rfMRI_REST1_LR_hp2000_clean_rclean_tclean.nii.gz",
    "rfMRI_REST1_RL": "rfMRI_REST1_RL_hp2000_clean_rclean_tclean.nii.gz",
    "rfMRI_REST2_LR": "rfMRI_REST2_LR_hp2000_clean_rclean_tclean.nii.gz",
    "rfMRI_REST2_RL": "rfMRI_REST2_RL_hp2000_clean_rclean_tclean.nii.gz",
    # "rfMRI_REST1_LR": "rfMRI_REST1_LR_hp2000_clean_rclean_tclean.nii.gz",
    # "rfMRI_REST1_RL": "rfMRI_REST1_RL_hp2000_clean_rclean_tclean.nii.gz",
}
atlas_img = r"D:/Thesis/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
outdir = os.path.join(base, "roi_outputs")
os.makedirs(outdir, exist_ok=True)

TR = 0.72
low_pass = 0.1
high_pass = 0.01
standardize = True
detrend = True

# make masker (uses little memory)
masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False,
                           smoothing_fwhm=None, detrend=False,
                           memory=None, verbose=1)

for run_name, fname in runs.items():
    fpath = os.path.join(base, run_name, fname) if os.path.dirname(fname)=='' else os.path.join(base, fname)
    if not os.path.exists(fpath):
        print("Missing:", fpath); continue
    print(f"\nProcessing run: {run_name}\n  file: {fpath}")

    # use nibabel with mmap to check shape quickly
    img = nib.load(fpath, mmap=True)
    print("  img shape:", img.shape)

    # Extract ROI time-series (time x n_rois) — lightweight
    ts = masker.fit_transform(fpath)   # result is in memory but tiny: (timepoints, n_rois)
    print("  extracted time series shape:", ts.shape)

    # Clean ROI time-series (temporal filtering + standardize)
    ts_clean = clean(ts, detrend=detrend, standardize=standardize, t_r=TR, low_pass=low_pass, high_pass=high_pass)
    print("  cleaned TS shape:", ts_clean.shape)

    # Save time-series
    ts_path = os.path.join(outdir, f"{run_name}_roits.npy")
    np.save(ts_path, ts_clean)
    print("  saved timeseries:", ts_path)

    # Compute functional connectivity (Fisher z of correlation)
    conn = ConnectivityMeasure(kind="correlation")
    fc = conn.fit_transform([ts_clean])[0]   # (n_rois, n_rois)
    # fisher z:
    fc_z = np.arctanh(np.clip(fc, -0.999999, 0.999999))
    fc_path = os.path.join(outdir, f"{run_name}_fc_z.npy")
    np.save(fc_path, fc_z)
    print("  saved FC (z) matrix:", fc_path)

print("\nDone. ROI time-series and FC matrices are in:", outdir)
