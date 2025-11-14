import os
import nibabel as nib

base = "D:/Thesis/101007/101006/MNINonLinear/Results"  # change this to folder that contains the 5 folders
folders = ["rfMRI_REST", "rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]

for f in folders:
    fp = os.path.join(base, f)
    if not os.path.exists(fp):
        print(f"{f} : MISSING at {fp}")
        continue
    nifti_files = [os.path.join(fp, x) for x in os.listdir(fp) if x.endswith(('.nii','.nii.gz'))]
    print(f"\nFolder: {f}  -> {len(nifti_files)} nifti files")
    for nf in nifti_files:
        try:
            img = nib.load(nf)
            shape = img.shape
            affine = img.affine
            print("  ", os.path.basename(nf), "shape:", shape)
        except Exception as e:
            print("  ", os.path.basename(nf), "FAILED to load:", e)
