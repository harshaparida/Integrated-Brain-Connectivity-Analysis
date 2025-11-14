import nibabel as nib
img = nib.load("D:/Thesis/101007/101006/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean_rclean_tclean.nii.gz")
print(img.header.get_zooms())