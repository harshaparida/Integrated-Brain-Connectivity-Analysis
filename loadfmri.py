import nibabel as nib

import numpy as np

import matplotlib.pyplot as plt



# Path to one fMRI file — change to your actual path

file_path = "D:/Thesis/101007/101006/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean_rclean_tclean.nii.gz"



# Load the fMRI NIfTI file

img = nib.load(file_path)

data = img.get_fdata()



print("Shape of fMRI data:", data.shape)

print("Voxel size:", img.header.get_zooms())



# Show a single slice (for visualization)

plt.imshow(data[:, :, 40, 10], cmap="gray")

plt.title("fMRI Slice Example")

plt.show()

