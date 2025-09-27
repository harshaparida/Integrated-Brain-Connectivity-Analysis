# 🧠 MRI Preprocessing and Feature Extraction Pipeline

This repository documents the workflow for **converting DICOM MRI scans to NIfTI format** and preprocessing them using the **Human Connectome Project (HCP) Minimal Preprocessing Pipelines**.

The end goal is to prepare cleaned MRI/fMRI data for **feature extraction** and **matrix creation**, as described in the referenced research paper.

---

## 📂 Workflow Overview

1. **Data Conversion**

   * Convert raw **DICOM (.dcm)** files to **NIfTI (.nii)** format.
   * Tools: [`dcm2niix`](https://github.com/rordenlab/dcm2niix), [`dicom2nifti`](https://pypi.org/project/dicom2nifti/), or **MRIcroGL**.

2. **Preprocessing Pipelines (HCP Minimal Pipelines)**

   * Run **PreFreeSurferPipeline** for initial alignment & normalization.
   * Run **FreeSurferPipeline** for structural surface reconstruction.
   * Run **fMRIVolumePipeline** for fMRI data preprocessing.

3. **Feature Extraction**

   * Extract brain connectivity features (not yet covered here).
   * Generate matrices for downstream analysis.

---

## ⚙️ Installation & Dependencies

Before running the pipeline, install the following tools:

* [**FSL (FMRIB Software Library)**](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index)
* [**FreeSurfer**](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)

  * Requires `license.txt` (obtain from [FreeSurfer Registration](https://surfer.nmr.mgh.harvard.edu/registration.html))
* [**HCP Pipelines**](https://github.com/Washington-University/HCPpipelines)
* [**Connectome Workbench**](https://www.humanconnectome.org/software/connectome-workbench)

---

## 🔄 Step 1: Convert DICOM → NIfTI

Example using **MRIcroGL**:

```bash
dcm2niix -o /mnt/d/brain/NIfTI/I1092241 \
         -f I1092241_Axial_rsfMRI_%p_%s \
         /mnt/d/brain/DICOM/I1092241
```

Resulting file:

```
I1092241_Axial_rsfMRI_(EYES_OPEN)_20181213125500_9.nii.gz
```

---

## 🔄 Step 2: Run HCP Pipelines

### 1. Source Setup Script

```bash
source /mnt/d/brain/HCPpipelines-master/Examples/Scripts/SetUpHCPPipeline.sh
```

### 2. PreFreeSurfer Pipeline (T1w + T2w)

```bash
/mnt/d/brain/HCPpipelines-master/PreFreeSurfer/PreFreeSurferPipeline.sh \
--path=/mnt/d/brain \
--subject=I1092241 \
--t1=/mnt/d/brain/NIfTI/I1092241/unprocessed/3T/T1w.nii.gz \
--t2=/mnt/d/brain/NIfTI/I1092241/unprocessed/3T/T2w.nii.gz \
--t1template=${HCPPIPEDIR_Templates}/MNI152_T1_1mm.nii.gz \
--t1templatebrain=${HCPPIPEDIR_Templates}/MNI152_T1_1mm_brain.nii.gz \
--t1template2mm=${HCPPIPEDIR_Templates}/MNI152_T1_2mm.nii.gz \
--t2template=${HCPPIPEDIR_Templates}/MNI152_T2_1mm.nii.gz \
--t2templatebrain=${HCPPIPEDIR_Templates}/MNI152_T2_1mm_brain.nii.gz \
--t2template2mm=${HCPPIPEDIR_Templates}/MNI152_T2_2mm.nii.gz \
--templatemask=${HCPPIPEDIR_Templates}/MNI152_T1_1mm_brain_mask.nii.gz \
--template2mmmask=${HCPPIPEDIR_Templates}/MNI152_T1_2mm_brain_mask_dil.nii.gz \
--brainsize=150 \
--fnirtconfig=${HCPPIPEDIR_Config}/T1_2_MNI152_2mm.cnf \
--seechospacing=0.0007 \
--seunwarpdir=y \
--t1samplespacing=0.0005 \
--t2samplespacing=0.0005 \
--unwarpdir=y \
--gdcoeffs=NONE \
--avgrdcmethod=FUGUE
```

### 3. FreeSurfer Pipeline

```bash
/mnt/d/brain/HCPpipelines-master/FreeSurfer/FreeSurferPipeline.sh \
--session=I1092241 \
--session-dir=/mnt/d/brain/I1092241 \
--t1w-image=/mnt/d/brain/I1092241/T1w/T1w_acpc_dc_restore.nii.gz \
--t1w-brain=/mnt/d/brain/I1092241/T1w/T1w_acpc_dc_restore_brain.nii.gz \
--t2w-image=/mnt/d/brain/I1092241/T2w/T2w_acpc_dc_restore.nii.gz
```

### 4. fMRIVolume Pipeline

```bash
/mnt/d/brain/HCPpipelines-master/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh \
--studyfolder="/mnt/d/brain" \
--session="I1092241" \
--fmritcs="/mnt/d/brain/NIfTI/I1092241/I1092241_Axial_rsfMRI_(EYES_OPEN)_20181213125500_9.nii.gz" \
--fmriname="Axial_rsfMRI_Eyes_Open_20181213" \
--fmrires="2" \
--biascorrection="NONE" \
--dcmethod="NONE" \
--gdcoeffs="NONE" \
--processing-mode="LegacyStyleData"
```

---

## 🛠 Troubleshooting

* ❌ `ERROR: unrecognized option: '--subjects-dir'`

  * ✅ Use `--session-dir` instead (the pipeline already infers subjects path).

* ❌ `fsl: imcp not found`

  * ✅ Fix by installing missing FSL binaries or creating symbolic links.

---

## 📊 Next Steps

* Run **feature extraction scripts** on preprocessed data.
* Generate **connectivity matrices** for machine learning.

---

## 📚 References

* Human Connectome Project (HCP) Pipelines: [GitHub](https://github.com/Washington-University/HCPpipelines)
* [FSL Installation Guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index)
* [FreeSurfer Documentation](https://surfer.nmr.mgh.harvard.edu/fswiki/Documentation)

---

Would you like me to also **add a folder structure diagram** (showing `/NIfTI`, `/I1092241/T1w`, `/I1092241/T2w`, etc.) so that anyone following your README knows exactly where to keep the files?
