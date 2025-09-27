Got it ✅ Here’s your **full README.md** with the flow diagram embedded at the right spot, written cleanly in Markdown so you can copy-paste directly into your repo:

```markdown
# 🧠 Integrated Brain Connectivity Analysis  

This repository contains code and documentation for the thesis project **Integrated Brain Connectivity Analysis**, based on:  

> Qu, G., Zhou, Z., Calhoun, V. D., Zhang, A., & Wang, Y.-P. (2025). *Integrated brain connectivity analysis with fMRI, DTI, and sMRI powered by interpretable graph neural networks.* *Medical Image Analysis*, 103570.  

The goal is to build an integrated, multi-modal model of human brain connectivity using fMRI, DTI, and sMRI, leveraging **interpretable Graph Neural Networks (GNNs)** for biomarker discovery.  

---

## 📌 Current Status — Data Preprocessing  

We have completed the **data preprocessing phase** as per the research paper:  

### 🔄 Step 1: DICOM → NIfTI conversion  
Tools:  
- [`dcm2niix`](https://github.com/rordenlab/dcm2niix)  
- `dicom2nifti` (Python-based)  
- `mrconvert` (MRtrix)  

Example:  
- Subject **I1092241** was converted from `.dcm` to `.nii` using **MRIcroGL**.  
- Output:  
```

I1092241_Axial_rsfMRI_(EYES_OPEN)_20181213125500_9.nii

````

---

### 🧹 Step 2: Preprocessing with HCP Minimal Pipelines  
As per the reference paper, **HCP pipelines** were applied to clean and standardize the fMRI, T1w, and T2w data.  

#### 🛠 Dependencies Installed
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index)  
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/registration.html) (+ `license.txt`)  
- [HCP Pipelines](https://github.com/Washington-University/HCPpipelines)  
- [Connectome Workbench](https://www.humanconnectome.org/software/connectome-workbench)  

#### ⚡ Environment Setup
```bash
source /mnt/d/brain/HCPpipelines-master/Examples/Scripts/SetUpHCPPipeline.sh
````

#### 📍 PreFreeSurfer Pipeline

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

#### 📍 FreeSurfer Pipeline

> ⚠️ Issue fixed: Removed the invalid `--subjects-dir` option. Use `--session-dir` instead.

```bash
/mnt/d/brain/HCPpipelines-master/FreeSurfer/FreeSurferPipeline.sh \
--session=I1092241 \
--session-dir=/mnt/d/brain/I1092241 \
--t1w-image=/mnt/d/brain/I1092241/T1w/T1w_acpc_dc_restore.nii.gz \
--t1w-brain=/mnt/d/brain/I1092241/T1w/T1w_acpc_dc_restore_brain.nii.gz \
--t2w-image=/mnt/d/brain/I1092241/T2w/T2w_acpc_dc_restore.nii.gz
```

#### 📍 fMRI Volume Preprocessing Pipeline

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

## 🧩 Preprocessing Workflow — Visual Overview

```mermaid
flowchart TD
    A[DICOM (.dcm) files] --> B[Convert with dcm2niix / MRIcroGL]
    B --> C[NIfTI (.nii) format]

    C --> D[HCP PreFreeSurfer Pipeline<br/>(T1w + T2w structural preprocessing)]
    D --> E[HCP FreeSurfer Pipeline<br/>(surface reconstruction)]
    E --> F[HCP fMRIVolume Pipeline<br/>(fMRI preprocessing)]

    F --> G[Processed Data<br/>(clean, aligned, ready for analysis)]
    G --> H[Feature Extraction<br/>(Connectivity matrices: fMRI, DTI, sMRI)]
    H --> I[Graph Construction<br/>(multi-modal brain graph)]
    I --> J[Graph Neural Network (GNN)<br/>Biomarker Discovery & Interpretation]
```

---

## 📊 Next Steps (Planned)

* **Feature Extraction**: Build connectivity matrices from fMRI (functional), DTI (structural), and sMRI (morphological) data.
* **Graph Construction**: Create multimodal graphs (nodes = ROIs, edges = connections).
* **GNN Implementation**: Train interpretable Graph Neural Network for biomarker discovery.
* **Analysis & Visualization**: Identify key brain connectivity patterns and biomarkers.

---

## 🛠 Dependencies Summary

* `dcm2niix`
* **HCP Pipelines** (requires FSL, FreeSurfer, Connectome Workbench)
* Python packages:

  * `nibabel`, `nilearn`, `dipy`
  * `numpy`, `pandas`, `scipy`
  * `matplotlib`, `seaborn`
  * `torch`, `torch-geometric` (for future GNN work)

---

## ⚖️ License

This repository is released under the **MIT License**.

---

## 🙋 Author

* **Harshabardhana Parida**

---

## 📖 Citation

If you use this repository or its pipeline in your work, please cite the original paper that inspired this thesis:

> Qu, G., Zhou, Z., Calhoun, V. D., Zhang, A., & Wang, Y.-P. (2025). *Integrated brain connectivity analysis with fMRI, DTI, and sMRI powered by interpretable graph neural networks.* *Medical Image Analysis*, 103570.

```
