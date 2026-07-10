# Liver_Disease_Ablation_Segmentation

Research code for training and evaluating deep-learning models that segment liver
disease (tumor/GTV) and post-treatment ablation zones on CT, and for deploying those
segmentations back into the clinical treatment-planning system. This is the analysis
and model-training codebase behind the associated liver disease/ablation segmentation
publications (manuscripts and figures are collected under `Paper/`).

## What it does

The pipeline runs end to end from raw imaging to clinical auto-contours:

- **`Pre_Processing/`** — converts DICOM/RTStructure and the public LiTS dataset into
  NIfTI plus TFRecord-style training records; splits into train/validation/test.
- **`Fixing_LiTs_Data/`** — corrects voxel spacing/orientation issues in the LiTS data.
- **`Deep_Learning/`** — trains and optimizes a 3D fully-atrous segmentation network
  (TensorFlow 2 / Keras; `Main_TF2.py` is the entry point). Includes learning-rate
  finding, hyperparameter optimization, and an H-DenseUNet comparison baseline.
- **`Deep_Learning/Evaluate_Model/`** — scores predictions with Dice and surface-distance
  metrics stratified by lesion volume, and produces box plots (see `Images/`).
- **`Lits_Test_To_Dicom/`**, **`Raystation_Code/`** — write predictions back out as DICOM
  RTStructures and generate liver/disease/ablation contours inside RayStation via its
  `connect` scripting API.
- **`Local_Recurrence_Work/`** — analyzes minimum distance-to-agreement (DTA) between
  ablation zones and recurrence for the recurrence-assessment paper.

## Tech stack

Python, TensorFlow 2 / Keras, SimpleITK, NumPy, DICOM/NIfTI tooling, and RayStation
scripting (`connect`). Several of Brian Anderson's utility repos are pulled in as git
submodules (e.g. `Base_Deeplearning_Code`, `Dicom_RT_and_Images_to_Mask`,
`Plot_And_Scroll_Images`, `Make_Single_Images`); clone with `--recurse-submodules`.

## Status

Research project tied to specific institutional datasets and file-share paths; not a
packaged/installable tool. Provided as a reference implementation for the published work
rather than a turnkey library.
