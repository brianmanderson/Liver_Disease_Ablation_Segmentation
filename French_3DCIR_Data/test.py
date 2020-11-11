__author__ = 'Brian M Anderson'
# Created on 9/30/2020
import os
from Dicom_RT_and_Images_to_Mask.src.DicomRTTool import plot_scroll_Image
from Segmentation_Evaluation_Tools.src.SegmentationEvaluationTools import *

import SimpleITK as sitk

path = r'H:\Liver_Disease_Ablation\3Dircadb1\Niftii_Files'

truth = os.path.join(path, 'Patient_9_Truth.nii')
pred = os.path.join(path, 'Patient_9_Prediction.nii')

truth_handle = sitk.ReadImage(truth)
pred_handle = sitk.ReadImage(pred)

# k = identify_overlap_metrics(prediction_handle=pred_handle, truth_handle=truth_handle,
#                              perform_distance_measures=True)
# k = determine_false_positive_rate_and_false_volume(prediction_handle=pred_handle, truth_handle=truth_handle)
k = determine_sensitivity(prediction_handle=pred_handle, truth_handle=truth_handle)
xxx = 1