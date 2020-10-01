__author__ = 'Brian M Anderson'
# Created on 8/25/2020

import SimpleITK as sitk
import numpy as np
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import os
import pandas as pd
from Segmentation_Evaluation_Tools.src.SegmentationEvaluationTools import determine_sensitivity,\
    determine_false_positive_rate_and_false_volume


def compile_sensitivity(path):
    prediction_files = [i for i in os.listdir(path) if i.endswith('_Prediction.nii') and
                        i.split('_Prediction')[0] not in ['Patient_5', 'Patient_11',
                                                          'Patient_12', 'Patient_18', 'Patient_20']]
    out_dict = {'Patient_Name': []}
    for prediction_file in prediction_files:
        patient_name = prediction_file.split('_Prediction')[0]
        print(patient_name)
        truth_file = prediction_file.replace('_Prediction', '_Truth')
        prediction_handle = sitk.ReadImage(os.path.join(path, prediction_file))
        truth_handle = sitk.ReadImage(os.path.join(path, truth_file), sitk.sitkUInt8)
        patient_sensitivity = determine_sensitivity(prediction_handle=prediction_handle, truth_handle=truth_handle)
        for key in patient_sensitivity.keys():
            if key not in out_dict.keys():
                out_dict[key] = []
            out_dict[key] += patient_sensitivity[key]
        out_dict['Patient_Name'] += [patient_name for _ in range(len(patient_sensitivity[key]))]
    return out_dict


def determine_false_positive_rate(path=r'H:\Liver_Disease_Ablation\Predictions_93\TestTF2_Multi_Cube_1mm'):
    prediction_files = [i for i in os.listdir(path) if i.endswith('_Prediction.nii') and
                        i.split('_Prediction')[0] not in ['Patient_5', 'Patient_11',
                                                          'Patient_12', 'Patient_18', 'Patient_20']]
    out_dict = {'Patient_Name': []}
    for prediction_file in prediction_files:
        patient_name = prediction_file.split('_Prediction')[0]
        print(patient_name)
        '''
        First, load up the truth and prediction
        '''
        truth_file = prediction_file.replace('_Prediction', '_Truth')
        prediction_handle = sitk.ReadImage(os.path.join(path, prediction_file))
        truth_handle = sitk.ReadImage(os.path.join(path, truth_file), sitk.sitkUInt8)
        patient_data = determine_false_positive_rate_and_false_volume(prediction_handle=prediction_handle,
                                                                      truth_handle=truth_handle)
        for key in patient_data:
            if key not in out_dict.keys():
                out_dict[key] = []
            out_dict[key].append(patient_data[key])
        out_dict['Patient_Name'].append(patient_name)
    return out_dict


def write_sensitivity_specificity(excel_path=os.path.join('.', 'Sensitivity_and_FP2.xlsx'), nifti_path=r'.'):
    out_dict_false_postive = determine_false_positive_rate(path=nifti_path)
    out_dict_sensitivity = compile_sensitivity(path=nifti_path)
    with pd.ExcelWriter(excel_path) as writer:
        df = pd.DataFrame(out_dict_false_postive)
        df.to_excel(writer, sheet_name='False Positive Rate', index=0)
        df = pd.DataFrame(out_dict_sensitivity)
        df.to_excel(writer, sheet_name='Sensitivity', index=0)


if __name__ == '__main__':
    pass
