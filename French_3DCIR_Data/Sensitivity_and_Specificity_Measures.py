__author__ = 'Brian M Anderson'
# Created on 8/25/2020

import SimpleITK as sitk
import numpy as np
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import os
import pandas as pd
from Segmentation_Evaluation_Tools.src.SegmentationEvaluationTools import determine_sensitivity,\
    determine_false_positive_rate_and_false_volume, identify_overlap_metrics


def single_site_comparison(path):
    prediction_files = [i for i in os.listdir(path) if i.endswith('_Prediction.nii') and
                        i.split('_Prediction')[0] not in ['Patient_5', 'Patient_11',
                                                          'Patient_12', 'Patient_18', 'Patient_20']]
    out_dict = {'Patient_Name': [], 'Volume (cc)': []}
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    Connected_Threshold = sitk.ConnectedThresholdImageFilter()
    Connected_Threshold.SetLower(1)
    Connected_Threshold.SetUpper(2)
    truth_stats = sitk.LabelShapeStatisticsImageFilter()
    pred_stats = sitk.LabelShapeStatisticsImageFilter()
    for prediction_file in prediction_files:
        patient_name = prediction_file.split('_Prediction')[0]
        print(patient_name)
        '''
        First, load up the truth and prediction
        '''
        prediction_handle_base = sitk.ReadImage(os.path.join(path, prediction_file))
        truth_handle_base = sitk.ReadImage(os.path.join(path, prediction_file.replace('_Prediction', '_Truth')),
                                           sitk.sitkUInt8)
        spacing = truth_handle_base.GetSpacing()
        '''
        Next, identify each independent segmentation in both
        '''
        truth_handle = Connected_Component_Filter.Execute(truth_handle_base)
        prediction_handle = Connected_Component_Filter.Execute(prediction_handle_base)
        '''
        Get the independent lesions of both
        '''
        truth_stats.Execute(truth_handle)
        pred_stats.Execute(prediction_handle)

        prediction = sitk.GetArrayFromImage(prediction_handle_base)
        for label in truth_stats.GetLabels():
            truth_site = truth_handle == label
            truth = sitk.GetArrayFromImage(truth_site)
            overlap = truth * prediction
            if np.max(overlap) == 0:  # If there is no overlap, take the closest one
                truth_seed = prediction_handle_base.TransformPhysicalPointToIndex(truth_stats.GetCentroid(label))
                pred_seeds = [pred_stats.GetCentroid(l) for l in pred_stats.GetLabels()]
                pred_seeds = [prediction_handle_base.TransformPhysicalPointToIndex(i) for i in pred_seeds]
                seeds = [pred_seeds[np.argmin(np.sqrt(np.sum(np.subtract(pred_seeds, truth_seed)**2, axis=-1)),
                                              axis=-1)]]
            else:
                seeds = np.transpose(np.asarray(np.where(overlap > 0)))[..., ::-1]
                seeds = [[int(i) for i in j] for j in seeds]
            Connected_Threshold.SetSeedList(seeds)
            grown_prediction = Connected_Threshold.Execute(prediction_handle_base)
            volume = np.sum(truth) * np.prod(spacing) / 1000
            tumor_data = identify_overlap_metrics(prediction_handle=grown_prediction, truth_handle=truth_site,
                                                  perform_distance_measures=True)
            out_dict['Patient_Name'].append(patient_name)
            out_dict['Volume (cc)'].append(volume)
            for key in tumor_data.keys():
                if key not in out_dict:
                    out_dict[key] = []
                out_dict[key].append(tumor_data[key])
    return out_dict


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
