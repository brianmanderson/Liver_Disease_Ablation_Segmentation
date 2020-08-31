__author__ = 'Brian M Anderson'
# Created on 8/25/2020

import SimpleITK as sitk
import numpy as np
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import os
import pandas as pd


def determine_sensitivity(path=r'H:\Liver_Disease_Ablation\Predictions_93\TestTF2_Multi_Cube_1mm'):
    prediction_files = [i for i in os.listdir(path) if i.endswith('_FinalPrediction.nii.gz')]
    out_dict = {'Patient_Name': [], 'Site_Number': [], '% Covered': [], 'Volume (cc)': []}
    i = 0
    for prediction_file in prediction_files:
        patient_name = prediction_file.split('_FinalPrediction')[0]
        print(patient_name)
        truth_file = prediction_file.replace('_FinalPrediction', '_Truth')
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, prediction_file))).astype('int')
        truth = sitk.ReadImage(os.path.join(path, truth_file), sitk.sitkUInt8)
        stats = sitk.LabelShapeStatisticsImageFilter()
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        labeled_truth = Connected_Component_Filter.Execute(truth)
        stats.Execute(labeled_truth)
        tumor_labels = stats.GetLabels()
        spacing = np.prod(truth.GetSpacing())
        for tumor_label in tumor_labels:
            single_site = sitk.GetArrayFromImage(labeled_truth == tumor_label).astype('int')
            total = np.sum(single_site)
            difference = single_site - prediction > 0
            remainder = np.sum(difference)
            covered = (total - remainder) / total * 100
            out_dict['Patient_Name'].append(patient_name)
            out_dict['Site_Number'].append(i)
            out_dict['% Covered'].append(covered)
            out_dict['Volume (cc)'].append(total * spacing / 1000)  # Record in cc
            i += 1
    return out_dict


def determine_false_positive_rate(path=r'H:\Liver_Disease_Ablation\Predictions_93\TestTF2_Multi_Cube_1mm'):
    prediction_files = [i for i in os.listdir(path) if i.endswith('_FinalPrediction.nii.gz')]
    out_dict = {'Patient_Name': [], 'Number False Positives': [], 'Volume (cc)': [], 'False Seeded Volume (cc)': [],
                'False Volume (cc)': []}
    for prediction_file in prediction_files:
        patient_name = prediction_file.split('_FinalPrediction')[0]
        print(patient_name)
        '''
        First, load up the truth and prediction
        '''
        truth_file = prediction_file.replace('_FinalPrediction', '_Truth')
        prediction_handle = sitk.ReadImage(os.path.join(path, prediction_file), sitk.sitkUInt8)
        prediction = sitk.GetArrayFromImage(prediction_handle).astype('int')
        truth_handle = sitk.ReadImage(os.path.join(path, truth_file), sitk.sitkUInt8)
        truth = sitk.GetArrayFromImage(truth_handle)
        spacing = np.prod(truth_handle.GetSpacing())
        total_difference = np.sum(prediction - truth > 0) * spacing / 1000
        out_dict['False Volume (cc)'].append(total_difference)
        '''
        Next, get stats and connected component, these will make each site an individual label
        '''
        stats = sitk.LabelShapeStatisticsImageFilter()
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        labeled_truth = Connected_Component_Filter.Execute(truth_handle)
        stats.Execute(labeled_truth)
        seeds = [stats.GetCentroid(l) for l in stats.GetLabels()]
        seeds = [prediction_handle.TransformPhysicalPointToIndex(i) for i in seeds]
        '''
        The seeds represent the starting points of the ground truth, we'll use these to determine truth from false
        '''
        Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        Connected_Threshold.SetUpper(2)
        Connected_Threshold.SetSeedList(seeds)
        Connected_Threshold.SetLower(1)
        seed_grown_pred = sitk.GetArrayFromImage(Connected_Threshold.Execute(prediction_handle))
        '''
        This is the prediction grown by seeds in the ground truth
        '''
        difference = prediction - seed_grown_pred  #Now we have our prediction, subtracting those that include truth
        labeled_difference = Connected_Component_Filter.Execute(sitk.GetImageFromArray(difference.astype('int')))
        stats.Execute(labeled_difference)
        difference_labels = stats.GetLabels()
        out_dict['Patient_Name'].append(patient_name)
        out_dict['Number False Positives'].append(len(difference_labels))
        difference -= truth  # Subtract truth in case seed didn't land
        out_dict['False Seeded Volume (cc)'].append(np.sum(difference > 0) * spacing / 1000)
        volume = np.sum(sitk.GetArrayFromImage(truth_handle)) * spacing / 1000
        out_dict['Volume (cc)'].append(volume)
    return out_dict


def write_sensitivity_specificity(excel_path=os.path.join('.', 'Sensitivity_and_FP.xlsx')):
    out_dict_false_postive = determine_false_positive_rate()
    out_dict_sensitivity = determine_sensitivity()
    with pd.ExcelWriter(excel_path) as writer:
        df = pd.DataFrame(out_dict_false_postive)
        df.to_excel(writer, sheet_name='False Positive Rate', index=0)
        df = pd.DataFrame(out_dict_sensitivity)
        df.to_excel(writer, sheet_name='Sensitivity', index=0)


if __name__ == '__main__':
    # write_sensitivity_specificity()
    pass
