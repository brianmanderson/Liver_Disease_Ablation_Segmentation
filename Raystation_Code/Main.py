__author__ = 'Brian M Anderson'
# Created on 6/9/2020

import os
import pandas as pd
'''
First, Fix_Excel to make MRNs, Pre, Post have CT #, like RS
'''
fix_excel = False
if fix_excel:
    from Raystation_Code.Fix_Excel import fix_excel
    fix_excel()

write_csv = True
if write_csv:
    df = pd.read_excel(os.path.join('.','MRNs_All_Primary_Secondary_exam.xlsx'))
    df.to_csv(os.path.join('.','MRNs_All_Primary_Secondary_exam.csv'), index=0)

'''
Now, run Predict_Disease_Ablation_From_CSV in Raystation and export to path
'''

export_path = r'H:\Liver_Disease_Ablation\Disease_Ablation_From_Raystation_Test'

calculate_dsc = True
if calculate_dsc:
    import SimpleITK as sitk
    import numpy as np
    from enum import Enum
    from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt


    class OverlapMeasures(Enum):
        jaccard, dice, volume_similarity, false_negative, false_positive = range(5)


    class SurfaceDistanceMeasures(Enum):
        hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(
            5)

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    MRNs = os.listdir(export_path)
    out_dict = {'MRN':[], 'Primary':[],'Secondary':[]}
    pred_roi_base = 'Liver_Disease_Ablation_BMA_Program_0'
    stats = sitk.LabelShapeStatisticsImageFilter()
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    Connected_Threshold = sitk.ConnectedThresholdImageFilter()
    Connected_Threshold.SetUpper(2)
    for MRN in MRNs:
        print(MRN)
        out_dict['MRN'].append(MRN)
        patient_path = os.path.join(export_path, MRN)
        for gt_name, exam_name in zip(['GTV_Primary.mhd','Ablation_Secondary.mhd'], ['Primary','Secondary']):
            pred_roi = pred_roi_base + '_' + exam_name
            if os.path.exists(os.path.join(patient_path,pred_roi_base)):
                if os.path.exists(os.path.join(patient_path,pred_roi)):
                    pred_primary = sitk.ReadImage(os.path.join(patient_path,pred_roi)) > 255/2
                    gt = sitk.ReadImage(os.path.join(patient_path,gt_name)) > 255/2
                    if os.path.exists(os.path.join(patient_path, 'Liver_Primary.mhd')):
                        liver = sitk.ReadImage(os.path.join(patient_path,'Liver_Primary.mhd')) > 255/2
                    elif os.path.exists(os.path.join(patient_path, 'Liver_BMA_Program_4_Primary.mhd')):
                        liver = sitk.ReadImage(os.path.join(patient_path, 'Liver_BMA_Program_4_Primary.mhd')) > 255 / 2
                    pred_primary = pred_primary * liver
                    gt = gt * liver
                    connected_image = Connected_Component_Filter.Execute(gt)
                    stats.Execute(connected_image)
                    seeds_gt = [stats.GetCentroid(l) for l in stats.GetLabels()]
                    seeds_gt = [pred_primary.TransformPhysicalPointToIndex(i) for i in seeds_gt]

                    Connected_Threshold.SetSeedList(seeds_gt)
                    Connected_Threshold.SetLower(1)
                    pred_grown = Connected_Threshold.Execute(pred_primary)
                    overlap_measures_filter.Execute(pred_grown, gt)
                    out_dict[exam_name].append(overlap_measures_filter.GetDiceCoefficient())
    xxx = 1