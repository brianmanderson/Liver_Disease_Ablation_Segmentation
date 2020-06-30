__author__ = 'Brian M Anderson'
# Created on 6/26/2020

import SimpleITK as sitk
import os, time
import numpy as np
from Raystation_Code.Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack
from Raystation_Code.Dicom_RT_and_Images_to_Mask.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image


path = r'H:\Single_Site'
stats = sitk.LabelShapeStatisticsImageFilter()
Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
Connected_Threshold = sitk.ConnectedThresholdImageFilter()
Connected_Threshold.SetUpper(2)
while True:
    while not os.path.exists(os.path.join(path,'Completed_Export.txt')):
        print('Waiting...')
        time.sleep(1)
    reader = Dicom_to_Imagestack()
    reader.Make_Contour_From_directory(path)
    mhd_files = [os.path.join(path,i) for i in os.listdir(path) if i.endswith('.mhd')]
    pred_file = [i for i in mhd_files if i.find('BMA') != -1]
    pred_primary = sitk.ReadImage(pred_file[0]) > 255 / 2
    gt = [i for i in mhd_files if i.find('BMA') == -1]
    gt = sitk.ReadImage(gt[0]) > 255 / 2
    connected_image = Connected_Component_Filter.Execute(gt * pred_primary)
    stats.Execute(connected_image)
    seeds_pred = [stats.GetCentroid(l) for l in stats.GetLabels()]
    seeds_pred = [pred_primary.TransformPhysicalPointToIndex(i) for i in seeds_pred]

    Connected_Threshold.SetSeedList(seeds_pred)
    Connected_Threshold.SetLower(1)
    pred_grown = Connected_Threshold.Execute(pred_primary)
    prediction = sitk.GetArrayFromImage(pred_grown)
    for i in os.listdir(path):
        os.remove(os.path.join(path,i))
    reader.with_annotations(np.repeat(prediction[...,None],axis=-1, repeats=2).astype('int'), output_dir=path, ROI_Names=['Solidary_Prediction_BMA'])
    time.sleep(1)