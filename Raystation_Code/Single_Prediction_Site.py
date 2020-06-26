__author__ = 'Brian M Anderson'
# Created on 6/26/2020

import SimpleITK as sitk
import os, time
import numpy as np
from .Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack


path = r'H:\Single_Site'
stats = sitk.LabelShapeStatisticsImageFilter()
Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
Connected_Threshold = sitk.ConnectedThresholdImageFilter()
Connected_Threshold.SetUpper(2)
while True:
    if not os.path.exists(os.path.join(path,'Completed.txt')):
        print('Waiting...')
        time.sleep(5)
    reader = Dicom_to_Imagestack()
    reader.Make_Contour_From_directory(path)
    mhd_files = [os.path.join(path,i) for i in os.listdir(path) if i.endswith('.mhd')]
    pred_file = [i for i in mhd_files if i.find('BMA') != -1]
    prediction = sitk.ReadImage(pred_file[0]) > 255 / 2
    gt = [i for i in mhd_files if i.find('GTV') != -1 or i.find('Ablation') != -1]
    gt = sitk.ReadImage(gt) > 255 / 2