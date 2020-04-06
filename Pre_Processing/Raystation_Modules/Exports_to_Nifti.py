__author__ = 'Brian M Anderson'
# Created on 4/5/2020


import os
import SimpleITK as sitk
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image


base_path = r'D:\Liver_Disease_Ablation'
reader = sitk.ImageFileReader()
data_dict = {}
for folder in ['Train','Test','Validation']:
    lits_files = [i for i in os.listdir(os.path.join(base_path,folder)) if i.find('Overall_Data')==0]
    for file in lits_files:
        iteration = file.split('_')[-1].split('.nii')[0]
        data_dict[iteration] = os.path.join(base_path,folder,file)
working_path = os.path.join(base_path,'Raystation_Exports')
files = [i for i in os.listdir(working_path) if i.find('.mhd') != -1]
for file in files:
    iteration = file.split('_')[1]
    reader.SetFileName(data_dict[iteration])
    reader.ReadImageInformation()
    out_file = os.path.join(working_path,file.replace('.mhd','.nii.gz'))
    annotation = sitk.ReadImage(os.path.join(working_path,file))
    annotation = sitk.BinaryThreshold(annotation,lowerThreshold=255//2)
    annotation.SetSpacing(reader.GetSpacing())
    annotation.SetOrigin(reader.GetOrigin())
    annotation.SetDirection(reader.GetDirection())
    sitk.WriteImage(annotation,os.path.join(working_path,out_file))
    xxx = 1