__author__ = 'Brian M Anderson'
# Created on 6/3/2020
import SimpleITK as sitk
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt
import os
import pandas as pd


reader = sitk.ImageFileReader()
image_path = r'H:\Liver_Disease_Ablation'
description = {'Folder':[],'File':[],'Spacing_x':[],'Spacing_y':[],'Spacing_z':[]}
for folder in ['Train','Validation','Test']:
    path = os.path.join(image_path, folder)
    files = [i for i in os.listdir(path) if i.startswith('Overall_Data')]
    for file in files:
        reader.SetFileName(os.path.join(path, file))
        reader.ReadImageInformation()
        spacing = reader.GetSpacing()
        print(spacing)
        description['Folder'].append(folder)
        description['File'].append(file)
        description['Spacing_x'].append(spacing[0])
        description['Spacing_y'].append(spacing[1])
        description['Spacing_z'].append(spacing[2])
df = pd.DataFrame(description)
df.to_excel(os.path.join('.','File_Slicing.xlsx'), index=0)