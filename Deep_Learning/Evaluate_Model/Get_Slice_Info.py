__author__ = 'Brian M Anderson'
# Created on 3/5/2020

import SimpleITK as sitk
import os
import pandas as pd

path_base = r'D:\Liver_Segments'

reader = sitk.ImageFileReader()
reader.LoadPrivateTagsOn()
out_dict = {'File_Name':[],'Slice_Thickness':[]}
for folder in ['Train','Test','Validation']:
    path = os.path.join(path_base,folder)
    files = [i for i in os.listdir(path) if i.find('Overall_Data') == 0]
    for file in files:
        print(file)
        reader.SetFileName(os.path.join(path_base,folder,file))
        reader.ReadImageInformation()
        out_dict['File_Name'].append(file)
        out_dict['Slice_Thickness'].append(reader.GetSpacing()[-1])
        xxx = 1
df = pd.DataFrame(out_dict)
df.to_excel(os.path.join(path_base,'Slice_Thickness.xlsx'), index=0)
xxx = 1