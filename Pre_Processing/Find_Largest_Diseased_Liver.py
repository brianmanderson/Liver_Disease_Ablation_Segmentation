__author__ = 'Brian M Anderson'
# Created on 5/17/2020

import os
import SimpleITK as sitk
import numpy as np
from Make_Single_Images.Image_Processors_Module.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt

path = r'H:\Liver_Disease_Ablation\Train'
files = [i for i in os.listdir(path) if i.find('mask') != -1]

out_dict = {'image_names':[],'percent_disease':[]}
for file in files:
    handle = sitk.ReadImage(os.path.join(path,file))
    liver = np.sum(sitk.GetArrayFromImage(handle>0))
    disease = np.sum(sitk.GetArrayFromImage(handle>1))
    print('{} {}% Disease'.format(file, disease/liver*100))
    out_dict['image_names'].append(file)
    out_dict['percent_disease'].append(disease/liver*100)
xxx = 1