__author__ = 'Brian M Anderson'
# Created on 5/10/2021
import os
import SimpleITK as sitk
import numpy as np
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image


def _orient_handles_(moving_handle, fixed_handle):
    moving_handle.SetSpacing(fixed_handle.GetSpacing())
    moving_handle.SetOrigin(fixed_handle.GetOrigin())
    moving_handle.SetDirection(fixed_handle.GetDirection())
    return moving_handle


path_base = r'H:\Liver_Disease_Ablation'
for folder in ['Train', 'Validation', 'Test']:
    print(folder)
    path = os.path.join(path_base, folder)
    files = [i for i in os.listdir(path) if i.startswith('Overall_mask_LiTs')]
    for file in files:
        print(file)
        image_handle = sitk.ReadImage(os.path.join(path, file))
        image_array = sitk.GetArrayFromImage(image_handle)
        image_array[image_array == 1] = 3
        image_array[image_array > 0] -= 1
        image_handle = _orient_handles_(moving_handle=sitk.GetImageFromArray(image_array), fixed_handle=image_handle)
        sitk.WriteImage(image=image_handle, fileName=os.path.join(path, file))