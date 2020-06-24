import SimpleITK as sitk
import numpy as np
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt
import os


path = r'H:\Liver_Disease_Ablation\Predictionsdense\ValidationTF2_Multi_Cube_Dense'
bad_image = os.path.join(path, 'Overall_Data_LiTs_{}.nii.gz_Image.nii.gz'.format(1))
good_image = os.path.join(path, 'Overall_Data_LiTs_{}.nii.gz_Image.nii.gz'.format(123))

bad = sitk.GetArrayFromImage(sitk.ReadImage(bad_image))
good = sitk.GetArrayFromImage(sitk.ReadImage(good_image))
data_bad = bad[bad!=0]
data_good = good[good!=0]
xxx = 1