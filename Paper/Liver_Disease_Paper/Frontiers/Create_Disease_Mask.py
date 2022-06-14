import SimpleITK as sitk
import os
import numpy as np

path = r'K:\LiTs\Training_Batch2\media\nas\01_Datasets\CT\LITS\Training Batch 2'
segmentation_handle = sitk.ReadImage(os.path.join(path, 'segmentation-108.nii'))
segmentation_array = sitk.GetArrayFromImage(segmentation_handle).astype('float32')
segmentation_array -= 1
segmentation_array[segmentation_array < 0] = 0
out_handle = sitk.GetImageFromArray(segmentation_array.astype('int8'))
out_handle.SetOrigin(segmentation_handle.GetOrigin())
out_handle.SetSpacing(segmentation_handle.GetSpacing())
out_handle.SetDirection(segmentation_handle.GetDirection())
sitk.WriteImage(out_handle, os.path.join(path, 'segmentation-108_new.nii'))