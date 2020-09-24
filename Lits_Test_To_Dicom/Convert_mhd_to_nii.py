__author__ = 'Brian M Anderson'
# Created on 9/24/2020


import os
import SimpleITK as sitk
from Pre_Processing.Dicom_RT_and_Images_to_Mask.src.DicomRTTool import plot_scroll_Image, DicomReaderWriter, np


def mhd_to_nii():
    base_path = r'H:\Liver_Disease_Ablation\LiTs_Test\Nifti'
    pred_base_path = os.path.join(base_path, 'predictions')

    pred_files = [i for i in os.listdir(pred_base_path) if i.startswith('Pred_') and i.endswith('.mhd')]
    for file in pred_files:
        index = file.split('_')[-1].split('.')[0]
        pred_out_path = os.path.join(base_path, 'test-segmentation-{}.nii'.format(index))
        print(file)
        if os.path.exists(pred_out_path):
            continue
        image_handle = sitk.ReadImage(os.path.join(base_path, 'test-volume-{}.nii'.format(index)))
        prediction_handle = sitk.ReadImage(os.path.join(pred_base_path, file))
        prediction = sitk.GetArrayFromImage(prediction_handle > 255/2)
        liver_handle = sitk.ReadImage(os.path.join(pred_base_path, file.replace('Pred_', 'Liver_')))
        liver = sitk.GetArrayFromImage(liver_handle > 255/2)
        if np.max(liver) > 0:
            prediction[liver == 0] = 0
            prediction[prediction > 0] = 2
            out_handle = sitk.GetImageFromArray(prediction)
            out_handle.SetOrigin(image_handle.GetOrigin())
            out_handle.SetDirection(image_handle.GetDirection())
            out_handle.SetSpacing(image_handle.GetSpacing())
            sitk.WriteImage(out_handle, pred_out_path)
    return None


if __name__ == '__main__':
    pass
