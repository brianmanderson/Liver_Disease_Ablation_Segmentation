__author__ = 'Brian M Anderson'
# Created on 2/14/2020
# With a large amount of help from Bastien githut.com/guatavita

from Pre_Processing.Nifti_To_Dicom.Conversion_Definition import convert_niftii_to_dicom
import os
from Pre_Processing.Nifti_To_Dicom.Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, sitk

base_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
out_path_base = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Nifti_to_dicom\LiTs'

Dicom_reader = Dicom_to_Imagestack()
for folder in ['Train','Test','Validation']:
    file_path = os.path.join(base_path,folder)
    files = os.listdir(file_path)
    files = [i for i in files if i.find('Overall_mask') == 0]
    for file in files:
        data_file = file.replace('Overall_mask','Overall_Data').replace('_y','_')
        desc = data_file.split('Data_')[1].split('.nii')[0]
        out_path = os.path.join(out_path_base, desc)
        if not os.path.exists(out_path):
            print(desc)
            convert_niftii_to_dicom(os.path.join(file_path,data_file),out_path,patient_name=desc,
                                    patient_id=desc,is_structure=False)
            fid = open(os.path.join(out_path,'Completed.txt'),'w+')
            fid.close()