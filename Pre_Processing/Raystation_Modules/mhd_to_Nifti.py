__author__ = 'Brian M Anderson'
# Created on 4/5/2020


import os, shutil
import SimpleITK as sitk
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image


def get_patient_data_dict():
    base_path = r'D:\Liver_Disease_Ablation'
    data_dict = {}
    for folder in ['Train','Test','Validation']:
        lits_files = [i for i in os.listdir(os.path.join(base_path,folder)) if i.find('Overall_mask')==0]
        for file in lits_files:
            iteration = file.split('_y')[-1].split('.nii')[0]
            data_dict[iteration] = os.path.join(base_path,folder,file)
    return data_dict


def mhd_files_to_niftii(working_path=r'D:\Liver_Disease_Ablation\Raystation_Exports'):
    data_dict = get_patient_data_dict()
    files = [i for i in os.listdir(working_path) if i.find('.mhd') != -1]
    for file in files:
        print(file)
        iteration = file.split('_')[1]
        if iteration not in data_dict:
            continue
        old_file_name = data_dict[iteration]
        out_file = os.path.join(working_path, old_file_name.split('\\')[-1])
        if os.path.exists(out_file):
            continue
        old_annotation_handle = sitk.ReadImage(old_file_name)
        old_annotation = sitk.GetArrayFromImage(old_annotation_handle)
        annotation = sitk.ReadImage(os.path.join(working_path,file))
        annotation = sitk.BinaryThreshold(annotation,lowerThreshold=255//2)
        annotation = sitk.GetArrayFromImage(annotation)
        old_annotation[old_annotation>0] = 1
        annotation[old_annotation==0] = 0 # mask within liver
        old_annotation[annotation>0] = 2
        new_annotation_handle = sitk.GetImageFromArray(old_annotation)
        new_annotation_handle.SetSpacing(old_annotation_handle.GetSpacing())
        new_annotation_handle.SetOrigin(old_annotation_handle.GetOrigin())
        new_annotation_handle.SetDirection(old_annotation_handle.GetDirection())
        sitk.WriteImage(new_annotation_handle,os.path.join(working_path,old_file_name))
    return None


def replace_old_masks(working_path=r'D:\Liver_Disease_Ablation\Raystation_Exports', new_output=r'D:\Liver_Disease_Ablation\Old_Masks'):
    data_dict = get_patient_data_dict()
    files = [i for i in os.listdir(working_path) if i.find('.nii.gz') != -1]
    for file in files:
        print(file)
        iteration = file.split('_y')[-1].split('.nii')[0]
        if iteration not in data_dict:
            continue
        old_file = data_dict[iteration]
        if not os.path.exists(os.path.join(new_output,old_file.split('\\')[-1])):
            os.rename(old_file,os.path.join(new_output,old_file.split('\\')[-1]))
            os.rename(os.path.join(working_path, file), old_file)


if __name__ == '__main__':
    pass
