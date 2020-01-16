__author__ = 'Brian M Anderson'
# Created on 1/15/2020

from Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, sitk
import os
import numpy as np
import scipy.misc as scm


def write_image(base_path,image_file,out_path, no_rotations, no_flip_lr, images_description='LiTs'):
    print(image_file)
    index = int((image_file.split('volume-')[1]).split('.nii')[0])
    if os.path.exists(os.path.join(out_path, 'Overall_mask_LiTs_y' + str(index))):
        return None
    label_handle = sitk.ReadImage(os.path.join(base_path, 'segmentation-{}.nii'.format(index)))
    label = sitk.GetArrayFromImage(label_handle)
    if np.max(label) == 1:
        return None
    image_handle = sitk.ReadImage(os.path.join(base_path, image_file))
    img = sitk.GetArrayFromImage(image_handle)
    num_turns = 2
    if index in no_rotations:
        num_turns = 0
    for i in range(num_turns):
        img = np.rot90(img,axes=(1,2))
    for i in range(num_turns):
        label = np.rot90(label, axes=(1,2))
    if index not in no_flip_lr:
        img = img[:, :, ::-1]
        label = label[:,:,::-1]
    write_png_images = False
    if write_png_images:
        path = r'C:\Users\bmanderson\Desktop\images_LiTs'
        locations = np.where(np.max(label,axis=(1,2)) != 0)[0]
        i = locations[len(locations)//2]
        out = np.zeros([512, 512 * 2])
        out[:, 0:512] = img[i]
        max_val = img[i].max()
        out[:, 512:] = label[i] * max_val
        scm.imsave(os.path.join(path, 'combined_' + str(index) + '.png'), out)
    out_image_handle = sitk.GetImageFromArray(img)
    out_image_handle.SetSpacing(image_handle.GetSpacing())
    out_image_handle.SetOrigin(image_handle.GetOrigin())
    out_image_handle.SetDirection(image_handle.GetDirection())

    out_annotation_handle = sitk.GetImageFromArray(label)
    out_annotation_handle.SetSpacing(label_handle.GetSpacing())
    out_annotation_handle.SetOrigin(label_handle.GetOrigin())
    out_annotation_handle.SetDirection(label_handle.GetDirection())

    image_path = os.path.join(out_path, 'Overall_Data_' + images_description + '_' + str(index) + '.nii.gz')
    annotation_path = os.path.join(out_path, 'Overall_mask_' + images_description + '_y' + str(index) + '.nii.gz')
    sitk.WriteImage(out_annotation_handle, annotation_path)
    sitk.WriteImage(out_image_handle, image_path)
    return None



'''
default to two rotations and a flip LR
'''

def create_NIFTI_images(data_path,out_path, images_description='LiTs'):
    no_rotations = list(np.arange(53,68))+list(np.arange(83,131))
    no_flip_lr = list(np.arange(53,131))
    files = [i for i in os.listdir(data_path) if i.find('volume') == 0]
    files.sort(key=lambda x: int(x.split('volume-')[1][:-4]))
    for file in files:
        index = file.split('volume-')[1][:-4]
        # if os.path.exists(r'C:\Users\bmanderson\Desktop\images_LiTs\combined_{}.png'.format(index)):
        #     continue
        image_path = os.path.join(out_path, 'Overall_Data_' + images_description + '_' + str(index) + '.nii.gz')
        if os.path.exists(image_path):
            continue
        write_image(data_path,file,out_path, no_rotations,no_flip_lr, images_description=images_description)


def main():
    data_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Fuentes_Data\LiTs\Images'
    out_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
    images_desc = 'LiTs'
    create_NIFTI_images(data_path, out_path, images_desc)