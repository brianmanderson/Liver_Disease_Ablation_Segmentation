__author__ = 'Brian M Anderson'
# Created on 4/6/2020
import sys

sys.path.append('.')
'''
First, need to copy the rois from the old segmentations to a new one
'''
copy_rois = False
if copy_rois:
    xxx = 1
    # run Copy_Rois_To_Another in Raystation

'''
Next, make any changes you want, and export them as .mhd files
'''
create_mhd = False
if create_mhd:
    xxx = 1
    # Run Export_New_Rois_As_mhd in Raystation

'''
Now, convert those mhd files into niftii files
'''
convert_to_niftii = False
if convert_to_niftii:
    from mhd_to_Nifti import mhd_files_to_niftii
    mhd_files_to_niftii()

'''
Replace the other masks with the new ones
'''
replace_old = False
if replace_old:
    from mhd_to_Nifti import replace_old_masks
    replace_old_masks()

'''
Remake single_images
'''
remake_single = False
if remake_single:
    from Pre_Processing.Make_Single_Images.Make_Single_Images_Class import run_main
    path = r'H:\Liver_Disease_Ablation'
    desired_output_spacing = (None, None, None)
    extension = 32
    write_images = True
    re_write_pickle = True
    run_main(path, desired_output_spacing, extension, write_images, re_write_pickle, file_ext='_None')