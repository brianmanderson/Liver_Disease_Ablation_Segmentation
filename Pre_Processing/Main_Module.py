__author__ = 'Brian M Anderson'
# Created on 1/15/2020

from Pre_Processing.LiTs_Into_Niftii import create_NIFTI_images, os
from Pre_Processing.Make_Single_Images.Make_Single_Images_Class import run_main
from Pre_Processing.Separate_Into_Train_Validation_Test import separate_files


def run_LiTs_to_NIFTII():
    data_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Fuentes_Data\LiTs\Images'
    out_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
    images_desc = 'LiTs'
    create_NIFTI_images(data_path, out_path, images_desc)


create_niftii_images = False
separate_to_train_etc = False
make_single_images = False
make_TF2_images = True

if create_niftii_images:
    run_LiTs_to_NIFTII()

if separate_to_train_etc:
    path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
    separate_files(path)

if make_single_images:
    # path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
    path = r'D:\Liver_Disease_Ablation'
    desired_output_spacing = (None, None, None)
    extension = 32
    write_images = False
    re_write_pickle = True
    run_main(path,desired_output_spacing,extension, write_images, re_write_pickle, file_ext='_None')

if make_TF2_images:
    path = r'D:\Liver_Disease_Ablation'
    from Pre_Processing.Make_Single_Images.Make_TFRecord_Class import write_tf_record
    from Pre_Processing.Make_Single_Images.Image_Processors_Module.Image_Processors_TFRecord import *
    image_processors = [Clip_Images_By_Extension(8), Distribute_into_3D(max_z=16, mirror_small_bits=True)]
    write_tf_record(os.path.join(path, 'Train'), record_name='Train', image_processors=image_processors,
                    is_3D=True, rewrite=True, thread_count=1)