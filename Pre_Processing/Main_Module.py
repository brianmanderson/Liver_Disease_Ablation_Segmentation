__author__ = 'Brian M Anderson'
# Created on 1/15/2020

import os


def run_LiTs_to_NIFTII():
    from Pre_Processing.LiTs_Into_Niftii import create_NIFTI_images
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
    from Pre_Processing.Separate_Into_Train_Validation_Test import separate_files
    path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
    separate_files(path)

if make_single_images:
    from Pre_Processing.Make_Single_Images.Make_Single_Images_Class import run_main
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
    cube_size = (32, 120, 120)
    base_normalizer = [Normalize_to_annotation(annotation_value_list=[1,2], mirror_max=True,
                                               lower_percentile=25, upper_percentile=75)]
    image_processors_train = []
    image_processors_train += base_normalizer
    image_processors_train += [Split_Disease_Into_Cubes(cube_size=cube_size, disease_annotation=2,
                                                        min_voxel_volume=300, max_voxels=1350000),
                               Distribute_into_3D(max_z=cube_size[0], max_rows=cube_size[1], max_cols=cube_size[2],
                                                  min_z=cube_size[0])]

    write_tf_record(os.path.join(path, 'Train'), record_name='Train_32', image_processors=image_processors_train,
                    is_3D=True, rewrite=True, shuffle=True, thread_count=10)

    image_processors_validation = []
    image_processors_validation += base_normalizer
    image_processors_validation += [Box_Images(wanted_vals_for_bbox=[1,2],power_val_z=2**3, power_val_r=2**3,
                                               power_val_c=2**3),
                                    Distribute_into_3D(max_z=64, mirror_small_bits=True, chop_ends=False,
                                                       desired_val=2)]
    write_tf_record(os.path.join(path, 'Validation'), record_name='Validation', thread_count=10,
                    image_processors=image_processors_validation, is_3D=True, rewrite=True, shuffle=True)

    image_processors_validation = []
    image_processors_validation += base_normalizer
    image_processors_validation += [Box_Images(wanted_vals_for_bbox=[1,2],power_val_z=2**3, power_val_r=2**3,
                                               power_val_c=2**3),
                                    Distribute_into_3D(mirror_small_bits=True, chop_ends=False, desired_val=2)]
    write_tf_record(os.path.join(path, 'Validation'), record_name='Validation_whole', thread_count=10,
                    image_processors=image_processors_validation,
                    is_3D=True, rewrite=True, shuffle=True)

    image_processors_test = []
    image_processors_test += base_normalizer
    image_processors_test += [Box_Images(wanted_vals_for_bbox=[1,2],power_val_z=2**3, power_val_r=2**3,
                                         power_val_c=2**3),
                              Distribute_into_3D(mirror_small_bits=True, chop_ends=False, desired_val=2)]
    write_tf_record(os.path.join(path, 'Test'), record_name='Test', image_processors=image_processors_test,
                    is_3D=True, rewrite=True, shuffle=True, thread_count=10)