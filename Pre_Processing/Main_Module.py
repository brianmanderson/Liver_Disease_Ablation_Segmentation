__author__ = 'Brian M Anderson'
# Created on 1/15/2020

import os


def return_patient_dictionary_list(path):
    patient_dict_list = []
    files = [i for i in os.listdir(path) if i.startswith('Overall_mask')]
    for file in files:
        index = file.split('_y')[-1].split('.nii')[0]
        desc = file.split('mask_')[-1].split('_y')[0]
        image = 'Overall_Data_{}_{}.nii.gz'.format(desc, index)
        out_record = '{}_{}.tfrecord'.format(desc, index)
        patient_dict_list.append({'image_path': os.path.join(path, image),
                                  'annotation_path': os.path.join(path, file),
                                  'out_name': out_record})
    return patient_dict_list


def run_LiTs_to_NIFTII():
    from Pre_Processing.LiTs_Into_Niftii import create_NIFTI_images
    data_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Fuentes_Data\LiTs\Images'
    out_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
    images_desc = 'LiTs'
    create_NIFTI_images(data_path, out_path, images_desc)


create_niftii_images = False
nifti_from_dicom = False
separate_to_train_etc = False
make_TF2_images = True

if create_niftii_images:
    run_LiTs_to_NIFTII()

if nifti_from_dicom:
    from Deep_Learning.Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter, plot_scroll_Image
    associations = {'Liver_BMA_Program_4': 'Liver', 'Liver_BMA_Program_4og': 'Liver',
                    'Retro_Ablation': 'Ablation', 'Ablation_For_Gary_Review': 'Ablation',
                    'ablation_gary_review': 'Ablation'}
    dicom_path = r'H:\Liver_Disease_Ablation\Dicom_Exports'
    reader = DicomReaderWriter(Contour_Names=('Ablation', 'Liver'), associations=associations,
                               arg_max=True, description='Ablation_Segmentation')
    reader.walk_through_folders(input_path=dicom_path)
    reader.which_indexes_lack_all_rois()
    reader.write_parallel(out_path=r'H:\Liver_Disease_Ablation\Train',
                          excel_file=os.path.join('.', 'Patient_Distribution.xlsx'))
    from Pre_Processing.Fix_Disease_Liver_Labeling import fix_labelins_of_LiTs
    fix_labelins_of_LiTs(path_base=r'H:\Liver_Disease_Ablation')
    xxx = 1

if separate_to_train_etc:
    from Pre_Processing.Separate_Into_Train_Validation_Test import separate_files
    path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
    separate_files(path)

if make_TF2_images:
    path = r'H:\Liver_Disease_Ablation'
    import Deep_Learning.Base_Deeplearning_Code.Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
    from Deep_Learning.Base_Deeplearning_Code.Image_Processors_Module.src.Processors.TFRecordWriter import parallel_record_writer, RecordWriter
    import numpy as np
    thread_count = 10
    cube_size = (32, 64, 64)
    power_val = 32
    image_processors_train = [
        Processors.LoadNifti(nifti_path_keys=('image_path', 'annotation_path'),
                             out_keys=('image_handle', 'annotation_handle')),
        Processors.ResampleSITKHandles(desired_output_spacing=(0.75, 0.75, 1.0),
                                       resample_interpolators=('Linear', 'Nearest'),
                                       resample_keys=('image_handle', 'annotation_handle')),
        Processors.NiftiToArray(nifti_keys=('image_handle', 'annotation_handle'),
                                out_keys=('image', 'annotation'), dtypes=('float32', 'int8')),
        Processors.NormalizeToAnnotation(image_key='image', annotation_key='annotation',
                                         annotation_value_list=(1, 2), mirror_max=True),
        Processors.ToCategorical(annotation_keys=('annotation',), num_classes=3),
        Processors.Split_Disease_Into_Cubes(cube_size=cube_size, disease_annotation=1, min_voxel_volume=300,
                                            max_voxels=1350000, image_key='image', annotation_key='annotation'),
        Processors.Distribute_into_3D(max_z=cube_size[0], max_rows=cube_size[1], max_cols=cube_size[2],
                                      min_z=cube_size[0])
    ]
    record_writer = RecordWriter(out_path=os.path.join(path, 'Records_1mm', 'Train_{}_Records'.format(cube_size[0])),
                                 file_name_key='out_name', rewrite=True)
    patient_dict_list = return_patient_dictionary_list(path=os.path.join(path, 'Train'))
    parallel_record_writer(dictionary_list=patient_dict_list, image_processors=image_processors_train,
                           max_records=np.inf, recordwriter=record_writer, debug=False)

    image_processors_validation = [
        Processors.LoadNifti(nifti_path_keys=('image_path', 'annotation_path'),
                             out_keys=('image_handle', 'annotation_handle')),
        Processors.ResampleSITKHandles(desired_output_spacing=(0.75, 0.75, 1.0),
                                       resample_interpolators=('Linear', 'Nearest'),
                                       resample_keys=('image_handle', 'annotation_handle')),
        Processors.NiftiToArray(nifti_keys=('image_handle', 'annotation_handle'),
                                out_keys=('image', 'annotation'), dtypes=('float32', 'int8')),
        Processors.NormalizeToAnnotation(image_key='image', annotation_key='annotation',
                                         annotation_value_list=(1, 2), mirror_max=True),
        Processors.ToCategorical(annotation_keys=('annotation',), num_classes=3),
        Processors.Box_Images(image_keys=('image',), annotation_key='annotation', wanted_vals_for_bbox=(1, 2),
                              bounding_box_expansion=(0, 0, 0), power_val_z=32, power_val_c=32, power_val_r=32)
    ]
    record_writer = RecordWriter(out_path=os.path.join(path, 'Records_1mm', 'Validation_Records'),
                                 file_name_key='out_name', rewrite=True)
    patient_dict_list = return_patient_dictionary_list(path=os.path.join(path, 'Validation'))
    parallel_record_writer(dictionary_list=patient_dict_list, image_processors=image_processors_validation,
                           max_records=np.inf, recordwriter=record_writer, debug=False)
