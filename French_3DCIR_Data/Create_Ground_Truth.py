__author__ = 'Brian M Anderson'
# Created on 8/31/2020

import os, pydicom, shutil
import SimpleITK as sitk
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack
from Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import numpy as np
import pandas as pd


def add_dicom_tag(path):
    for file in os.listdir(path):
        if not file.endswith('.dcm') and not file.endswith('.txt'):
            os.rename(os.path.join(path, file), os.path.join(path, file + '.dcm'))
        elif '.txt' in file:
            os.rename(os.path.join(path, file), os.path.join(path, file.split('.')[0] + '.txt'))
        else:
            os.rename(os.path.join(path, file), os.path.join(path, file.split('.')[0] + '.' + file.split('.')[-1]))
    return None


def add_FrameOfReferenceUID_patid(path, patient_id):
    if not os.path.exists(os.path.join(path, 'completed_frame_id_position.txt')):
        for file in os.listdir(path):
            if file.endswith('.dcm'):
                ds = pydicom.read_file(os.path.join(path, file))
                ds.FrameOfReferenceUID = '1.0.1'
                ds.PatientPosition = 'HFS'
                ds.PatientID = '3DCIR_Patient_{}'.format(patient_id)
                pydicom.dcmwrite(os.path.join(path, file), ds)
        fid = open(os.path.join(path, 'completed_frame_id_position.txt'), 'w+')
        fid.close()
    return None


def copy_predictions():
    prediction_out_path = r'L:\Clinical\Auto_Contour_Sites\Liver_Disease_Ablation_Auto_Contour\Output'
    image_path = r'H:\Liver_Disease_Ablation\3Dircadb1\Fixed_Patients'
    for folder in os.listdir(prediction_out_path):
        if folder.startswith('3DCIR'):
            patient_folder = os.path.join(image_path, 'Patient_{}'.format(folder.split('_')[-1]))
            down_folder = os.listdir(os.path.join(prediction_out_path, folder))[0]
            files = os.listdir(os.path.join(prediction_out_path, folder, down_folder))
            for file in files:
                if file.endswith('.dcm'):
                    shutil.copyfile(os.path.join(prediction_out_path, folder, down_folder, file),
                                    os.path.join(patient_folder, file))
    return None


def create_predictions():
    prediction_path = r'L:\Clinical\Auto_Contour_Sites\Liver_Disease_Ablation_Auto_Contour\Input_3'
    image_path = r'H:\Liver_Disease_Ablation\3Dircadb1\Fixed_Patients'
    for patient in os.listdir(image_path):
        print(patient)
        if os.path.exists(os.path.join(image_path, patient, 'Copied_Files.txt')):
            continue
        if not os.path.exists(os.path.join(prediction_path, patient)):
            os.makedirs(os.path.join(prediction_path, patient))
        for file in os.listdir(os.path.join(image_path, patient)):
            if file.endswith('.dcm'):
                shutil.copyfile(os.path.join(image_path, patient, file), os.path.join(prediction_path, patient, file))
        fid = open(os.path.join(prediction_path, patient, 'Completed.txt'), 'w+')
        fid.close()
        fid = open(os.path.join(image_path, patient, 'Copied_Files.txt'), 'w+')
        fid.close()
    return None


def create_dicom_RT():
    path = r'C:\Users\bmanderson\Downloads\Unzipped\3Dircadb1'
    out_path = r'H:\Liver_Disease_Ablation\3Dircadb1\Fixed_Patients'
    for patient in os.listdir(path):
        print(patient)
        patient_id = patient.split('.')[-1]
        out_dir = os.path.join(out_path, 'Patient_{}'.format(patient_id))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif os.path.exists(os.path.join(out_dir, 'Completed.txt')):
            continue
        else:
            continue
        patient_path = os.path.join(path, patient)
        patient_dicom_path = os.path.join(patient_path, 'PATIENT_DICOM')
        copy_files = [i for i in os.listdir(patient_dicom_path) if i.endswith('.dcm') and i not in os.listdir(out_dir)]
        for file in copy_files:
            shutil.copyfile(os.path.join(patient_dicom_path, file), os.path.join(out_dir, file))
        # add_dicom_tag(patient_dicom_path)
        add_FrameOfReferenceUID_patid(patient_dicom_path, patient_id)
        for folder in os.listdir(os.path.join(patient_path, 'MASKS_DICOM')):
            print(folder)
            # add_dicom_tag(os.path.join(patient_path, 'MASKS_DICOM', folder))
            add_FrameOfReferenceUID_patid(os.path.join(patient_path, 'MASKS_DICOM', folder), patient_id)

        '''
        Now use the dicom readers to create the ground truth masks
        '''
        image_reader = Dicom_to_Imagestack(get_images_mask=True)
        image_reader.Make_Contour_From_directory(patient_dicom_path)
        background = np.zeros(image_reader.ArrayDicom.shape)
        liver_reader = Dicom_to_Imagestack(get_images_mask=True)
        liver_reader.Make_Contour_From_directory(os.path.join(patient_path, 'MASKS_DICOM', 'liver'))
        liver = liver_reader.ArrayDicom
        liver_tumor_folders = [i for i in os.listdir(os.path.join(patient_path, 'MASKS_DICOM'))
                               if i.startswith('livertumor') or i.startswith('metastasectomie')]
        annotation_stack = [background]
        '''
        Combine all tumors into 'Disease'
        '''
        tumor = np.zeros(background.shape)
        for tumor_folder in liver_tumor_folders:
            tumor_reader = Dicom_to_Imagestack(get_images_mask=True)
            tumor_reader.Make_Contour_From_directory(os.path.join(patient_path, 'MASKS_DICOM', tumor_folder))
            tumor += tumor_reader.ArrayDicom
        site_names = ['Disease', 'Liver']
        annotation_stack.append(tumor > 0)
        annotation_stack.append(liver + tumor > 0)
        for label in os.listdir(os.path.join(patient_path, 'MASKS_DICOM')):
            print(label)
            if label.startswith('liver'):
                continue
            site_names.append(label)
            site = np.zeros(image_reader.ArrayDicom.shape)
            site_reader = Dicom_to_Imagestack(get_images_mask=True)
            site_reader.Make_Contour_From_directory(os.path.join(patient_path, 'MASKS_DICOM', label))
            site += site_reader.ArrayDicom
            annotation_stack.append(site > 0)
        '''
        Write out the RT structure
        '''
        background = np.stack(annotation_stack, axis=-1)
        image_reader.with_annotations(background, output_dir=out_dir, ROI_Names=site_names)


def compare_predictions():
    path = r'H:\Liver_Disease_Ablation\3Dircadb1\Fixed_Patients'
    out_dict = {'Patient_ID':[], 'DSC': []}
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    for patient in os.listdir(path):
        print(patient)
        patient_path = os.path.join(path, patient)
        dicom_reader = Dicom_to_Imagestack(get_images_mask=True, arg_max=False,
                                           Contour_Names=['Disease', 'Liver_Disease_Ablation_BMA_Program_0'])
        dicom_reader.Make_Contour_From_directory(patient_path)
        truth = sitk.GetImageFromArray(dicom_reader.mask[..., 1])
        prediction = sitk.GetImageFromArray(dicom_reader.mask[..., -1])
        for handle in [truth, prediction]:
            handle.SetSpacing(dicom_reader.dicom_handle.GetSpacing())
            handle.SetDirection(dicom_reader.dicom_handle.GetDirection())
            handle.SetOrigin(dicom_reader.dicom_handle.GetOrigin())
        overlap_measures_filter.Execute(truth, prediction)
        dsc = overlap_measures_filter.GetDiceCoefficient()
        print('DSC was {} for {}'.format(dsc, patient))
        out_dict['DSC'].append(dsc)
        out_dict['Patient_ID'].append(patient)
    print('Mean was {}'.format(np.mean(out_dict['DSC'])))
    df = pd.DataFrame(out_dict)
    df.to_excel(os.path.join('.', 'Dice_Results.xlsx'), index=0)
    return None


if __name__ == '__main__':
    create_dicom_RT()
    # create_predictions()
    # copy_predictions()
    # compare_predictions()
    '''
    Load into Raystation for viewing
    '''
