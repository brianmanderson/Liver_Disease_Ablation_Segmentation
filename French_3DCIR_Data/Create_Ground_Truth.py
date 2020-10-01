__author__ = 'Brian M Anderson'
# Created on 8/31/2020

import os, pydicom, shutil
import SimpleITK as sitk
from Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter
from Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import numpy as np
import pandas as pd
from Segmentation_Evaluation_Tools.src.SegmentationEvaluationTools import identify_overlap_metrics


def add_dicom_tag(path):
    for file in os.listdir(path):
        if not file.endswith('.dcm') and not file.endswith('.txt'):
            os.rename(os.path.join(path, file), os.path.join(path, file + '.dcm'))
        # elif '.txt' in file:
        #     os.rename(os.path.join(path, file), os.path.join(path, file.split('.')[0] + '.txt'))
        # else:
        #     os.rename(os.path.join(path, file), os.path.join(path, file.split('.')[0] + '.' + file.split('.')[-1]))
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


def copy_predictions(prediction_out_path, image_path):
    for folder in os.listdir(prediction_out_path):
        if folder.startswith('3DCIR'):
            print(folder)
            patient_folder = os.path.join(image_path, 'Patient_{}'.format(folder.split('_')[-1]))
            down_folder = os.listdir(os.path.join(prediction_out_path, folder))[0]
            files = os.listdir(os.path.join(prediction_out_path, folder, down_folder))
            for file in files:
                print(file)
                if file.endswith('.dcm'):
                    shutil.copyfile(os.path.join(prediction_out_path, folder, down_folder, file),
                                    os.path.join(patient_folder, 'Prediction_RS.dcm'))
    return None


def create_predictions(prediction_path, image_path):
    for patient in os.listdir(image_path):
        print(patient)
        if os.path.exists(os.path.join(image_path, patient, 'Copied_Files.txt')):
            continue
        out_path = os.path.join(prediction_path.replace('Input_3', 'Output'), '3DCIR_{}'.format(patient))
        if os.path.exists(out_path):
            for file in os.listdir(out_path):
                os.remove(os.path.join(out_path, file))
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


def create_dicom_RT(path, out_path):
    for patient in os.listdir(path):
        print(patient)
        patient_id = patient.split('.')[-1]
        out_dir = os.path.join(out_path, 'Patient_{}'.format(patient_id))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif os.path.exists(os.path.join(out_dir, 'Completed.txt')):
            continue
        elif [i for i in os.listdir(out_dir) if i.startswith('RS_MRN')]:
            continue
        patient_path = os.path.join(path, patient)
        patient_dicom_path = os.path.join(patient_path, 'PATIENT_DICOM')
        add_dicom_tag(patient_dicom_path)
        add_FrameOfReferenceUID_patid(patient_dicom_path, patient_id)
        copy_files = [i for i in os.listdir(patient_dicom_path) if i.endswith('.dcm') and i not in os.listdir(out_dir)]
        for file in copy_files:
            shutil.copyfile(os.path.join(patient_dicom_path, file), os.path.join(out_dir, file))
        for folder in os.listdir(os.path.join(patient_path, 'MASKS_DICOM')):
            print(folder)
            add_dicom_tag(os.path.join(patient_path, 'MASKS_DICOM', folder))
            add_FrameOfReferenceUID_patid(os.path.join(patient_path, 'MASKS_DICOM', folder), patient_id)

        '''
        Now use the dicom readers to create the ground truth masks
        '''
        image_reader = DicomReaderWriter(get_images_mask=True)
        image_reader.Make_Contour_From_directory(patient_dicom_path)
        background = np.zeros(image_reader.ArrayDicom.shape)
        liver_reader = DicomReaderWriter(get_images_mask=True)
        liver_reader.Make_Contour_From_directory(os.path.join(patient_path, 'MASKS_DICOM', 'liver'))
        liver = liver_reader.ArrayDicom
        liver_tumor_folders = [i for i in os.listdir(os.path.join(patient_path, 'MASKS_DICOM'))
                               if i.startswith('livertumor') or i.startswith('metastasectomie')
                               or i.find('tumor') != -1]
        annotation_stack = [background]
        '''
        Combine all tumors into 'Disease'
        '''
        tumor = np.zeros(background.shape)
        for tumor_folder in liver_tumor_folders:
            tumor_reader = DicomReaderWriter(get_images_mask=True)
            tumor_reader.Make_Contour_From_directory(os.path.join(patient_path, 'MASKS_DICOM', tumor_folder))
            tumor += tumor_reader.ArrayDicom
        site_names = []
        if np.max(tumor) > 0:
            site_names.append('Disease')
            annotation_stack.append(tumor > 0)
        site_names.append('Liver')
        annotation_stack.append(liver + tumor > 0)
        for label in os.listdir(os.path.join(patient_path, 'MASKS_DICOM')):
            print(label)
            continue
            if label.startswith('liver'):
                continue
            site_names.append(label)
            site = np.zeros(image_reader.ArrayDicom.shape)
            site_reader = DicomReaderWriter(get_images_mask=True)
            site_reader.Make_Contour_From_directory(os.path.join(patient_path, 'MASKS_DICOM', label))
            site += site_reader.ArrayDicom
            annotation_stack.append(site > 0)
        '''
        Write out the RT structure
        '''
        background = np.stack(annotation_stack, axis=-1)
        image_reader.with_annotations(background, output_dir=out_dir, ROI_Names=site_names)


def compare_predictions(path, out_path):
    out_dict = {'Patient_ID':[]}
    pred_stack = []
    truth_stack = []
    for patient in os.listdir(path):
        if patient in ['Patient_5', 'Patient_11', 'Patient_12', 'Patient_18', 'Patient_20']:  # Exclusion criteria
            continue
        print(patient)
        if os.path.exists(os.path.join(out_path, '{}_Image.nii'.format(patient))):
            prediction_handle = sitk.ReadImage(os.path.join(out_path, '{}_Prediction.nii'.format(patient)))
            truth_handle = sitk.ReadImage(os.path.join(out_path, '{}_Truth.nii'.format(patient)))
        else:
            patient_path = os.path.join(path, patient)
            dicom_reader = DicomReaderWriter(get_images_mask=True, arg_max=False,
                                             Contour_Names=['Disease', 'Liver_Disease_Ablation_BMA_Program_0'])
            dicom_reader.Make_Contour_From_directory(patient_path)
            truth_handle = sitk.GetImageFromArray(dicom_reader.mask[..., 1])
            prediction_handle = sitk.GetImageFromArray(dicom_reader.mask[..., -1])
            for handle in [truth_handle, prediction_handle]:
                handle.SetSpacing(dicom_reader.dicom_handle.GetSpacing())
                handle.SetDirection(dicom_reader.dicom_handle.GetDirection())
                handle.SetOrigin(dicom_reader.dicom_handle.GetOrigin())
            sitk.WriteImage(dicom_reader.dicom_handle, os.path.join(out_path, '{}_Image.nii'.format(patient)))
            sitk.WriteImage(prediction_handle, os.path.join(out_path, '{}_Prediction.nii'.format(patient)))
            sitk.WriteImage(truth_handle, os.path.join(out_path, '{}_Truth.nii'.format(patient)))
        pred_stack.append(sitk.GetArrayFromImage(prediction_handle))
        truth_stack.append(sitk.GetArrayFromImage(truth_handle))
        pat_data = identify_overlap_metrics(prediction_handle=prediction_handle, truth_handle=truth_handle,
                                            perform_distance_measures=True)
        for key in pat_data.keys():
            if key not in out_dict:
                out_dict[key] = []
            out_dict[key].append(pat_data[key])
        out_dict['Patient_ID'].append(patient)
    print('Mean was {}'.format(np.mean(out_dict['dice'])))
    combined_pred = np.concatenate(pred_stack, axis=0)
    combined_truth = np.concatenate(truth_stack, axis=0)
    combined_pred_handle = sitk.GetImageFromArray(combined_pred)
    combined_truth_handle = sitk.GetImageFromArray(combined_truth)
    combined_data = identify_overlap_metrics(prediction_handle=combined_pred_handle, truth_handle=combined_truth_handle,
                                             perform_distance_measures=False)
    for key in combined_data.keys():
        if key not in out_dict:
            out_dict[key] = []
        out_dict[key].append(combined_data[key])
    out_dict['Patient_ID'].append('Combined')
    for key in out_dict.keys():
        if len(out_dict[key]) < len(out_dict['dice']):
            out_dict[key].append('-')
    df = pd.DataFrame(out_dict)
    df.to_excel(os.path.join('.', 'PatientResults.xlsx'), index=0)
    return None


if __name__ == '__main__':
    pass
