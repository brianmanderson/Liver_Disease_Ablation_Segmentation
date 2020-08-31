__author__ = 'Brian M Anderson'
# Created on 8/31/2020

import os, pydicom
import SimpleITK as sitk
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack
from Dicom_RT_and_Images_to_Mask.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import numpy as np


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


def main():
    path = r'C:\Users\bmanderson\Downloads\Unzipped\3Dircadb1'
    out_path = r'H:\Liver_Disease_Ablation\3Dircadb1\Fixed_Patients'
    for patient in os.listdir(path):
        patient_id = patient.split('.')[-1]
        patient_path = os.path.join(path, patient)
        patient_dicom_path = os.path.join(patient_path, 'PATIENT_DICOM')
        # add_dicom_tag(patient_dicom_path)
        add_FrameOfReferenceUID_patid(patient_dicom_path, patient_id)
        image_reader = Dicom_to_Imagestack(get_images_mask=True)
        image_reader.Make_Contour_From_directory(patient_dicom_path)
        for folder in os.listdir(os.path.join(patient_path, 'MASKS_DICOM')):
            print(folder)
            # add_dicom_tag(os.path.join(patient_path, 'MASKS_DICOM', folder))
            add_FrameOfReferenceUID_patid(os.path.join(patient_path, 'MASKS_DICOM', folder), patient_id)
        liver_reader = Dicom_to_Imagestack(get_images_mask=True)
        liver_reader.Make_Contour_From_directory(os.path.join(patient_path, 'MASKS_DICOM', 'liver'))
        liver = liver_reader.ArrayDicom
        liver = np.stack(liver, axis=-1)


if __name__ == '__main__':
    main()
