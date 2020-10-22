__author__ = 'Brian M Anderson'
# Created on 9/24/2020


from Pre_Processing.Nifti_To_Dicom.Conversion_Definition import convert_niftii_to_dicom
import os
from Pre_Processing.Nifti_To_Dicom.Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import \
    Dicom_to_Imagestack, sitk, plot_scroll_Image
from threading import Thread
from multiprocessing import cpu_count
from queue import *


def write_nii_gz(file):
    print(file)
    handle = sitk.ReadImage(file)
    sitk.WriteImage(handle, file + '.gz')
    return None


def worker_def(A):
    q = A[0]
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                write_nii_gz(**item)
            except:
                print('failed?')
            q.task_done()


def convert_nii_to_dicom():
    base_path = r'H:\Liver_Disease_Ablation\LiTs_Test\Nifti'
    out_path_base = r'H:\Liver_Disease_Ablation\LiTs_Test\Dicom'

    files = [i for i in os.listdir(base_path) if i.endswith('.nii') and not i.endswith('.nii.gz')]
    for file in files:
        pat_id = file.split('-')[-1].split('.nii')[0]
        desc = 'Lits_Test_{}'.format(pat_id)
        out_path = os.path.join(out_path_base, desc)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.listdir(out_path):
            print(file)
            convert_niftii_to_dicom(os.path.join(base_path, file), out_path, patient_name=desc,
                                    patient_id=desc, is_structure=False)
    return None


if __name__ == '__main__':
    pass
