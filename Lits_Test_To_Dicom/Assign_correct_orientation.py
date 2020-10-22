__author__ = 'Brian M Anderson'
# Created on 9/22/2020
import pydicom, os
from Pre_Processing.Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter, plot_scroll_Image


def assign_orientation():
    path = r'H:\Liver_Disease_Ablation\LiTs_Test\Dicom\Lits_Test_{}'
    patient_dict = {}
    # patient_dict[0] = 'FFS'
    # patient_dict[53] = 'FFS'
    patient_dict[57] = 'FFS'
    patient_dict[59] = 'FFS'
    reader = DicomReaderWriter()
    for patient in patient_dict.keys():
        pat_path = path.format(patient)
        print(pat_path)
        reader.down_folder(pat_path)
        files = [i for i in os.listdir(pat_path) if i.startswith('image') and i.endswith('.dcm')]
        for file in files:
            ds = pydicom.read_file(os.path.join(pat_path, file))
            ds.PatientPosition = patient_dict[patient]
            pydicom.write_file(os.path.join(pat_path, file), ds)
    return None


if __name__ == '__main__':
    pass
