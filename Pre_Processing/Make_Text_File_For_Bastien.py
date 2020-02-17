__author__ = 'Brian M Anderson'
# Created on 2/15/2020
import os


def make_patient_list_file(base_path):
    fid = open('Patient_List.txt','w+')
    fid.write('MRN CT case\n')
    fid_shuffle = open('shuffle_id.txt','w+')
    fid_shuffle.write('MRN description\n')
    for folder in ['Train','Test','Validation']:
        file_path = os.path.join(base_path,folder)
        files = os.listdir(file_path)
        files = [i for i in files if i.find('Overall_mask') == 0]
        for file in files:
            data_file = file.replace('Overall_mask','Overall_Data').replace('_y','_')
            desc = data_file.split('Data_')[1].split('.nii')[0]
            fid.write('{} 1 CASE_1\n'.format(desc))
            fid_shuffle.write('{} {}\n'.format(desc, folder.upper()))
    fid_shuffle.close()
    fid.close()

base_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
make_patient_list_file(base_path)