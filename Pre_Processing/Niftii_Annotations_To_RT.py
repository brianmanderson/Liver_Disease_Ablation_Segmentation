__author__ = 'Brian M Anderson'
# Created on 2/14/2020
# With a large amount of help from Bastien githut.com/guatavita
from Nifti_To_Dicom.Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, sitk, os
from Nifti_To_Dicom.Dicom_RT_and_Images_to_Mask.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import time
from tensorflow.python.keras.utils.np_utils import to_categorical

base_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
out_path_base = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Nifti_to_dicom\LiTs'

Dicom_reader = Dicom_to_Imagestack()
done = False
while not done:
    done = True
    for folder in ['Train','Test','Validation']:
        file_path = os.path.join(base_path,folder)
        files = os.listdir(file_path)
        files = [i for i in files if i.find('Overall_mask') == 0]
        for file in files:
            data_file = file.replace('Overall_mask','Overall_Data').replace('_y','_')
            desc = data_file.split('Data_')[1].split('.nii')[0]
            out_path = os.path.join(out_path_base, desc,'Images')
            if os.path.exists(os.path.join(out_path,'Completed.txt')) and not os.path.exists(os.path.join(out_path_base,desc,'RT')):
                print(desc)
                Dicom_reader.make_array(out_path)
                annotations = sitk.ReadImage(os.path.join(file_path,file))
                annotations = sitk.GetArrayFromImage(annotations)
                annotations = to_categorical(annotations)
                # annotations[...,1] += annotations[...,2]
                Dicom_reader.with_annotations(annotations,os.path.join(out_path_base,desc,'RT'),ROI_Names=['Liver_San_Disease','Disease'])
                xxx = 1
            elif os.path.exists(os.path.join(out_path_base,desc,'RT')):
                continue
            else:
                done = False
    print('Finished a lap...')
    time.sleep(20)