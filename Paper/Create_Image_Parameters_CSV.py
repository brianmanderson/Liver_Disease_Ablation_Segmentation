__author__ = 'Brian M Anderson'
# Created on 7/21/2020

import os
import pandas as pd
import SimpleITK as sitk


def identify_image_parameters(images_path=r'H:\Data\Local_Recurrence_Exports',
                              xlsx_path=r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Image_parameters.xlsx'):
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    out_dict = {'MRN':[], 'Slice Thickness': [], 'Pixel_X': [], 'Pixel_Y': []}
    for patient_path, dirs, files in os.walk(images_path):
        files = [os.path.join(patient_path, i) for i in files if i.endswith('.dcm') and
                 (i.startswith('CT') or i.startswith('image'))]
        if files:
            print(patient_path)
            reader.SetFileNames(files[:2])
            dicom_handle = reader.Execute()
            slice_thickness = float(reader.GetMetaData(0, "0018|0050"))
            pixel_spacing = [float(i) for i in reader.GetMetaData(0, "0028|0030").split('\\')]
            MRN = reader.GetMetaData(0, "0010|0020").strip(' ')
            if MRN in out_dict['MRN']:
                continue
            out_dict['MRN'].append(MRN)
            out_dict['Slice Thickness'].append(slice_thickness)
            out_dict['Pixel_X'].append(pixel_spacing[0])
            out_dict['Pixel_Y'].append(pixel_spacing[1])
    df = pd.DataFrame(out_dict)
    columns = ['MRN', 'Slice Thickness', 'Pixel_X', 'Pixel_Y']
    df[columns].to_excel(xlsx_path, index=0)
    return None


if __name__ == '__main__':
    pass
