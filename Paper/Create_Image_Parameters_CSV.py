__author__ = 'Brian M Anderson'
# Created on 7/21/2020

import os
import SimpleITK as sitk

images_path = r'H:\Data\Local_Recurrence_Exports'
reader = sitk.ImageSeriesReader()
reader.MetaDataDictionaryArrayUpdateOn()
reader.LoadPrivateTagsOn()
fid = open(r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Image_parameters.csv', 'w+')
fid.write('MRN,Slice Thickness,Pixel_X,Pixel_Y\n')
for patient_path, dirs, files in os.walk(images_path):
    files = [os.path.join(patient_path, i) for i in files if i.endswith('.dcm') and i.startswith('CT')]
    if files:
        reader.SetFileNames(files[:2])
        dicom_handle = reader.Execute()
        slice_thickness = float(reader.GetMetaData(0, "0018|0050"))
        pixel_spacing = [float(i) for i in reader.GetMetaData(0, "0028|0030").split('\\')]
        MRN = reader.GetMetaData(0, "0010|0020")
        fid.write('{},{},{},{}\n'.format(MRN, slice_thickness, pixel_spacing[0], pixel_spacing[1]))
fid.close()