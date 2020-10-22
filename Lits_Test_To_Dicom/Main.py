__author__ = 'Brian M Anderson'
# Created on 9/8/2020

'''
Obtain image parameters
'''
obtain_parameters = False
if obtain_parameters:
    from Obtain_Image_Acquisition_Parameters import obtain_image_parameters
    obtain_image_parameters()

'''
First, convert the nii files to dicom
'''
convert_lits_to_dicom = False
if convert_lits_to_dicom:
    from convert_lits_nii_to_dicom import *
    convert_nii_to_dicom()

'''
Upload the dicom files to Raystation for viewing and liver/disease prediction
'''


'''
Next, create predictions for the liver from the dicom files
'''

#  Run Create_Liver_Contours
'''
Edit the liver contours as needed, ensure that potential disease is included as 'liver'
'''

'''
Some files do not have the correct orientation...
'''
correct_orientation = False
if correct_orientation:
    from Assign_correct_orientation import assign_orientation
    assign_orientation()

'''
Create disease contours
'''
#  Run Create_Disease_Contours

'''
Export predictions as .mhd files or RS structures...
'''
#  Run Export_Predictions_as_mhd
#  Run Export_Predictions_as_RS
'''
Convert Prediction .mhd files to .nii files for submission
'''
predictions_to_nii = True
if predictions_to_nii:
    from Convert_predictions_to_nii import mhd_to_nii, RS_to_nii
    # mhd_to_nii()
    RS_to_nii()

zip_predictions = False
if zip_predictions:
    from Zip_Results import zip_files
    zip_files()