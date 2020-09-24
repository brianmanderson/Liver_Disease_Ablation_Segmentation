__author__ = 'Brian M Anderson'
# Created on 9/8/2020


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
Export predictions as .mhd files
'''
#  Run Export_Predictions_as_mhd

'''
Convert Prediction .mhd files to .nii files for submission
'''
convert_mhd_to_nii = True
if convert_mhd_to_nii:
    from Convert_mhd_to_nii import mhd_to_nii
    mhd_to_nii()