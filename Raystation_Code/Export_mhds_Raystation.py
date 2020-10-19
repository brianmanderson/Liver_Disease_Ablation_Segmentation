__author__ = 'Brian M Anderson'
# Created on 6/26/2020

from connect import *
import os, time


patient_db = get_current('PatientDB')
patient = get_current('Patient')
case = get_current('Case')
exam = get_current('Examination')
rois_in_case = []
for roi in case.PatientModel.RegionsOfInterest:
    rois_in_case.append(roi.Name)
path = r'H:\Single_Site\Ablation'
path = os.path.join(path, patient.PatientID, case.CaseName, exam.Name)
if not os.path.exists(path):
    os.makedirs(path)
for roi in rois_in_case:
    if roi in ['Ablation','GTV','Liver_Disease_Ablation_BMA_Program_0', 'Liver', 'Liver_BMA_Program_4']: #
        if case.PatientModel.StructureSets[exam.Name].RoiGeometries[roi].HasContours():
            case.PatientModel.StructureSets[exam.Name].RoiGeometries[roi].ExportRoiGeometryAsMetaImage(
                MetaFileName=os.path.join(path,'{}.mhd'.format(roi)), AsExamination=True)
fid = open(os.path.join(path,'Completed_Export.txt'),'w+')
fid.close()